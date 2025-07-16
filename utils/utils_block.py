from collections import OrderedDict

from einops import rearrange
from torch import nn
from torch.nn import functional
from torch.nn import functional as F
import numpy as np
import torch
from typing import Optional


####################
# Basic blocks

def Filter(x: np.array, kernel: np.array, in_channels: Optional[int] = 1, out_channels: Optional[int] = 1,
           pad_type: Optional[str] = 'reflect', dilation: Optional[int] = 1, groups: Optional[int] = 1) -> torch.Tensor:
    """
        filter conv function
        x:np.array
        return [B,C,H,W]
    """
    device = x.device
    kernel_size = kernel.shape
    kernel = torch.FloatTensor(np.ascontiguousarray(kernel)).expand(out_channels, in_channels // groups, kernel_size[0],
                                                                    kernel_size[1]).to(
        device)
    weight = nn.Parameter(data=kernel, requires_grad=False).to(device)

    padding = get_same_padding(kernel_size, dilation)
    pad_x = F.pad(x, pad=padding, mode=pad_type)
    return F.conv2d(pad_x, weight)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FilterConv(nn.Module):
    def __init__(self, kernel:np.array, in_channels:Optional[int]=1, out_channels:Optional[int]=1, pad_type:Optional[str]='reflect', dilation:Optional[int]=1, groups:Optional[int]=1):
        super(FilterConv, self).__init__()
        self.kernel_size = kernel.shape
        self.kernel = torch.FloatTensor(np.ascontiguousarray(kernel)).expand(out_channels, in_channels // groups,
                                                                             self.kernel_size[0], self.kernel_size[1])

        self.weight = nn.Parameter(data=self.kernel, requires_grad=False)
        self.pad_type = pad_type
        self.dilation = dilation

    def forward(self, x):
        padding = get_same_padding(self.kernel_size, self.dilation)
        pad_x = F.pad(x, pad=padding, mode=self.pad_type)
        return F.conv2d(pad_x, self.weight)


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks=2):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).cuda()
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).cuda()
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)
        return x


def volume2frame(volume):
    b, g, c, k = volume.shape
    ans_volume = volume.contiguous().view(-1, c, k).unsqueeze(1)
    return ans_volume


def get_gradient(x: torch.Tensor) -> torch.Tensor:
    """

    :param x: Tensor B,C,H,W
    :return: grad
    """
    x = x.float()
    kernel_v = [[0, -1, 0],
                [0, 0, 0],
                [0, 1, 0]]
    kernel_h = [[0, 0, 0],
                [-1, 0, 1],
                [0, 0, 0]]
    device = x.device
    kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
    kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
    weight_h = nn.Parameter(data=kernel_h, requires_grad=False).to(device)

    weight_v = nn.Parameter(data=kernel_v, requires_grad=False).to(device)
    x_list = []
    for i in range(x.shape[1]):
        x_i = x[:, i]
        x_i_v = functional.conv2d(x_i.unsqueeze(1), weight_v, padding=1)
        x_i_h = functional.conv2d(x_i.unsqueeze(1), weight_h, padding=1)
        x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        x_list.append(x_i)

    x_out = torch.cat(x_list, dim=1)
    return x_out


def down_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_channels = int(int(channels * scale_factor) * scale_factor)
    out_height = int(in_height / scale_factor)
    out_width = int(in_width / scale_factor)

    block_size = int(1 / scale_factor)
    input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'gelu':
        layer = nn.GELU()
    elif act_type == 'sigmoid':
        layer = nn.Sigmoid()
    elif act_type == 'softmax':
        layer = nn.Softmax()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def get_same_padding(kernel_size, dilation=1):
    """
        The input and output is same and stride = 1
        return tuple(hpad,bpad,lpad,rpad)
    """
    if type(kernel_size) is int:
        kernel_size = (kernel_size, kernel_size)

    kernel_size = [i + (i - 1) * (dilation - 1) for i in kernel_size]
    needed_pad_height, needed_pad_width = [(i - 1) for i in kernel_size]
    t_pad = needed_pad_height // 2
    b_pad = needed_pad_height - t_pad
    l_pad = needed_pad_width // 2
    r_pad = needed_pad_width - l_pad
    return (l_pad, r_pad, t_pad, b_pad)


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size=1, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='gelu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
                  dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ReflectionPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, K], K is the total number of such blocks


####################
# Useful blocks
####################

class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
                 bias=True, pad_type='zero', norm_type=None, act_type='gelu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
                           norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
                           norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


####################
# Upsampler
####################

class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale=2, kernel_size=3, stride=1, bias=True, pad_type='zero',
                 norm_type=None, act_type='gelu'):
        super(PixelShuffleBlock, self).__init__()
        self.conv = conv_block(in_channel, out_channel * (scale ** 2), kernel_size=kernel_size, stride=stride,
                               bias=bias, pad_type=pad_type, norm_type=None, act_type=None)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.norm = norm(norm_type, out_channel) if norm_type else nn.Identity()
        self.act = act(act_type) if act_type else nn.Identity()

    def forward(self, x0):
        """
            Pixel shuffle layer
            (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
            Neural Network, CVPR17)
        """
        x1 = self.conv(x0)
        x2 = self.pixel_shuffle(x1)
        x3 = self.norm(x2)
        x4 = self.act(x3)
        return x4


class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale=2, kernel_size=3, stride=1, bias=True, pad_type='zero',
                 norm_type=None, act_type='relu', mode='nearest'):
        # Up conv
        # described in https://distill.pub/2016/deconv-checkerboard/
        self.upsample = nn.Upsample(scale_factor=scale, mode=mode)
        self.conv = conv_block(in_channel, out_channel, kernel_size, stride, bias=bias, pad_type=pad_type,
                               norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        return self.conv(self.upsample(x))


def patch_embed(x):
    x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
    return x


def patch_unembed(x, x_size):
    B, HW, C = x.shape
    x = x.transpose(1, 2).view(B, C, x_size[0], x_size[1])  # B Ph*Pw C
    return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

import torch
from torch import nn as nn


class Mlp(nn.Module):
    # two mlp, fc-relu-drop-fc-relu-drop
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), )

    def forward(self, x):
        return self.op(x)


class LAM_Module(nn.Module):
    """ Layer attention module"""

    def __init__(self, visual=False):
        super(LAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.visual = visual

    def forward(self, x):
        batchsize, head, num, dim = x.size()
        x = x.permute(0, 3, 1, 2)
        proj_query = x.view(batchsize, dim, -1)
        proj_key = x.view(batchsize, dim, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = energy - torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        attention = self.softmax(energy)
        proj_value = x.view(batchsize, dim, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(batchsize, dim, head, num)
        out = self.gamma * out + x
        out = out.permute(0, 2, 3, 1)
        return out



if __name__ == '__main__':
    pix = PixelShuffleBlock(in_channel=32, out_channel=1, scale=2, kernel_size=3, stride=1, bias=True, act_type='gelu',
                            norm_type=None, )
    x0 = torch.randn((1, 32, 120, 120))
    print(pix)
    res = pix(x0)
    print(res.shape)
