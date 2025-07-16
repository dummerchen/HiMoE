import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
# from eniops import rearrange

class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            r: int = 0,
            lora_alpha: int = 1,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            enable_lora: List[bool] = [False],
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0),
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result


class PromptMergedLinear(nn.Linear, LoRALayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: float = 1,
            lora_dropout: float = 0.,
            enable_lora: List[bool] = [False, True, True],
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        # Actual trainable parameters
        if any(enable_lora):
            self.scaling = self.lora_alpha
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)

    def zero_pad(self, x, nw):
        if len(x.shape) == 2:
            # 360,180
            result = x.new_zeros((len(self.lora_ind), x.shape[1]))
            result[self.lora_ind, ...] = x
        else:
            # B,360,180
            result = x.new_zeros((x.shape[0], len(self.lora_ind), x.shape[2]))
            result[:, self.lora_ind, ...] = x
            result = torch.repeat_interleave(result, repeats=nw, dim=0)

        return result

    # bak
    def merge_AB(self, pa, pb, nb):
        # (B) 2*E D, D*2 E or E*2 D , D*2 E
        # B 36 180 , 360 18 1 => T(1 360 180)
        if len(pb.shape) == len(pa.shape):
            if len(pa.shape) == 2:
                # ED,DE => 1 1 E*2 D, 1 D*2 E 1
                pa = pa.unsqueeze(0).unsqueeze(1)
                pb = pb.unsqueeze(0).unsqueeze(-1)
            elif len(pa.shape) == 3:
                pa = pa.unsqueeze(1)
                pb = pb.unsqueeze(0).unsqueeze(-1)
        delta_w = self.group_conv1d(pa, pb, groups=sum(self.enable_lora))
        return self.zero_pad(delta_w, nw=nb // pa.shape[0])


    def group_conv1d(self, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=2):
        """
        实现与 F.conv1d 等价的分组卷积操作，但使用矩阵乘法更灵活

        参数:
        - input: 输入张量，形状为 (batch_size, t, in_channels, seq_len)
        - weight: 卷积核张量，形状为 (t, out_channels, channels_per_group, kernel_size)
        - bias: 偏置张量，默认为 None
        - stride: 步幅，默认为 1
        - padding: 填充，默认为 0
        - dilation: 膨胀，默认为 1
        - groups: 分组数，默认为 1

        返回:
        - output: 输出张量，形状为 (batch_size, out_channels, output_len)
        """
        batch_size, t, in_channels, seq_len = input.shape
        t, out_channels, channels_per_group, kernel_size = weight.shape
        output_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        assert input.shape != weight.shape
        # 处理 padding
        if padding > 0:
            input = F.pad(input, (padding, padding))

        # 将输入和权重 reshape 以便于矩阵乘法
        input = input.view(batch_size, t, groups, channels_per_group, seq_len)
        weight = weight.view(t, groups, out_channels // groups, channels_per_group)

        # output = torch.zeros(batch_size, t, out_channels, output_len, device=input.device)
        # # 执行矩阵乘法
        # for g in range(groups):
        #     input_group = input[:, :, g, :, :]  # (batch_size, channels_per_group, seq_len)
        #     weight_group = weight[:, g].unsqueeze(0).expand(batch_size, -1, -1, -1) # (out_channels // groups, channels_per_group)
        #     output[:, :, g * (out_channels // groups):(g + 1) * (out_channels // groups), :] = weight_group @ input_group
        # 将输出 reshape 为 (batch_size, out_channels, output_len)
        # output = output.view(batch_size, t, out_channels, output_len)
        output = torch.einsum('btgci,tgoc->btgoi', input, weight)
        output = output.reshape(batch_size, t, out_channels, output_len)
        # 将输出 reshape 为 (batch_size, out_channels, output_len)
        # output_manual = output_reshaped.reshape(batch_size, t, out_channels, seq_len)
        # print(torch.allclose(tmp, output))
        # 处理 stride
        if stride > 1:
            output = output[:, :, :, ::stride]

        # 加上 bias
        if bias is not None:
            output += bias.view(1, 1, -1, 1)

        return torch.sum(output, dim=1)

    def forward(self, x: torch.Tensor, x_querry=None, l: str = 0, pid=0, prompt=None):

        result = F.linear(x, self.weight, bias=self.bias)
        if prompt is not None and x_querry is not None:
            for pid in range(prompt.pnum):
                if int(l[0]) in prompt.prompt_layers[pid] and int(l[1]) in prompt.prompt_block[pid]:
                    [pa, pb] = prompt.forward(x_querry, str(l), pid)
                    tmp = self.merge_AB(pa, pb, nb=x.shape[0])
                    if len(tmp.shape) == 2:
                        prompt_result = torch.einsum('b l c, d c -> b l d', self.lora_dropout(x), tmp) / pa.shape[-2]/2
                    else:
                        prompt_result = torch.einsum('b l c, b d c -> b l d', self.lora_dropout(x), tmp) / pa.shape[-2]/2
                    result = result + prompt_result
        return result


class PromptMergedLinear_wolora(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            # r: int = 0,
            # lora_alpha: int = 1,
            # lora_dropout: float = 0.,
            enable_lora: List[bool] = [False, True, True],
            # merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        #                    merge_weights=merge_weights)
        # assert out_features % len(enable_lora) == 0, \
        #     'The length of enable_lora must divide out_features'
        # self.enable_lora = enable_lora
        # # Actual trainable parameters
        # if any(enable_lora):
        #     self.scaling = self.lora_alpha
        #     self.weight.requires_grad = False
        self.lora_ind = self.weight.new_zeros(
            (out_features,), dtype=torch.bool
        ).view(len(enable_lora), -1)
        self.lora_ind[enable_lora, :] = True
        self.lora_ind = self.lora_ind.view(-1)

    def zero_pad(self, x, nw):
        if len(x.shape) == 2:
            # 360,180
            result = x.new_zeros((len(self.lora_ind), x.shape[1]))
            result[self.lora_ind, ...] = x
        else:
            # B,360,180
            result = x.new_zeros((x.shape[0], len(self.lora_ind), x.shape[2]))
            result[:, self.lora_ind, ...] = x
            result = torch.repeat_interleave(result, repeats=nw, dim=0)
        return result

    def forward(self, x: torch.Tensor, x_querry=None, l: str = 0, pid=0, prompt=None):

        result = F.linear(x, self.weight, bias=self.bias)
        if prompt is not None and x_querry is not None:
            for pid in range(prompt.pnum):
                if int(l[0]) in prompt.prompt_layers[pid] and int(l[1]) in prompt.prompt_block[pid]:
                    [w] = prompt.forward(x_querry, str(l), pid)
                    B, _, _ = w.shape
                    tmp = self.zero_pad(w, nw=x.shape[0] // B)
                    prompt_result = torch.einsum('b l c, b d c -> b l d', x, tmp)
                    result = result + prompt_result
                    break
        return result


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0.,
                 merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros((out_channels // self.conv.groups * kernel_size, r * kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)


class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            r: int = 0,
            lora_alpha: int = 2,
            lora_dropout: float = 0.,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # print("in init")
        # embed()
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels * kernel_size, r * kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):  # True for train and False for eval

        nn.Conv2d.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            # print("test")
            # embed()
            if self.merge_weights and not self.merged:
                # print("merging")
                # embed()
                # Merge the weights and mark it
                self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):

        if self.r > 0 and not self.merged:
            return F.conv2d(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )

        return nn.Conv2d.forward(self, x)
