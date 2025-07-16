import torch
import torchvision

def show_kv(net):
    for k, v in net.items():
        print(k)

# x2 -> x3
'''
in_filter = pretrained_net['model.2.weight'] # 256, 64, 3, 3
new_filter = torch.Tensor(576, 64, 3, 3)
new_filter[0:256, :, :, :] = in_filter
new_filter[256:512, :, :, :] = in_filter
new_filter[512:, :, :, :] = in_filter[0:576-512, :, :, :]
crt_net['model.2.weight'] = new_filter

in_bias = pretrained_net['model.2.bias']  # 256, 64, 3, 3
new_bias = torch.Tensor(576)
new_bias[0:256] = in_bias
new_bias[256:512] = in_bias
new_bias[512:] = in_bias[0:576 - 512]
crt_net['model.2.bias'] = new_bias

torch.save(crt_net, '../pretrained_tmp.pth')
'''

# x2 -> x8
'''
crt_net['model.5.weight'] = pretrained_net['model.2.weight']
crt_net['model.5.bias'] = pretrained_net['model.2.bias']
crt_net['model.8.weight'] = pretrained_net['model.2.weight']
crt_net['model.8.bias'] = pretrained_net['model.2.bias']
crt_net['model.11.weight'] = pretrained_net['model.5.weight']
crt_net['model.11.bias'] = pretrained_net['model.5.bias']
crt_net['model.13.weight'] = pretrained_net['model.7.weight']
crt_net['model.13.bias'] = pretrained_net['model.7.bias']
torch.save(crt_net, '../pretrained_tmp.pth')
'''

# x3/4/8 RGB -> Y

def rgb2gray_net(net, only_input=True):

    if only_input:
        in_filter = net['0.weight']
        in_new_filter = in_filter[:,0,:,:]*0.2989 + in_filter[:,1,:,:]*0.587 + in_filter[:,2,:,:]*0.114
        in_new_filter.unsqueeze_(1)
        net['0.weight'] = in_new_filter
    return net

def calculate_parameters(model):
    tot_params = 0
    for name, param in model.named_parameters():
        tot_params += param.numel()
    return tot_params
