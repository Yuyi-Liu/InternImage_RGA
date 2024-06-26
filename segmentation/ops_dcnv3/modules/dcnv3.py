# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from ..functions import DCNv3Function, dcnv3_core_pytorch


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class DCNv3_pytorch(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(
            channels,
            group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)

        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x


class DCNv3(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(
            channels,
            group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    # def forward(self, image_and_depth):
    def forward(self,input,depth=None):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        # input, depth = image_and_depth
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x
        dtype = x.dtype

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1).type(dtype)

        x = DCNv3Function.apply(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256)
        
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1, bias=None, modulation=True):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        if self.padding:
            self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        # if modulation:
        #     self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        #     nn.init.constant_(self.m_conv.weight, 0)
        #     self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x, depth):
        # torch.Size([1, 320, 224, 224]) torch.Size([1, 1, 224, 224])
        
        # 计算偏移量offset。p_conv是一个卷积层，用于从输入特征图x中学习偏移量。
        offset = self.p_conv(x)  # torch.Size([1, 18, 224, 224])


        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            depth_pad = self.zero_padding(depth)

        # (b, 2N, h, w)
        # 计算实际采样点的位置p
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)

        # 计算实际采样点位置p的左上（left-top）邻居q_lt
        q_lt = p.detach().floor()

        # 计算实际采样点位置p的右下（right-bottom）邻居q_rb
        q_rb = q_lt + 1

        # 将q_lt的x坐标和y坐标限制在输入特征图的范围内
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()

        # 计算实际采样点位置p的左下（left-bottom）邻居q_lb
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        
        # 计算实际采样点位置p的右上（right-top）邻居q_rt
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # 将实际采样点位置p的x坐标和y坐标限制在输入特征图的范围内
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        # 计算双线性插值所需的权重
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        

        # (b, c, h, w, N)
        # 提取输入特征图x在相邻格点位置的值
        depth_q_lt = self._get_x_q(depth_pad, q_lt, N)
        depth_q_rb = self._get_x_q(depth_pad, q_rb, N)
        depth_q_lb = self._get_x_q(depth_pad, q_lb, N)
        depth_q_rt = self._get_x_q(depth_pad, q_rt, N)

        # (b, c, h, w, N)
        # 对输入特征图x进行双线性插值
        depth_offset = g_lt.unsqueeze(dim=1) * depth_q_lt + \
                   g_rb.unsqueeze(dim=1) * depth_q_rb + \
                   g_lb.unsqueeze(dim=1) * depth_q_lb + \
                   g_rt.unsqueeze(dim=1) * depth_q_rt
        

        # 计算深度插值，得到深度相似度权重      
        # print(depth.shape, depth_offset.shape)  # torch.Size([1, 1, 224, 224]) torch.Size([1, 1, 224, 224, 9])
        # print(depth.unsqueeze(dim=4).repeat(1, 1, 1, 1, N).shape)
        depth_diff = torch.abs(depth.unsqueeze(dim=4).repeat(1, 1, 1, 1, N) - depth_offset)
        # print(depth_diff.shape) # torch.Size([1, 1, 224, 224, 9])

        # 得到深度相似度权重
        depth_weight = torch.exp(-4.0 * depth_diff) + 0.25 # torch.Size([1, 1, 224, 224, 9])

        # 将深度相似度权重复制为x、y两个方向
        depth_weight_repeat = torch.cat([depth_weight, depth_weight], dim=-1)
        # print(depth_weight_repeat.shape)  # torch.Size([1, 1, 224, 224, 18])
        depth_weight_repeat = depth_weight_repeat.permute(0, 4, 2, 3, 1).squeeze(dim=4)
        # print(depth_weight_repeat.shape) # torch.Size([1, 18, 224, 224])

        # --------------修正偏移量--------------
        offset = offset * depth_weight_repeat
        if self.padding:
            x = self.zero_padding(x)
    
        # (b, 2N, h, w)
        # 计算实际采样点的位置p
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)

        # 计算实际采样点位置p的左上（left-top）邻居q_lt
        q_lt = p.detach().floor()

        # 计算实际采样点位置p的右下（right-bottom）邻居q_rb
        q_rb = q_lt + 1

        # 将q_lt的x坐标和y坐标限制在输入特征图的范围内
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        
        # 计算实际采样点位置p的左下（left-bottom）邻居q_lb
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        
        # 计算实际采样点位置p的右上（right-top）邻居q_rt
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # 将实际采样点位置p的x坐标和y坐标限制在输入特征图的范围内
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        # 计算双线性插值所需的权重
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        # 提取输入特征图x在相邻格点位置的值
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        # 对输入特征图x进行双线性插值
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # 如果启用了调制（modulation），则根据深度相似性，计算调制因子m
        if self.modulation:
            # m = torch.sigmoid(self.m_conv(x))
            # m = m.contiguous().permute(0, 2, 3, 1)
            # m = m.unsqueeze(dim=1)
            # m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            # print(m.shape) # torch.Size([1, 320, 224, 224, 9])
            
            m = torch.exp(-1.0 * depth_diff)
            m = m.repeat(1, x_offset.shape[1], 1, 1, 1)
            x_offset *= m

        # 将插值后的特征图x_offset重塑为适合卷积的形状
        x_offset = self._reshape_x_offset(x_offset, ks)

        # 对重塑后的特征图x_offset应用卷积
        out = self.conv(x_offset)

        return out

    def forward_base(self, x, depth):
        # torch.Size([1, 320, 224, 224]) torch.Size([1, 1, 224, 224])
        
        # 计算偏移量offset。p_conv是一个卷积层，用于从输入特征图x中学习偏移量。
        offset = self.p_conv(x) # offset torch.Size([1, 18, 224, 224])
        print('offset', offset.shape)
        
        # 如果启用了调制（modulation），则计算调制因子m
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        # 计算实际采样点的位置p
        p = self._get_p(offset, dtype) # p torch.Size([1, 18, 224, 224])
        print('p', p.shape)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        print('p', p.shape) # p torch.Size([1, 224, 224, 18])

        # 计算实际采样点位置p的左上（left-top）邻居q_lt
        q_lt = p.detach().floor()
        print('q_lt', q_lt.shape) # q_lt torch.Size([1, 224, 224, 18])

        # 计算实际采样点位置p的右下（right-bottom）邻居q_rb
        q_rb = q_lt + 1

        # 将q_lt的x坐标和y坐标限制在输入特征图的范围内
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        print('q_lt', q_lt.shape) # q_lt torch.Size([1, 224, 224, 18])

        # 计算实际采样点位置p的左下（left-bottom）邻居q_lb
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        print('q_lb', q_lb.shape) # q_lb torch.Size([1, 224, 224, 18])
        
        # 计算实际采样点位置p的右上（right-top）邻居q_rt
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        print('q_rt', q_rt.shape) # q_rt torch.Size([1, 224, 224, 18])

        # 将实际采样点位置p的x坐标和y坐标限制在输入特征图的范围内
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
        print('p', p.shape) # p torch.Size([1, 224, 224, 18])

        # bilinear kernel (b, h, w, N)
        # 计算双线性插值所需的权重
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        print('g_lt', g_lt.shape) # g_lt torch.Size([1, 224, 224, 9]


        # (b, c, h, w, N)
        # 提取输入特征图x在相邻格点位置的值
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        print('x_q_lt', x_q_lt.shape) # x_q_lt torch.Size([1, 320, 224, 224, 9])

        # (b, c, h, w, N)
        # 对输入特征图x进行双线性插值
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        print('x_offset', x_offset.shape) # x_offset torch.Size([1, 320, 224, 224, 9])

        # 如果启用了调制（modulation），则应用调制因子m。调制可以增强可变形卷积的表达能力。
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            print(m.shape)
            assert 0
            x_offset *= m

        # 将插值后的特征图x_offset重塑为适合卷积的形状
        x_offset = self._reshape_x_offset(x_offset, ks)
        print('x_offset', x_offset.shape) # x_offset torch.Size([1, 320, 672, 672])
        
        # 对重塑后的特征图x_offset应用卷积
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class DCNv3_RGA(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))

        # self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        # if center_feature_scale:
        #     self.center_feature_scale_proj_weight = nn.Parameter(
        #         torch.zeros((group, channels), dtype=torch.float))
        #     self.center_feature_scale_proj_bias = nn.Parameter(
        #         torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
        #     self.center_feature_scale_module = CenterFeatureScaleModule()

        self.deformable_conv = DeformConv2d(channels, channels, self.kernel_size, self.stride, self.pad)


    def _reset_parameters(self):
        # xavier_uniform_(self.input_proj.weight.data)
        # constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self,input,depth):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """

        # input, depth = x  # torch.Size([1, 224, 224, 320]) torch.Size([1, 1, 224, 224])
        
        # x = self.input_proj(input)
        
        x1 = input.permute(0, 3, 1, 2) # (N, C, H, W)
        x1 = self.dw_conv(x1)
        x1 = x1.permute(0, 3, 1, 2) # torch.Size([1, 192, 160, 160])

        x = self.deformable_conv(x1, depth)  # torch.Size([1, 320, 224, 224])
        x = x.permute(0, 2, 3, 1)  # torch.Size([1, 224, 224, 320])
        
        x = self.output_proj(x)

        return x
