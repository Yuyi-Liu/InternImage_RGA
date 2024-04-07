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



# kernel:
class DCNv3_failed(nn.Module):

    # def __init__(self,
    #              in_channels,
    #              out_channels,
    #              kernel_size,
    #              stride=1,
    #              padding=1,
    #              dilation=1,
    #              bias=False):
    def __init__(self,
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
                 center_feature_scale=False,
                 bias=False):
        super(DCNv3, self).__init__()
        
        assert isinstance(kernel_size, int)
        assert dilation <= 2
        self.in_channels = channels
        self.out_channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = pad
        self.dilation = dilation

        self.alpha = 4.0

        K = self.kernel_size
        self.weight_offset = nn.Parameter(torch.Tensor(K * K * 2, channels + 1, K, K))
        self.bias_offset = nn.Parameter(torch.Tensor(K * K * 2))

        self.weight = nn.Parameter(torch.Tensor(channels, channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(pad))
        else:
            self.register_parameter('bias', None)
        self.init_weight()

    def init_weight(self):
        nn.init.constant_(self.weight_offset, 0.)
        if self.bias_offset is not None:
            nn.init.constant_(self.bias_offset, 0.)

    def _reset_parameters(self):
        self.init_weight()

    def init_delta(self, B, out_H, out_W, dtype, device):
        K = self.kernel_size
        h_st = -((K - 1) // 2) * self.dilation
        h_ed = (K // 2) * self.dilation + 1
        w_st = -((K - 1) // 2) * self.dilation
        w_ed = (K // 2) * self.dilation + 1

        delta_x, delta_y = torch.meshgrid(
            torch.arange(h_st, h_ed, self.dilation),
            torch.arange(w_st, w_ed, self.dilation)
        )
        delta_x = delta_x.contiguous().view(1, K * K, 1, 1)
        delta_y = delta_y.contiguous().view(1, K * K, 1, 1)
        delta = torch.cat((delta_x, delta_y), dim=1).repeat(B, 1, out_H, out_W).type(dtype)
        return delta.to(device)
    
    def init_base(self, B, out_h, out_w, x_h, x_w, dtype, device):
        K = self.kernel_size
        h_st = ((K - 1) // 2) * self.dilation
        h_ed = x_h - (K // 2) * self.dilation
        w_st = ((K - 1) // 2) * self.dilation
        w_ed = x_w - (K // 2) * self.dilation

        base_x, base_y = torch.meshgrid(
            torch.arange(h_st, h_ed, self.stride),
            torch.arange(w_st, w_ed, self.stride)
        )
        base_x = base_x.contiguous().view(1, 1, out_h, out_w).repeat(B, K * K, 1, 1)
        base_y = base_y.contiguous().view(1, 1, out_h, out_w).repeat(B, K * K, 1, 1)
        base = torch.cat((base_x, base_y), dim=1).type(dtype)
        return base.to(device)
    
    def unfold(self, x: torch.Tensor, base: torch.Tensor, delta: torch.Tensor):
        K = self.kernel_size
        B, C, H, W = x.shape
        out_H = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_W = (W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        H += 2 * self.dilation
        W += 2 * self.dilation
        x = F.pad(x, [self.dilation] * 4, mode='constant', value=0.)

        coordinate = base + delta  # [B, K * K * 2, out_H, out_W]

        coordinate = coordinate.contiguous().view(B, K * K * 2, -1)  # [B, K * K * 2, out_H * out_W]

        c_x = coordinate[:, :K * K, :].contiguous().view(B, -1)   # [B, K * K * out_H * out_W]
        c_y = coordinate[:, K * K:, :].contiguous().view(B, -1)

        c_x = torch.clamp(c_x, 0, H - 1)
        c_y = torch.clamp(c_y, 0, W - 1)

        tl_x = c_x.detach().floor().long()
        tl_y = c_y.detach().floor().long()
        br_x = tl_x + 1
        br_y = tl_y + 1

        tl_x = torch.clamp(tl_x, 0, H - 2)
        tl_y = torch.clamp(tl_y, 0, W - 2)
        br_x = torch.clamp(br_x, 1, H - 1)
        br_y = torch.clamp(br_y, 1, W - 1)

        index_tl = (tl_x * W + tl_y)  # [B, K * K * out_H * out_W]
        index_tr = (tl_x * W + br_y)
        index_bl = (br_x * W + tl_y)
        index_br = (br_x * W + br_y)
        index_tl = index_tl.unsqueeze(dim=1).repeat(1, C, 1)  # [B, C, K * K * out_H * out_W]
        index_tr = index_tr.unsqueeze(dim=1).repeat(1, C, 1)
        index_bl = index_bl.unsqueeze(dim=1).repeat(1, C, 1)
        index_br = index_br.unsqueeze(dim=1).repeat(1, C, 1)

        x = x.contiguous().view(B, C, -1)  #  [B, C, H * W]

        x_tl = x.gather(dim=-1, index=index_tl)   # [B, C, K * K * out_H * out_W]
        x_tr = x.gather(dim=-1, index=index_tr)
        x_bl = x.gather(dim=-1, index=index_bl)
        x_br = x.gather(dim=-1, index=index_br)

        #  [B, K * K * out_H * out_W]
        c_tl = (1 - (c_x - tl_x.type_as(c_x))) * \
               (1 - (c_y - tl_y.type_as(c_y)))
        c_tr = (1 - (c_x - tl_x.type_as(c_x))) * \
               (1 + (c_y - br_y.type_as(c_y)))
        c_bl = (1 + (c_x - br_x.type_as(c_x))) * \
               (1 - (c_y - tl_y.type_as(c_y)))
        c_br = (1 + (c_x - br_x.type_as(c_x))) * \
               (1 + (c_y - br_y.type_as(c_y)))
    
        #  [B, C, K * K * out_H * out_W]
        out = x_tl * c_tl.unsqueeze(dim=1) + \
                x_tr * c_tr.unsqueeze(dim=1) + \
                x_bl * c_bl.unsqueeze(dim=1) + \
                x_br * c_br.unsqueeze(dim=1)
        out = out.contiguous().view(B, -1, out_H * out_W)  # [B, C * K * K, out_H * out_W]
        return out

    def forward(self, x):

        x, depth = x
        x = x.permute(0, 3, 1, 2) # (N, C, H, W)

        B, C, H, W = x.shape
        out_H = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_W = (W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        K = self.kernel_size
        dtype = x.dtype
        device = x.device

        base = self.init_base(B, out_H, out_W, H + 2 * self.dilation, W + 2 * self.dilation, dtype, device)
        delta = self.init_delta(B, out_H, out_W, dtype, device)

        x_depth = torch.cat([x, depth], dim=1)
        x_depth_col = F.unfold(x_depth, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

        offset = torch.matmul(self.weight_offset.view(-1, (C + 1) * K * K), x_depth_col)
        offset = offset.view(B, -1, out_H, out_W)
        offset += self.bias_offset.view(1, -1, 1, 1)

        delta = delta + offset

        depth_center = depth[:, :, ::self.stride, ::self.stride].contiguous().view(B, 1, out_H * out_W)

        delta_0 = delta * 0.75
        delta_1 = delta * 1.00
        delta_2 = delta * 1.25

        depth_col_0 = self.unfold(depth, base, delta_0)
        depth_col_1 = self.unfold(depth, base, delta_1)
        depth_col_2 = self.unfold(depth, base, delta_2)

        depth_col_0 = torch.abs(depth_col_0 - depth_center)
        depth_col_1 = torch.abs(depth_col_1 - depth_center)
        depth_col_2 = torch.abs(depth_col_2 - depth_center)

        depth_mask = torch.argmin(torch.cat([
            depth_col_0.unsqueeze(dim=-1),
            depth_col_1.unsqueeze(dim=-1),
            depth_col_2.unsqueeze(dim=-1),
        ], dim=-1), dim=-1)

        mask_0 = torch.zeros_like(depth_mask, requires_grad=False, device=device)
        mask_1 = torch.zeros_like(depth_mask, requires_grad=False, device=device)
        mask_2 = torch.zeros_like(depth_mask, requires_grad=False, device=device)

        mask_0[depth_mask == 0] = 1.0
        mask_1[depth_mask == 1] = 1.0
        mask_2[depth_mask == 2] = 1.0

        mask_0 = mask_0.repeat(1, C, 1)
        mask_1 = mask_1.repeat(1, C, 1)
        mask_2 = mask_2.repeat(1, C, 1)

        x_col_0 = self.unfold(x, base, delta_0)
        x_col_1 = self.unfold(x, base, delta_1)
        x_col_2 = self.unfold(x, base, delta_2)

        mouduls_0 = torch.exp(-self.alpha * depth_col_0).repeat(1, C, 1)
        mouduls_1 = torch.exp(-self.alpha * depth_col_1).repeat(1, C, 1)
        mouduls_2 = torch.exp(-self.alpha * depth_col_2).repeat(1, C, 1)

        x_col_0 *= mouduls_0
        x_col_1 *= mouduls_1
        x_col_2 *= mouduls_2

        x_col = mask_0 * x_col_0 + mask_1 * x_col_1 + mask_2 * x_col_2

        out = torch.matmul(self.weight.view(-1, C * K * K), x_col)
        out = out.view(B, -1, out_H, out_W)
        if self.bias:
            out += self.bias.view(1, -1, 1, 1)
        print(out.shape)  # torch.Size([1, 192, 160, 160])
        
        out = out.permute(0, 2, 3, 1)
        return (out, depth)


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
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

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

# class DeformableConv2D(nn.Module):
#     def __init__(self, input_channels, output_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, group_channels, offset_scale, im2col_step):
#         super(DeformableConv2D, self).__init__()
#         self.kernel_h = kernel_h
#         self.kernel_w = kernel_w
#         self.stride_h = stride_h
#         self.stride_w = stride_w
#         self.pad_h = pad_h
#         self.pad_w = pad_w
#         self.dilation_h = dilation_h
#         self.dilation_w = dilation_w
#         self.group = group
#         self.group_channels = group_channels
#         self.offset_scale = offset_scale
#         self.im2col_step = im2col_step

#         self.conv_offset = nn.Conv2d(input_channels, 2 * kernel_h * kernel_w * group_channels, kernel_size=1, stride=1, padding=0, bias=True)
#         self.conv_mask = nn.Conv2d(input_channels, kernel_h * kernel_w * group_channels, kernel_size=1, stride=1, padding=0, bias=True)
#         self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w), groups=group)

#     def forward(self, x):
#         offset = self.conv_offset(x)
#         mask = torch.sigmoid(self.conv_mask(x))
#         # print(offset.shape, mask.shape) # torch.Size([1, 576, 224, 224]) torch.Size([1, 288, 224, 224])
#         # print(x.shape) # torch.Size([1, 320, 224, 224])
#         x = self.deform_conv2d(x, offset, mask, self.conv.weight, None, self.stride_h, self.stride_w, self.pad_h, self.pad_w, self.dilation_h, self.dilation_w, self.group, self.group_channels, self.offset_scale, self.im2col_step)
#         return x
    
#     def deform_conv2d(self, x, offset, mask, weight, bias, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, group_channels, offset_scale, im2col_step):
#         batch_size, input_channels, input_height, input_width = x.size()
#         output_channels, _, kernel_h, kernel_w = weight.size()

#         # Reshape the offset tensor and apply the offset_scale
#         offset = offset.view(batch_size, 2 * group_channels, -1, input_height, input_width) * offset_scale
#         offset = offset.permute(0, 2, 3, 4, 1)

#         # Create the grid for grid_sample
#         grid_y, grid_x = torch.meshgrid(torch.arange(input_height), torch.arange(input_width))
#         grid = torch.stack((grid_x, grid_y), 2).float().unsqueeze(0)
#         grid = grid.repeat(batch_size, 1, 1, 1)

#         # Move the grid tensor to the same device as the offset tensor
#         grid = grid.to(offset.device)

#         # Apply the offsets to the grid
#         grid_offset = grid + offset
#         grid_offset = (grid_offset / torch.tensor([input_width - 1, input_height - 1], device=grid_offset.device)) * 2 - 1

#         # Reshape the mask tensor
#         mask = mask.view(batch_size, group_channels, -1, input_height, input_width)
#         mask = mask.permute(0, 2, 3, 4, 1)

#         # Apply the offsets and masks to the input feature map
#         x_offset = F.grid_sample(x, grid_offset, align_corners=False)
#         x_mask = F.grid_sample(x, grid_offset, align_corners=False) * mask

#         # Perform the convolution for each group
#         x_deform = []
#         for g in range(group):
#             x_deform_g = F.conv2d(x_offset[:, g * group_channels:(g + 1) * group_channels] * x_mask[:, g * group_channels:(g + 1) * group_channels],
#                                 weight[g * output_channels // group:(g + 1) * output_channels // group],
#                                 bias[g * output_channels // group:(g + 1) * output_channels // group],
#                                 stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w)
#             x_deform.append(x_deform_g)
#         x_deform = torch.cat(x_deform, dim=1)

#         return x_deform


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

        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

        self.deformable_conv = DeformConv2d(channels, channels, self.kernel_size, self.stride, self.pad)


    def _reset_parameters(self):
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, x):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """

        input, depth = x  # torch.Size([1, 224, 224, 320]) torch.Size([1, 1, 224, 224])
        N, H, W, _ = input.shape
        x = self.input_proj(input)
        x_proj = x
        dtype = x.dtype
        x1 = input.permute(0, 3, 1, 2) # (N, C, H, W)
        x1 = self.dw_conv(x1)
        x1 = x1.permute(0, 3, 1, 2) # torch.Size([1, 192, 160, 160])

        x = self.deformable_conv(x1, depth)  # torch.Size([1, 320, 224, 224])
        x = x.permute(0, 2, 3, 1)  # torch.Size([1, 224, 224, 320])
        
        x = self.output_proj(x)

        return (x, depth)





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

    def forward(self, x):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """

        input, depth = x
        N, H, W, _ = input.shape
        # print('0', depth.shape)  # torch.Size([1, 1, 160, 160])
        # print('1', input.shape)  # torch.Size([1, 160, 160, 192])
        x = self.input_proj(input)
        # print('2', x.shape)  # torch.Size([1, 160, 160, 192])
        x_proj = x
        dtype = x.dtype

        x1 = input.permute(0, 3, 1, 2) # (N, C, H, W)
        # print('3', x1.shape) # torch.Size([1, 192, 160, 160])
        x1 = self.dw_conv(x1)
        # print('4', x1.shape) # torch.Size([1, 160, 160, 192])
        offset = self.offset(x1)
        # print('offset', offset.shape)  # torch.Size([1, 160, 160, 216])
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1).type(dtype)
        # print('mask', mask.shape)  # torch.Size([1, 160, 160, 108])

        '''
        0 torch.Size([1, 1, 160, 160])
        1 torch.Size([1, 160, 160, 192])
        2 torch.Size([1, 160, 160, 192])
        3 torch.Size([1, 192, 160, 160])
        4 torch.Size([1, 160, 160, 192])
        offset torch.Size([1, 160, 160, 216])
        mask torch.Size([1, 160, 160, 108])
        '''

        x = DCNv3Function.apply(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256)
        # print(x.shape)  # torch.Size([1, 160, 160, 192])
        
        if self.center_feature_scale: # defeat: False
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
                
        x = self.output_proj(x)
        # print(x.shape)  # torch.Size([1, 160, 160, 192])

        return (x, depth)
