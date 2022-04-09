import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.layers.Inv_Subnet_constructor import subnet

class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, img_size, num_heads, mlp_ratio, sr_ratio, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1,img_size,num_heads,mlp_ratio,sr_ratio)
        self.G = subnet_constructor(self.split_len1, self.split_len2,img_size,num_heads,mlp_ratio,sr_ratio)
        self.H = subnet_constructor(self.split_len1, self.split_len2,img_size,num_heads,mlp_ratio,sr_ratio)

    def forward(self,x,rev=False):

        x1,x2 = (x.narrow(1,0,self.split_len1),x.narrow(1,self.split_len1,self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1,y2),1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, reverse=False):
        if not reverse:
            output = self.squeeze2d(input,self.factor)  # Squeeze in forward
            return output
        else:
            output = self.unsqueeze2d(input,self.factor)
            return output

    def jacobian(self, x, rev=False):
        return 0

    @staticmethod
    def squeeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B, factor * factor * C, H // factor, W // factor)
        return x

    @staticmethod
    def unsqueeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert C % (factor2) == 0, "{}".format(C)
        x = input.view(B, factor, factor, C // factor2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x

class InvNet(nn.Module):

    def __init__(self, channel_in=2, block_num=[1,1,1,1], down_num=4, img_sizes=[128,64,32,16], num_heads=[2,2,4,8], mlp_ratios = [8,8,4,4], sr_ratios = [8,4,2,1]):
        super(InvNet, self).__init__()

        operations = []
        current_channel = channel_in

        for i in range(down_num):
            b = SqueezeLayer(2)
            operations.append(b)
            current_channel *= 4
            if i == 0:
                subnet_constructor = subnet("Transformer")
            elif i != 0:
                subnet_constructor = subnet("Transformer")
            for j in range(block_num[i]):
                channel_split_num = current_channel // 4
                b = InvBlockExp(subnet_constructor,current_channel, channel_split_num, img_sizes[i], num_heads[i], mlp_ratios[i], sr_ratios[i])
                operations.append(b)
        self.operations = nn.ModuleList(operations)

    def forward(self,x,rev=False):

        out = x
        if not rev:
            for op in self.operations:
                out = op.forward(out,rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
        return out