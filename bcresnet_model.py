# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch.nn.functional as F
from torch import nn
import torch
import torch.nn.init as init

from subspectralnorm import SubSpectralNorm
from module import ActFn, Conv2d, Linear

class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_plane,
        out_plane,
        idx,
        bitwidth,
        kernel_size=3,
        stride=1,
        groups=1,
        use_dilation=False,
        activation=True,
        #swish=False,
        BN=True,
        ssn=False,
    ):
        super().__init__()

        def get_padding(kernel_size, use_dilation):
            rate = 1  # dilation rate
            padding_len = (kernel_size - 1) // 2
            if use_dilation and kernel_size > 1:
                rate = int(2**self.idx)
                padding_len = rate * padding_len
            return padding_len, rate

        self.idx = idx
        self.bitwidth = bitwidth
        # padding and dilation rate
        if isinstance(kernel_size, (list, tuple)):
            padding = []
            rate = []
            for k_size in kernel_size:
                temp_padding, temp_rate = get_padding(k_size, use_dilation)
                rate.append(temp_rate)
                padding.append(temp_padding)
        else:
            padding, rate = get_padding(kernel_size, use_dilation)

        # convbnrelu block
        layers = []
        layers.append(
            Conv2d(in_plane, out_plane, kernel_size, stride, padding, rate, groups, bias=False, bitwidth=self.bitwidth)
        )
        if ssn:
            layers.append(SubSpectralNorm(out_plane, 5))
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
        # if swish:
        #     layers.append(nn.SiLU(True))
        # elif activation:
        #     layers.append(nn.ReLU(True))
        if activation:
            self.act = True
        self.block = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x):
        x = self.block(x)
        if self.act:
            self.ActFn = ActFn.apply
            self.alpha=nn.Parameter(torch.tensor(10.0))
            x = self.ActFn(x, self.alpha, self.bitwidth)
        return x
        #return self.block(x)


class BCResBlock(nn.Module):
    def __init__(self, in_plane, out_plane, idx, stride, bitwidth):
        super().__init__()
        self.transition_block = in_plane != out_plane
        kernel_size = (3, 3)
        self.bitwidth=bitwidth
        # 2D part (f2)
        layers = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_plane, out_plane, idx, self.bitwidth, 1, 1,))
            in_plane = out_plane
        layers.append(
            ConvBNReLU(
                in_plane,
                out_plane,
                idx,
                self.bitwidth,
                (kernel_size[0], 1),
                (stride[0], 1),
                groups=in_plane,
                ssn=True,
                activation=False,
            )
        )
        self.f2 = nn.Sequential(*layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, None))

        # 1D part (f1)
        self.f1 = nn.Sequential(
            ConvBNReLU(
                out_plane,
                out_plane,
                idx,
                self.bitwidth,
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_plane,
                #swish=True,
                use_dilation=True,
            ),
            Conv2d(out_plane, out_plane, 1, bias=False, bitwidth = self.bitwidth),
            nn.Dropout2d(0.1),
        )
        self.apply(_weights_init)

    def forward(self, x):
        # 2D part
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.avg_gpool(x)

        # 1D part
        x = self.f1(x)
        x = x + aux_2d_res
        if not self.transition_block:
            x = x + shortcut
        
        self.ActFn = ActFn.apply
        self.alpha = nn.Parameter(torch.tensor(10.0))
        x = self.ActFn(x, self.alpha, self.bitwidth)
        return x
    
        # x = F.relu(x, True)
        # return x


def BCBlockStage(num_layers, last_channel, cur_channel, idx, use_stride, bitwidth):
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlock(in_plane=channels[i], out_plane=channels[i + 1], idx=idx, stride=stride, bitwidth=bitwidth))
    return stage


class BCResNets(nn.Module):
    def __init__(self, base_c, bitwidth, num_classes=12):
        super().__init__()
        self.num_classes = num_classes
        self.n = [2, 2, 4, 4]  # identical modules repeated n times
        self.c = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]  # num channels
        self.s = [1, 2]  # stage using stride
        self.bitwidth=bitwidth
        self._build_network()

    def _build_network(self,): 
        # Head: (Conv-BN-ReLU)
        
        self.cnn_head = nn.Sequential(
            Conv2d(1, self.c[0], 5, (2, 1), 2, bias=False, bitwidth=self.bitwidth),
            nn.BatchNorm2d(self.c[0]),
            #nn.ReLU(True),
            
        )
        # Body: BC-ResBlocks
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride, bitwidth=self.bitwidth))

        # Classifier
        self.classifier = nn.Sequential(
            Conv2d(
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2), bitwidth=self.bitwidth
            ),
            Conv2d(self.c[-2], self.c[-1], 1, bias=False, bitwidth=self.bitwidth),
            nn.BatchNorm2d(self.c[-1]),)
        
        self.classifier2=nn.Sequential(
            #nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv2d(self.c[-1], self.num_classes, 1, bitwidth=self.bitwidth),
        )
        self.apply(_weights_init)

    def forward(self, x):
        self.ActFn = ActFn.apply
        
        self.cnn_head_alpha = nn.Parameter(torch.tensor(10.0))
        x = self.cnn_head(x)
        x= self.ActFn(x, self.cnn_head_alpha, self.bitwidth)
        
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)
                
        self.classifier_alpha = nn.Parameter(torch.tensor(10.0))        
        x = self.classifier(x)
        x = self.ActFn(x, self.classifier_alpha, self.bitwidth)
        x = self.classifier2(x)
        
        x = x.view(-1, x.shape[1])
        return x
    
def _weights_init(m):
    """
    Initialize weights using Kaiming Normal Initialization
    """
    #classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d): 
        init.kaiming_normal_(m.weight) # Kaiming Normal Initialization

