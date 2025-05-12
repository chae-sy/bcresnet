# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
from torch.autograd import Function
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn

from subspectralnorm import SubSpectralNorm
class ActFn(Function):
    @staticmethod
    def forward(ctx, x, alpha, k=8):
        """
        summary

        Args:
            ctx (_type_): Context object that can be used to save information for the backward pass.
            x (_type_): input
            alpha (_type_): alpha for restricting the max value of ReLU
            k (_type_): number of bits for quantization
        Returns:
            y_q (_type_): quantized output of PACT 
        """
        ctx.save_for_backward(x, alpha)
        y = torch.relu(x)
        y = torch.minimum(x, alpha)
        scale = (2**k - 1) / alpha
        y_q = torch.round(y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        """
        # Backward function, borrowed code from...
        """
        x, alpha = ctx.saved_tensors
        lower_bound = x < 0
        upper_bound = x > alpha
        mask = ~(lower_bound | upper_bound)
        grad_alpha = torch.sum(dLdy_q * (x >= alpha).float())
        grad_x = dLdy_q * mask.float()
        return grad_x, grad_alpha, None

def _weights_init(m):
    """
    Initialize weights using Kaiming Normal Initialization
    """
    #classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d): 
        init.kaiming_normal_(m.weight) # Kaiming Normal Initialization

class PACTActivation(nn.Module):
    def __init__(self, k: int = 8, alpha_init: float = 6.0, eps: float = 1e-3):
        super().__init__()
        self.k   = k
        self.eps = eps
        # scalar alpha
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        # 1) autograd 경로를 유지하며 alpha를 eps 이상으로 clamp
        alpha = torch.clamp(self.alpha, min=self.eps)

        # 2) tensor 버전 clamp → alpha.item() 제거
        y = torch.relu(x)
        y=torch.minimum(y, alpha)

        # 3) numerically safe한 scale 계산
        qmax  = float(2**self.k - 1)
        scale = qmax / alpha

        # 4) quantization
        y_q = torch.round(y * scale) / scale

        return y_q

class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_plane,
        out_plane,
        idx,
        kernel_size=3,
        stride=1,
        groups=1,
        use_dilation=False,
        activation=True,
        swish=False,
        BN=True,
        ssn=False,
    ):
        super().__init__()
        

        ## ----- PACT ----- ##
        self.alpha1 = nn.Parameter(torch.tensor(10.))
        self.ActFn = ActFn.apply
        ## ---------------- ##

        def get_padding(kernel_size, use_dilation):
            rate = 1  # dilation rate
            padding_len = (kernel_size - 1) // 2
            if use_dilation and kernel_size > 1:
                rate = int(2**self.idx)
                padding_len = rate * padding_len
            return padding_len, rate

        self.idx = idx

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
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding, rate, groups, bias=False)
        )
        if ssn:
            layers.append(SubSpectralNorm(out_plane, 5))
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
        if swish:
            layers.append(PACTActivation(k=8, alpha_init=10.0))
            #layers.append(nn.SiLU(True))
        elif activation:
            layers.append(PACTActivation(k=8, alpha_init=10.0))
            #layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)
        
        self.apply(_weights_init)
        
    def forward(self, x):
        return self.block(x)


class BCResBlock(nn.Module):
    def __init__(self, in_plane, out_plane, idx, stride):
        super().__init__()
        self.transition_block = in_plane != out_plane
        kernel_size = (3, 3)

        # 2D part (f2)
        layers = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_plane, out_plane, idx, 1, 1))
            in_plane = out_plane
        layers.append(
            ConvBNReLU(
                in_plane,
                out_plane,
                idx,
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
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_plane,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )
        
        ## ----- PACT ----- ##
        self.alpha1 = nn.Parameter(torch.tensor(10.))
        self.ActFn = ActFn.apply
        self.apply(_weights_init)
        ## ---------------- ##

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
        #x = F.relu(x, True)

        ## ----- PACT ----- ##
        x = self.ActFn(x, self.alpha1)
        return x


def BCBlockStage(num_layers, last_channel, cur_channel, idx, use_stride):
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlock(channels[i], channels[i + 1], idx, stride))
    return stage


class BCResNets(nn.Module):
    def __init__(self, base_c, num_classes=12):
        super().__init__()
        self.num_classes = num_classes
        self.n = [1]  # identical modules repeated n times
        self.c = [
        
            base_c,
            base_c*2
        ]  # num channels
        self.s = [1, 2]  # stage using stride
        self._build_network()

    def _build_network(self):
        # Head: (Conv-BN-ReLU)
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (2, 1), 2, bias=True),
            nn.BatchNorm2d(self.c[0]),
            #nn.ReLU(True),
            PACTActivation(k=8, alpha_init=10.0)
        )
        # Body: BC-ResBlocks
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(
                self.c[-1], self.c[-1], (5, 5), bias=True, groups=self.c[-1], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-1], self.c[-1], 1, bias=True),
            nn.BatchNorm2d(self.c[-1]),
            PACTActivation(k=8, alpha_init=10.0),
            #nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], self.num_classes, 1),
        )
        self.apply(_weights_init)

    def forward(self, x):
        x = self.cnn_head(x)
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)
        x = self.classifier(x)
        x = x.view(-1, x.shape[1])
        return x
