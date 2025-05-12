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
		"""_summary_

		Args:
			ctx (_type_): Context object that can be used to save information for the backward pass.
			x (_type_): input
			alpha (_type_): alpha for restricting the max value of ReLU
			k (_type_): number of bits for quantization

		Returns:
			y_q (_type_): quantized output of PACT 
		"""
		ctx.save_for_backward(x, alpha)
		y = torch.clamp(x, min = 0, max = alpha.item())
		scale = (2**k - 1) / alpha
		y_q = torch.round( y * scale) / scale
		return y_q

	@staticmethod
	def backward(ctx, dLdy_q):
		# Backward function, I borrowed code from
		# https://github.com/obilaniu/GradOverride/blob/master/functional.py
		# We get dL / dy_q as a gradient
		x, alpha, = ctx.saved_tensors
		# Weight gradient is only valid when [0, alpha]
		# Actual gradient for alpha,
		# By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
		# dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range 
		lower_bound      = x < 0
		upper_bound      = x > alpha
		x_range = ~(lower_bound|upper_bound)
		grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
		return dLdy_q * x_range.float(), grad_alpha, None

def _weights_init(m):
    """
    Initialize weights using Kaiming Normal Initialization
    """
    #classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d): 
        init.kaiming_normal_(m.weight) # Kaiming Normal Initialization

class PACTActivation(nn.Module):
    """
    A module wrapper so we can plug the PACT quant‑ReLU into nn.Sequential.
    alpha becomes a learnable parameter; k (bit‑width) is fixed at construction.
    """
    def __init__(self, k: int = 8, alpha_init: float = 6.0):
        super().__init__()
        self.k = k
        # alpha is usually channel‑wise or tensor‑wise; here it's scalar
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        # ActFn.apply(*inputs) is the autograd‑aware call
        return ActFn.apply(x, self.alpha, self.k)

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
        self._build_network()

    def _build_network(self):
        # Head: (Conv-BN-ReLU)
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (2, 1), 2, bias=False),
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
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            PACTActivation(k=8, alpha_init=10.0),
            #nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], self.num_classes, 1),
        )
        self.apply(_weights_init)

    def forward(self, x):
        x = F.tanh(x)
        x = self.cnn_head(x)
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)
        x = self.classifier(x)
        x = x.view(-1, x.shape[1])
        return x
