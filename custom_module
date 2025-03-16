import torch
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from torch.autograd import Function
from torch import nn
from torch.autograd import Function, Variable


"""Custom modules for quantization"""

# k = 8
class ActFn(Function):
	@staticmethod
	def forward(ctx, x, alpha, k):
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

def quantize_k(r_i, k):
	scale = (2**k - 1)
	r_o = torch.round( scale * r_i ) / scale
	return r_o

class DoReFaQuant(Function): # a custom autograd function by inheriting from torch.autograd.Function.
	@staticmethod
	def forward(ctx, r_i, k):
		"""
		Quantize the input tensor r_i to k bits.	

		Args:
			ctx (_type_): Context object that can be used to save information for the backward pass.
			r_i (_type_): Input tensor (real-valued activations).
			k (_type_): Number of bits for quantization.

		Returns:
			r_o (_type_): quantized output
		"""
		# apply tanh to the input tensor
		tanh = torch.tanh(r_i).float()
		# Normalizes tanh to [0, 1]
		normalized_tanh= tanh / (2*torch.max(torch.abs(tanh)).detach()) + 0.5
		# Quantizes the normalized tanh to k bits
		quantized_tanh = quantize_k(normalized_tanh, k)
		# Maps the result back to [-1, 1]
		r_o = 2 * quantized_tanh - 1
		return r_o

	@staticmethod
	def backward(ctx, dLdr_o):
		"""
		computes gradients for the custom function.
		Args:
			ctx (_type_): Context object that can be used to save information for the backward pass.
			dLdr_o (_type_): The gradient of the loss with respect to the output(r_o) of the forward pass.

		Returns:
			dLdr_o, None: _description_
		"""
		# due to STE, dr_o / d_r_i = 1 according to formula (5)
		# The function is non-differentiable in forward(), but we approximate its gradient as 1.
		return dLdr_o, None # Gradient of loss r_i, Gradient for k is None


class Conv2d(nn.Conv2d):
	def __init__(self, in_places, out_planes, kernel_size, stride=1, padding = 0, groups=1, dilation=1, bias = False, bitwidth = 8):
		super(Conv2d, self).__init__(in_places, out_planes, kernel_size, stride, padding, groups, dilation, bias)
		self.quantize = DoReFaQuant.apply
		self.bitwidth = bitwidth

	def forward(self, x):
		vhat = self.quantize(self.weight, self.bitwidth)
		y = F.conv2d(x, vhat, self.bias, self.stride, self.padding, self.dilation, self.groups)
		return y

class Linear(nn.Linear):
	def __init__(self, in_features, out_features, bias = True, bitwidth = 8):
		super(Linear, self).__init__(in_features, out_features, bias)
		self.quantize = DoReFaQuant.apply
		self.bitwidth = bitwidth
	def forward(self, x):
		vhat = self.quantize(self.weight, self.bitwidth)
		y = F.linear(x, vhat, self.bias)
		return y