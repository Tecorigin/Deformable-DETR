"""Isolated FP32 SDAA deformable attention with explicit first-order gradients."""
import ctypes
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable


class Arguments(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in
                ('value', 'locations', 'weights', 'grad_output', 'output',
                 'grad_value', 'grad_locations', 'grad_weights')]
    _fields_ += [(name, ctypes.c_int) for name in
                 ('batch', 'spatial', 'heads', 'channels', 'queries', 'levels', 'points')]
    _fields_ += [(name, ctypes.c_int * 8) for name in ('height', 'width', 'start')]


class NativeAttention:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()), mode=ctypes.RTLD_LOCAL)
        self.library.detr_argument_size.restype = ctypes.c_int
        if self.library.detr_argument_size() != ctypes.sizeof(Arguments):
            raise RuntimeError('Native argument layout differs')
        for name in ('detr_attention_forward', 'detr_attention_backward'):
            function = getattr(self.library, name)
            function.argtypes = [ctypes.POINTER(Arguments), ctypes.c_void_p]
            function.restype = ctypes.c_int
        self.forward_calls = 0
        self.backward_calls = 0

    @staticmethod
    def arguments(value, shapes, starts, locations, weights):
        if value.ndim != 4 or locations.ndim != 6 or weights.ndim != 5:
            raise ValueError('Invalid attention tensor ranks')
        batch, spatial, heads, channels = value.shape
        b, queries, h, levels, points, xy = locations.shape
        if (b, h, xy) != (batch, heads, 2) or tuple(weights.shape) != (batch, queries, heads, levels, points):
            raise ValueError('Attention dimensions differ')
        if min(batch, spatial, heads, channels, queries, levels, points) < 1 or levels > 8:
            raise ValueError('Unsupported empty shape or level count')
        if max(batch, spatial, heads, channels, queries, levels, points) > 2**31 - 1:
            raise ValueError('Native integer shape exceeds range')
        if any(t.dtype != torch.float32 or t.device.type != 'sdaa' or t.device != value.device
               or not t.is_contiguous() for t in (value, locations, weights)):
            raise ValueError('Native attention requires contiguous FP32 tensors on one SDAA device')
        sizes = shapes.detach().cpu().tolist()
        offsets = starts.detach().cpu().tolist()
        if len(sizes) != levels or len(offsets) != levels:
            raise ValueError('Invalid level metadata')
        total = 0
        for size, offset in zip(sizes, offsets):
            if len(size) != 2 or min(size) < 1 or offset != total:
                raise ValueError('Invalid spatial shape or start')
            total += size[0] * size[1]
        if total != spatial:
            raise ValueError('Spatial shape total differs from value')
        args = Arguments()
        for name, tensor in (('value', value), ('locations', locations), ('weights', weights)):
            setattr(args, name, tensor.data_ptr())
        for name, number in zip(('batch', 'spatial', 'heads', 'channels', 'queries', 'levels', 'points'),
                                (batch, spatial, heads, channels, queries, levels, points)):
            setattr(args, name, number)
        for i, ((height, width), offset) in enumerate(zip(sizes, offsets)):
            args.height[i], args.width[i], args.start[i] = height, width, offset
        return args

    def apply(self, value, shapes, starts, locations, weights, step=64):
        return _Attention.apply(value, shapes, starts, locations, weights, self)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, shapes, starts, locations, weights, binding):
        value, locations, weights = [t.contiguous() for t in (value, locations, weights)]
        args = binding.arguments(value, shapes, starts, locations, weights)
        output = torch.empty((args.batch, args.queries, args.heads * args.channels),
                             dtype=value.dtype, device=value.device)
        args.output = output.data_ptr()
        with torch.sdaa.device(value.device):
            stream = torch.sdaa.current_stream().sdaa_stream
            code = binding.library.detr_attention_forward(ctypes.byref(args), stream)
        if code:
            raise RuntimeError('Native attention forward launch failed: ' + str(code))
        binding.forward_calls += 1
        ctx.save_for_backward(value, locations, weights)
        ctx.arguments, ctx.binding = args, binding
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, locations, weights = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_value, grad_locations, grad_weights = [torch.empty_like(t) for t in (value, locations, weights)]
        args, binding = ctx.arguments, ctx.binding
        for name, tensor in (('grad_output', grad_output), ('grad_value', grad_value),
                             ('grad_locations', grad_locations), ('grad_weights', grad_weights)):
            setattr(args, name, tensor.data_ptr())
        with torch.sdaa.device(value.device):
            stream = torch.sdaa.current_stream().sdaa_stream
            code = binding.library.detr_attention_backward(ctypes.byref(args), stream)
        if code:
            raise RuntimeError('Native attention backward launch failed: ' + str(code))
        binding.backward_calls += 1
        return grad_value, None, None, grad_locations, grad_weights, None
