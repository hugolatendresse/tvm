# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
from tvm import relax
import tvm.testing
import numpy as np
import torch
from torch import nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program
from torch.nn import Softmax, Upsample


def assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev):
    """
    This util ensures that a torch module can successfully be exported to TVM
    using torch.export and that the resuling IR program gives the same result
    as PyTorch when ran on CUDA.
    """
    raw_data_for_tvm = raw_data.copy()  # In case the data is modified
    torch_data = torch.from_numpy(raw_data)
    example_args = (torch_data,)

    with torch.no_grad():
        exported_program = export(torch_module, example_args)
        mod_from_torch = from_exported_program(exported_program, keep_params_as_input=True)

    tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)

    relax_pipeline = relax.get_default_pipeline(tvm.target.Target.from_device(tvm.cuda()))
    ex = relax.build(tvm_mod, target=target, relax_pipeline=relax_pipeline)
    vm = relax.VirtualMachine(ex, dev)

    gpu_data = tvm.nd.array(raw_data_for_tvm, dev)
    gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
    gpu_out = vm["main"](gpu_data, *gpu_params)

    pytorch_out = torch_module(torch_data)

    if isinstance(pytorch_out, tuple):
        for i in range(len(pytorch_out)):
            actual = gpu_out[i].numpy()
            desired = pytorch_out[i].detach().numpy()
            np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5)
    else:
        actual = gpu_out[0].numpy()
        desired = pytorch_out.detach().numpy()
        np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets("cuda")
def test_tensor_clamp(target, dev):
    class ClampBothTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("min_val", torch.tensor(-1.0))
            self.register_buffer("max_val", torch.tensor(1.0))

        def forward(self, x):
            return x.clamp(min=self.min_val, max=self.max_val)

    class ClampBothInt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.min_val = -1
            self.max_val = 1

        def forward(self, x):
            return x.clamp(min=self.min_val, max=self.max_val)

    class ClampMinOnlyTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("min_val", torch.tensor(0.0))

        def forward(self, x):
            return x.clamp(min=self.min_val)

    class ClampMinOnlyInt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.min_val = 0

        def forward(self, x):
            return x.clamp(min=self.min_val)

    class ClampMaxOnlyTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("max_val", torch.tensor(0.5))

        def forward(self, x):
            return x.clamp(max=self.max_val)

    class ClampMaxOnlyInt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.max_val = 0.5

        def forward(self, x):
            return x.clamp(max=self.max_val)

    class ClampDifferentValues(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.min_val = -2
            self.max_val = 2

        def forward(self, x):
            return x.clamp(min=self.min_val, max=self.max_val)

    # Create random data with values outside our clamp ranges
    raw_data = np.random.uniform(-3.0, 3.0, (2, 3, 4, 5)).astype(np.float32)

    torch_module0 = ClampBothTensor().eval()
    torch_module1 = ClampBothInt().eval()
    torch_module2 = ClampMinOnlyTensor().eval()
    torch_module3 = ClampMinOnlyInt().eval()
    torch_module4 = ClampMaxOnlyTensor().eval()
    torch_module5 = ClampMaxOnlyInt().eval()
    torch_module6 = ClampDifferentValues().eval()

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module1, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module2, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module3, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module4, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module5, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module6, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_tensor_expand_as(target, dev):
    class ExpandAs0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.template = torch.ones((1, 1, 1, 1))

        def forward(self, x):
            return self.template.expand_as(x)

    class ExpandAs1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.template = torch.ones((2, 1, 4, 1))

        def forward(self, x):
            return self.template.expand_as(x)

    class ExpandAs2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.template = torch.ones((2, 1, 1, 10))

        def forward(self, x):
            return self.template.expand_as(x)

    class ExpandAs3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.template = torch.ones((2, 3, 1, 1))

        def forward(self, x):
            return self.template.expand_as(x)

    raw_data = np.random.randn(2, 3, 4, 10).astype(np.float32)

    torch_module0 = ExpandAs0().eval()
    torch_module1 = ExpandAs1().eval()
    torch_module2 = ExpandAs2().eval()
    torch_module3 = ExpandAs3().eval()

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module1, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module2, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module3, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_copy_(target, dev):
    class CopyTester(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.register_buffer("buffer", torch.zeros(size))

        def forward(self, x):
            self.buffer.copy_(x)

            return x * 3 + self.buffer * 5

    size = (2, 2)
    raw_data = np.random.rand(*size).astype(np.float32)
    torch_module = CopyTester(size).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_upsample_with_size(target, dev):
    """
    The Upsample module can be used with the size arugment or the scale
    factor argument but not both. This tests the former.
    """
    batch_size = 1
    channels = 3
    height, width = 8, 8

    torch_module = Upsample(size=(64, 64), mode="nearest", recompute_scale_factor=None)

    raw_data = np.random.rand(batch_size, channels, height, width).astype("float32")

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_detach_no_change(target, dev):
    # In TVM, detach() is just identity
    class DetachTester(nn.Module):
        def forward(self, x):
            detached = x.detach()
            return detached

    raw_data = np.ones((2, 2)).astype(np.float32)
    torch_module = DetachTester().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_upsample_with_scale_factor(target, dev):
    """
    The Upsample module can be used with the size arugment or the scale
    factor argument but not both. This tests the latter.
    """
    batch_size = 2
    channels = 3
    height, width = 32, 32

    torch_module = Upsample(
        size=None, scale_factor=7, mode="nearest", align_corners=None, recompute_scale_factor=True
    )

    raw_data = np.random.rand(batch_size, channels, height, width).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_linalg_vector_norm(target, dev):
    class VectorNorm0(torch.nn.Module):
        def forward(self, x):
            return torch.linalg.vector_norm(x, ord=1, dim=-1)

    class VectorNorm1(torch.nn.Module):
        def forward(self, x):
            return torch.linalg.vector_norm(x, ord=2, dim=2)

    class VectorNorm2(torch.nn.Module):
        def forward(self, x):
            return torch.linalg.vector_norm(x, ord=1, dim=-1)

    class VectorNorm3(torch.nn.Module):
        def forward(self, x):
            return torch.linalg.vector_norm(x, ord=2, dim=2)

    raw_data = np.random.randn(2, 3, 4, 10).astype(np.float32)

    torch_module0 = VectorNorm0().eval()
    torch_module1 = VectorNorm1().eval()
    torch_module2 = VectorNorm2().eval()
    torch_module3 = VectorNorm3().eval()

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module1, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module2, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module3, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_batch_norm_prog(target, dev):
    # Default args, in a pytorch program (to ensure output is in proper type and format)
    raw_data = np.random.randn(2, 3, 2, 2).astype(np.float32)

    class BatchNormWrapper(nn.Module):
        def __init__(self):
            super(BatchNormWrapper, self).__init__()
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x):
            x = self.bn(x)
            x = x + 1
            return x

    torch_module = BatchNormWrapper().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_split_size(target, dev):
    # Test split using the split_size argument such that it is not a divisor
    # of the dimension to split (the last tensor will be smaller)
    batch = 2
    channels = 7
    height, width = 2, 2
    split_size = 3  # last tensor will have just 1 element
    dim = 1  # split across channels
    raw_data = np.random.rand(batch, channels, height, width).astype("float32")

    class SplitModelSplitSize(nn.Module):
        def __init__(self, split_size, dim):
            super().__init__()
            self.split_size = split_size
            self.dim = dim

        def forward(self, x):
            return torch.split(x, split_size_or_sections=self.split_size, dim=self.dim)

    torch_module = SplitModelSplitSize(split_size=split_size, dim=dim).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_split_sections_list(target, dev):
    # Test split using a list of section sizes
    batch = 3
    channels = 2
    height = 10
    width = 5
    sections = [3, 2, 5]
    dim = 2  # split across height
    raw_data = np.random.rand(batch, channels, height, width).astype("float32")

    class SplitModelSectionsList(nn.Module):
        def __init__(self, split_size, dim):
            super().__init__()
            self.split_size = split_size
            self.dim = dim

        def forward(self, x):
            return torch.split(x, split_size_or_sections=self.split_size, dim=self.dim)

    torch_module = SplitModelSectionsList(split_size=sections, dim=dim).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_batch_norm0(target, dev):
    # Eval, no momentum, no affine, no running stats
    raw_data = np.random.randn(8, 3, 4, 4).astype(np.float32)
    torch_module = nn.BatchNorm2d(
        3, eps=1e-02, momentum=0.0, affine=False, track_running_stats=False, device=None, dtype=None
    ).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_batch_norm1(target, dev):
    # Eval, with momentum, no affine, with running stats
    raw_data = np.random.randn(1, 4, 2, 2).astype(np.float32)
    torch_module = nn.BatchNorm2d(
        4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True, device=None, dtype=None
    ).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_batch_norm2(target, dev):
    # Eval, with momentum, affine, no running stats
    raw_data = np.random.randn(3, 4, 2, 2).astype(np.float32)
    torch_module = nn.BatchNorm2d(
        4, eps=1e-05, momentum=0.2, affine=True, track_running_stats=False
    ).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_batch_norm3(target, dev):
    # Eval, no momentum, affine, with running stats
    raw_data = np.random.randn(1, 2, 2, 2).astype(np.float32)
    torch_module = nn.BatchNorm2d(
        2, eps=1e-05, momentum=0.0, affine=True, track_running_stats=True
    ).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_chunk_even(target, dev):
    # Chunks is a divisor of the dimension size
    batch = 6
    channels = 2
    height = 3
    width = 4
    chunks = 3
    dim = 0
    raw_data = np.random.rand(batch, channels, height, width).astype("float32")

    class ChunkModel(nn.Module):
        def __init__(self, chunks, dim):
            super().__init__()
            self.chunks = chunks
            self.dim = dim

        def forward(self, x):
            return x.chunk(self.chunks, dim=self.dim)

    torch_module = ChunkModel(chunks=chunks, dim=dim).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_chunk_uneven(target, dev):
    # Chunks is not a divisor of the dimension size
    batch = 2
    channels = 5
    height = 4
    width = 5
    chunks = 2
    dim = 1
    raw_data = np.random.rand(batch, channels, height, width).astype("float32")

    class ChunkModel(nn.Module):
        def __init__(self, chunks, dim):
            super().__init__()
            self.chunks = chunks
            self.dim = dim

        def forward(self, x):
            return x.chunk(self.chunks, dim=self.dim)

    torch_module = ChunkModel(chunks=chunks, dim=dim).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_chunk_too_many(target, dev):
    # If user asks for more chunks than the size of the dim, pytorch simply splits in sections of size 1
    batch = 1
    channels = 3
    height = 2
    width = 2
    chunks = 99
    dim = 1
    raw_data = np.random.rand(batch, channels, height, width).astype("float32")

    class ChunkModel(nn.Module):
        def __init__(self, chunks, dim):
            super().__init__()
            self.chunks = chunks
            self.dim = dim

        def forward(self, x):
            return x.chunk(self.chunks, dim=self.dim)

    torch_module = ChunkModel(chunks=chunks, dim=dim).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_arange(target, dev):
    # arange.default
    raw_data = np.array([0, 0, 0, 0, 0])

    class ArangeDefaultModel(nn.Module):
        def forward(self, x):
            return x + torch.arange(5)

    torch_module = ArangeDefaultModel().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    # arange.start
    raw_data = np.array([0, 0, 0])

    class ArangeStartModel(nn.Module):
        def forward(self, x):
            return x + torch.arange(1, 4)

    torch_module = ArangeStartModel().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

    # arange.start_step
    raw_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    class ArangeStartStopModel(nn.Module):
        def forward(self, x):
            return x + torch.arange(1, 2.5, 0.5, dtype=torch.float32)

    torch_module = ArangeStartStopModel().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_index_select(target, dev):
    class IndexSelectModel(nn.Module):
        def forward(self, x):
            indices = torch.tensor([0, 2])
            return torch.index_select(x, 0, indices)

    raw_data = np.random.rand(3, 4).astype("float32")
    torch_module = IndexSelectModel().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_stack(target, dev):
    class StackModel(nn.Module):
        def forward(self, x):
            val1 = x[1, 4]
            val2 = x[3, 2]
            val3 = x[5, 6]
            z = torch.stack([val1, val2, val3])
            return z

    torch_module = StackModel().eval()
    raw_data = np.random.rand(10, 10, 10).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_sum(target, dev):
    class SumModel(nn.Module):
        def forward(self, x):
            new_vec = x[1, 4]
            return new_vec.sum()

    torch_module = SumModel().eval()

    raw_data = np.random.rand(10, 10, 10).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


# 1. Test for self._leakyrelu_module using torch.nn.LeakyReLU
@tvm.testing.parametrize_targets("cuda")
def test_leakyrelu_module(target, dev):
    class LeakyReLUModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.LeakyReLU(negative_slope=0.1)
        def forward(self, x):
            return self.act(x)
    raw_data = np.random.randn(2, 3).astype(np.float32)
    torch_module = LeakyReLUModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 2. Test for self._log_softmax_module using torch.nn.LogSoftmax
@tvm.testing.parametrize_targets("cuda")
def test_log_softmax_module(target, dev):
    class LogSoftmaxModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.logsoftmax = nn.LogSoftmax(dim=1)
        def forward(self, x):
            return self.logsoftmax(x)
    raw_data = np.random.randn(4, 5).astype(np.float32)
    torch_module = LogSoftmaxModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 3. Test for self._softmax_module using torch.nn.Softmax
@tvm.testing.parametrize_targets("cuda")
def test_softmax_module(target, dev):
    class SoftmaxModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax(dim=1)
        def forward(self, x):
            return self.softmax(x)
    raw_data = np.random.randn(4, 5).astype(np.float32)
    torch_module = SoftmaxModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 4. Test for self._adaptive_avg_pool2d_module using nn.AdaptiveAvgPool2d
@tvm.testing.parametrize_targets("cuda")
def test_adaptive_avg_pool2d_module(target, dev):
    class AdaptiveAvgPool2dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        def forward(self, x):
            return self.pool(x)
    raw_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
    torch_module = AdaptiveAvgPool2dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 5. Test for self._avg_pool2d_module using nn.AvgPool2d
@tvm.testing.parametrize_targets("cuda")
def test_avg_pool2d_module(target, dev):
    class AvgPool2dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=2)
        def forward(self, x):
            return self.pool(x)
    raw_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
    torch_module = AvgPool2dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 6. Test for self._batch_norm_2d_module using nn.BatchNorm2d
@tvm.testing.parametrize_targets("cuda")
def test_batch_norm_2d_module(target, dev):
    class BatchNorm2dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)
        def forward(self, x):
            return self.bn(x)
    raw_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
    torch_module = BatchNorm2dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 7. Test for self._conv1d_module using nn.Conv1d
@tvm.testing.parametrize_targets("cuda")
def test_conv1d_module(target, dev):
    class Conv1dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(in_channels=3, out_channels=4, kernel_size=3)
        def forward(self, x):
            return self.conv(x)
    raw_data = np.random.randn(2, 3, 10).astype(np.float32)
    torch_module = Conv1dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 8. Test for self._conv2d_module using nn.Conv2d
@tvm.testing.parametrize_targets("cuda")
def test_conv2d_module(target, dev):
    class Conv2dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
        def forward(self, x):
            return self.conv(x)
    raw_data = np.random.randn(2, 3, 10, 10).astype(np.float32)
    torch_module = Conv2dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 9. Test for self._conv3d_module using nn.Conv3d
@tvm.testing.parametrize_targets("cuda")
def test_conv3d_module(target, dev):
    class Conv3dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(in_channels=2, out_channels=3, kernel_size=3)
        def forward(self, x):
            return self.conv(x)
    raw_data = np.random.randn(1, 2, 8, 8, 8).astype(np.float32)
    torch_module = Conv3dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 10. Test for self._conv_transpose1d_module using nn.ConvTranspose1d
@tvm.testing.parametrize_targets("cuda")
def test_conv_transpose1d_module(target, dev):
    class ConvTranspose1dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_transpose = nn.ConvTranspose1d(in_channels=3, out_channels=2, kernel_size=3)
        def forward(self, x):
            return self.conv_transpose(x)
    raw_data = np.random.randn(2, 3, 10).astype(np.float32)
    torch_module = ConvTranspose1dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 11. Test for self._conv_transpose2d_module using nn.ConvTranspose2d
@tvm.testing.parametrize_targets("cuda")
def test_conv_transpose2d_module(target, dev):
    class ConvTranspose2dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_transpose = nn.ConvTranspose2d(in_channels=3, out_channels=2, kernel_size=3)
        def forward(self, x):
            return self.conv_transpose(x)
    raw_data = np.random.randn(2, 3, 10, 10).astype(np.float32)
    torch_module = ConvTranspose2dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 12. Test for self._cross_entropy_module using nn.CrossEntropyLoss
@tvm.testing.parametrize_targets("cuda")
def test_cross_entropy_module(target, dev):
    class CrossEntropyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.criterion = nn.CrossEntropyLoss()
            # For a batch of 4 with 3 classes.
            self.register_buffer("target", torch.tensor([0, 1, 2, 1]))
        def forward(self, x):
            return self.criterion(x, self.target)
    raw_data = np.random.randn(4, 3).astype(np.float32)
    torch_module = CrossEntropyModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 13. Test for self._group_norm_module using nn.GroupNorm
@tvm.testing.parametrize_targets("cuda")
def test_group_norm_module(target, dev):
    class GroupNormModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.gn = nn.GroupNorm(num_groups=1, num_channels=4)
        def forward(self, x):
            return self.gn(x)
    raw_data = np.random.randn(2, 4, 8, 8).astype(np.float32)
    torch_module = GroupNormModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 14. Test for self._layer_norm_module using nn.LayerNorm
@tvm.testing.parametrize_targets("cuda")
def test_layer_norm_module(target, dev):
    class LayerNormModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(normalized_shape=8)
        def forward(self, x):
            # x is assumed to have last dimension of size 8.
            return self.ln(x)
    raw_data = np.random.randn(2, 4, 8).astype(np.float32)
    torch_module = LayerNormModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 15. Test for self._linear_module using nn.Linear
@tvm.testing.parametrize_targets("cuda")
def test_linear_module(target, dev):
    class LinearModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        def forward(self, x):
            return self.linear(x)
    raw_data = np.random.randn(4, 10).astype(np.float32)
    torch_module = LinearModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 16. Test for self._max_pool2d_module using nn.MaxPool2d
@tvm.testing.parametrize_targets("cuda")
def test_max_pool2d_module(target, dev):
    class MaxPool2dModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=2)
        def forward(self, x):
            return self.pool(x)
    raw_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
    torch_module = MaxPool2dModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 17. Test for self._embedding_module using nn.Embedding
@tvm.testing.parametrize_targets("cuda")
def test_embedding_module(target, dev):
    class EmbeddingModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(num_embeddings=10, embedding_dim=3)
        def forward(self, x):
            return self.embed(x)
    # Raw data should be integer indices.
    raw_data = np.random.randint(0, 10, (2, 4)).astype(np.int64)
    torch_module = EmbeddingModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 18. Test for self._flatten_module using nn.Flatten
@tvm.testing.parametrize_targets("cuda")
def test_flatten_module(target, dev):
    class FlattenModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
        def forward(self, x):
            return self.flatten(x)
    raw_data = np.random.randn(2, 3, 4, 5).astype(np.float32)
    torch_module = FlattenModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 20. Test for self._numel using tensor.numel() wrapped as a tensor
@tvm.testing.parametrize_targets("cuda")
def test_numel(target, dev):
    class NumelModule(nn.Module):
        def forward(self, x):
            return torch.tensor(x.numel())
    raw_data = np.random.randn(2, 3, 4).astype(np.float32)
    torch_module = NumelModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 21. Test for self._scatter using torch.scatter
@tvm.testing.parametrize_targets("cuda")
def test_scatter(target, dev):
    class ScatterModule(nn.Module):
        def forward(self, x):
            # Assume x is of shape (2, 3)
            index = torch.tensor([[0, 1, 0], [1, 0, 1]])
            src = torch.full_like(x, 10.0)
            return x.scatter(dim=1, index=index, src=src)
    raw_data = np.random.randn(2, 3).astype(np.float32)
    torch_module = ScatterModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 22. Test for self._size using tensor.size() wrapped in a tensor
@tvm.testing.parametrize_targets("cuda")
def test_size(target, dev):
    class SizeModule(nn.Module):
        def forward(self, x):
            return torch.tensor(x.size(0))
    raw_data = np.random.randn(5, 4).astype(np.float32)
    torch_module = SizeModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 23. Test for self._sort using torch.sort
@tvm.testing.parametrize_targets("cuda")
def test_sort(target, dev):
    class SortModule(nn.Module):
        def forward(self, x):
            sorted_x, indices = torch.sort(x, dim=1)
            return sorted_x, indices
    raw_data = np.random.randn(3, 5).astype(np.float32)
    torch_module = SortModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


# 25. Test for self._full using torch.full
@tvm.testing.parametrize_targets("cuda")
def test_full(target, dev):
    class FullModule(nn.Module):
        def forward(self, x):
            return torch.full((2, 3), 5.0)
    raw_data = np.zeros((1,))  # dummy input; not used
    torch_module = FullModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 26. Test for self._inplace_masked_fill using tensor.masked_fill_
@tvm.testing.parametrize_targets("cuda")
def test_inplace_masked_fill(target, dev):
    class InplaceMaskedFillModule(nn.Module):
        def forward(self, x):
            mask = x > 0.5
            x.masked_fill_(mask, -1.0)
            return x
    raw_data = np.random.rand(3, 3).astype(np.float32)
    torch_module = InplaceMaskedFillModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 27. Test for self._ones using torch.ones
@tvm.testing.parametrize_targets("cuda")
def test_ones(target, dev):
    class OnesModule(nn.Module):
        def forward(self, x):
            return torch.ones((2, 3))
    raw_data = np.zeros((1,), ).astype(np.float32)
    torch_module = OnesModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 28. Test for self._tensor using torch.tensor
@tvm.testing.parametrize_targets("cuda")
def test_tensor(target, dev):
    class TensorModule(nn.Module):
        def forward(self, x):
            return torch.tensor([1, 2, 3])
    raw_data = np.zeros((1,))  # dummy input; not used
    torch_module = TensorModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 29. Test for self._type using tensor.type()
@tvm.testing.parametrize_targets("cuda")
def test_type(target, dev):
    class TypeModule(nn.Module):
        def forward(self, x):
            return x.type(torch.float64)
    raw_data = np.random.randn(2, 3).astype(np.float32)
    torch_module = TypeModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 30. Test for self._float using tensor.float()
@tvm.testing.parametrize_targets("cuda")
def test_float(target, dev):
    class FloatModule(nn.Module):
        def forward(self, x):
            return x.float()
    raw_data = np.random.randn(2, 3).astype(np.float32)
    torch_module = FloatModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 31. Test for self._half using tensor.half()
@tvm.testing.parametrize_targets("cuda")
def test_half(target, dev):
    class HalfModule(nn.Module):
        def forward(self, x):
            return x.half()
    raw_data = np.random.randn(2, 3).astype(np.float32)
    torch_module = HalfModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 32. Test for self._is_floating_point using tensor.is_floating_point()
@tvm.testing.parametrize_targets("cuda")
def test_is_floating_point(target, dev):
    class IsFloatingPointModule(nn.Module):
        def forward(self, x):
            return torch.tensor(x.is_floating_point())
    raw_data = np.random.randn(2, 3).astype(np.float32)
    torch_module = IsFloatingPointModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 33. Test for self._getattr using getattr on a tensor
@tvm.testing.parametrize_targets("cuda")
def test_getattr(target, dev):
    class GetAttrModule(nn.Module):
        def forward(self, x):
            # Use getattr to call the ndimension method.
            return torch.tensor(getattr(x, "ndimension")())
    raw_data = np.random.randn(2, 3, 4).astype(np.float32)
    torch_module = GetAttrModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 34. Test for self._sym_size_int using a symbolic size (here, x.shape[1])
@tvm.testing.parametrize_targets("cuda")
def test_sym_size_int(target, dev):
    class SymSizeIntModule(nn.Module):
        def forward(self, x):
            return torch.tensor(x.shape[1])
    raw_data = np.random.randn(2, 3, 4).astype(np.float32)
    torch_module = SymSizeIntModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 35. Test for self._interpolate using torch.nn.functional.interpolate
@tvm.testing.parametrize_targets("cuda")
def test_interpolate(target, dev):
    class InterpolateModule(nn.Module):
        def forward(self, x):
            # Upsample to a fixed size.
            return F.interpolate(x, size=(16, 16), mode="nearest")
    raw_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
    torch_module = InterpolateModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)

# 36. Test for self._lerp using torch.lerp
@tvm.testing.parametrize_targets("cuda")
def test_lerp(target, dev):
    class LerpModule(nn.Module):
        def forward(self, x):
            # Lerp between x and a constant tensor (2.0), with weight 0.3.
            return torch.lerp(x, torch.full_like(x, 2.0), 0.3)
    raw_data = np.random.randn(2, 3).astype(np.float32)
    torch_module = LerpModule().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


if __name__ == "__main__":
    tvm.testing.main()
