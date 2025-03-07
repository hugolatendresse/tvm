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
# pylint: disable=invalid-name
"""Reduction rule for operators including softmax, layer norm, RMS norm, etc"""
from typing import List, Union

from tvm import arith, tir
from tvm.target import Target

from ..analysis import normalize_prim_func
from ..base import try_inline_contiguous_spatial
from .base import GPUScheduleRule


class GeneralReduction(GPUScheduleRule):
    """General Reduction rule for operators including softmax, layer norm, RMS norm, etc"""

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        if target.kind.name == "cuda":
            len_tx = 256
            unroll_depth = 256
            # Maximum shared memory size per thread block for CUDA (in elements, assuming float32)
            max_shared_mem_elements = 12000  # ~48KB / 4 bytes per float
        elif target.kind.name == "opencl":
            len_tx = 256
            unroll_depth = 64
            max_shared_mem_elements = 8000  # ~32KB / 4 bytes per float
        else:
            len_tx = 64
            unroll_depth = 64
            max_shared_mem_elements = 4000  # ~16KB / 4 bytes per float

        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) == 0:
            return None

        dom_kind = block_infos[0].dom_kind()
        num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
        num_trailing_r = len(dom_kind) - len(dom_kind.rstrip("R"))

        # Align the number of block iters of the last block.
        num_last_block_iter = len(block_infos[-1].dom_kind())
        if num_last_block_iter < len(dom_kind):

            def f_layout_mapping(*iters):
                analyzer = arith.Analyzer()
                # Try to match the iters of last block to the iters of the first block.
                # For matched positions, use the iter from the input `iters`.
                # For unmatched positions, use a new iter which is constant 0.
                num_matched = 0
                target_layout_iters = []
                for block_iter in block_infos[0].iters:
                    if num_matched < len(iters) and analyzer.can_prove_equal(
                        block_iter.dom, block_infos[-1].iters[num_matched].dom
                    ):
                        target_layout_iters.append(iters[num_matched])
                        num_matched += 1
                    else:
                        target_layout_iters.append(tir.const(0, iters[0].dtype))

                # If all the iters of the last block can match, return the new layout.
                if num_matched == len(iters):
                    return target_layout_iters
                # Otherwise, fallback to appending zeros in the beginning.
                return [tir.const(0, iters[0].dtype)] * (
                    len(dom_kind) - num_last_block_iter
                ) + list(iters)

            index_map = tir.IndexMap.from_func(f_layout_mapping, ndim=num_last_block_iter)
            sch.transform_block_layout(block_infos[-1].block_rv, index_map)

        try:
            # TODO: fix num_leading_s = 0 case
            assert num_trailing_r > 0
            for block in block_infos[1:-1]:
                assert block.dom_kind() == dom_kind
            assert block_infos[-1].is_injective()
            assert len(block_infos[-1].dom_kind()) <= len(dom_kind)
        except AssertionError:
            return None

        # Check if we have a non-last dimension reduction
        is_non_last_dim_reduction = "R" in dom_kind[:-num_trailing_r]
        
        # Get the last dimension size
        # For softmax with shape [1, 4, 32, 8192], we need to know 8192
        last_dim_size = 1
        if len(block_infos[0].iters) > 0:
            last_dim_iter = block_infos[0].iters[-1]
            if hasattr(last_dim_iter.dom, 'extent'):
                extent = last_dim_iter.dom.extent
                if hasattr(extent, 'value'):
                    last_dim_size = extent.value

        # For large non-last dimension reductions, we need to use tiling to reduce shared memory usage
        use_tiling = is_non_last_dim_reduction and last_dim_size > max_shared_mem_elements
        
        # Set up normal scheduling when tiling isn't needed
        if not use_tiling:
            loops = sch.get_loops(block_infos[-1].block_rv)
            bx = sch.fuse(*loops[:num_leading_s])
            r_loop, tx = sch.split(loops[-1], [None, len_tx])
            sch.reorder(tx, r_loop)
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")
            sch.annotate(r_loop, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
            sch.annotate(r_loop, ann_key="pragma_unroll_explicit", ann_val=1)

            for block in reversed(block_infos[:-1]):
                block = block.block_rv
                for i, _ in enumerate(sch.get(block).writes):
                    sch.set_scope(block, buffer_index=i, storage_scope="shared")
                
                sch.compute_at(block, bx, preserve_unit_loops=True)
                r_loop = sch.fuse(*sch.get_loops(block)[-num_trailing_r:])
                r_loop, tx = sch.split(r_loop, [None, len_tx])
                sch.reorder(tx, r_loop)
                sch.bind(tx, "threadIdx.x")
                sch.annotate(r_loop, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
                sch.annotate(r_loop, ann_key="pragma_unroll_explicit", ann_val=1)
        else:
            # For large tensors with non-last dimension reduction, implement tiling approach
            # This ensures shared memory usage stays within limits
            
            # Calculate tile size to ensure shared memory usage is reasonable
            # We need two arrays for reduction, so divide the max by 2
            tile_size = max_shared_mem_elements // 2
            if tile_size > 1024:
                tile_size = 1024  # Cap at 1024 for better performance

            # Get loops for the final block
            loops = sch.get_loops(block_infos[-1].block_rv)
            
            # Split the spatial loops into block-level parallelism
            bx = sch.fuse(*loops[:num_leading_s])
            
            # For the last dimension, we'll use a tiled approach
            # Split it into tiles and process each tile with shared memory
            last_loop = loops[-1]
            outer, inner = sch.split(last_loop, [None, tile_size])
            
            # Set up the execution order and bindings
            sch.bind(bx, "blockIdx.x")
            
            # Further split the inner loop for thread parallelism
            inner, tx = sch.split(inner, [None, len_tx])
            sch.reorder(tx, inner)
            sch.bind(tx, "threadIdx.x")
            
            # Apply unrolling for better performance
            sch.annotate(inner, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
            sch.annotate(inner, ann_key="pragma_unroll_explicit", ann_val=1)
            
            # Set up the intermediate reduction blocks
            for block in reversed(block_infos[:-1]):
                block = block.block_rv
                
                # Use shared memory for the intermediate buffers
                for i, _ in enumerate(sch.get(block).writes):
                    sch.set_scope(block, buffer_index=i, storage_scope="shared")
                
                # Compute at the block level
                sch.compute_at(block, bx, preserve_unit_loops=True)
                
                # Get the reduction loops
                r_loops = sch.get_loops(block)[-num_trailing_r:]
                
                # If there are multiple reduction loops, handle them differently
                if len(r_loops) > 1:
                    # Fuse all but the last reduction loop
                    fused_r_loops = sch.fuse(*r_loops[:-1])
                    # Handle the last reduction loop separately
                    last_r_loop = r_loops[-1]
                    
                    # Split and reorder for tiling
                    outer_r, inner_r = sch.split(last_r_loop, [None, tile_size])
                    inner_r, tx_r = sch.split(inner_r, [None, len_tx])
                    sch.reorder(fused_r_loops, outer_r, tx_r, inner_r)
                    sch.bind(tx_r, "threadIdx.x")
                else:
                    # Only one reduction loop, split it for tiling
                    r_loop = r_loops[0]
                    outer_r, inner_r = sch.split(r_loop, [None, tile_size])
                    inner_r, tx_r = sch.split(inner_r, [None, len_tx])
                    sch.reorder(tx_r, inner_r)
                    sch.bind(tx_r, "threadIdx.x")
                
                # Apply unrolling
                sch.annotate(inner_r, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
                sch.annotate(inner_r, ann_key="pragma_unroll_explicit", ann_val=1)
                
                # Add memory synchronization if needed
                sch.annotate(block, ann_key="pragma_synchronize", ann_val=1)
                
        return sch