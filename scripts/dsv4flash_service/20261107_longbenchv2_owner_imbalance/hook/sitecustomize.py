"""Diagnostic capture of real prompt owner counts before native MoE dispatch."""
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import re
import sys

CAPTURE_DIR = os.environ.get("LB_OWNER_CAPTURE_DIR")
LAYERS, TOP_K, OWNERS, EXPERTS_PER_OWNER = 43, 6, 16, 16
REQUEST_ID = re.compile(r"^cmpl-(lb[a-z0-9]+)-(\d{4})-0(?:-|$)")
active = None


def patch_runner(module):
    cls = module.NPUModelRunner
    prepare_inputs = cls._prepare_inputs
    model_forward = cls._model_forward
    execute_model = cls.execute_model

    @functools.wraps(execute_model)
    def execute(self, *args, **kwargs):
        self._lb_owner_plan = None
        try:
            return execute_model(self, *args, **kwargs)
        finally:
            self._lb_owner_plan = None

    @functools.wraps(prepare_inputs)
    def prepare(self, scheduler_output, num_scheduled_tokens):
        result = prepare_inputs(self, scheduler_output, num_scheduled_tokens)
        batch = self.input_batch
        segments, cursor = [], 0
        for index, (request_id, scheduled) in enumerate(zip(batch.req_ids, num_scheduled_tokens, strict=True)):
            scheduled = int(scheduled)
            match = REQUEST_ID.match(request_id)
            if match:
                prompt_start = int(batch.num_computed_tokens_cpu[index])
                prompt_len = int(batch.num_prompt_tokens[index])
                count = max(0, min(scheduled, prompt_len - prompt_start))
                if count:
                    segments.append((match[1], int(match[2]), prompt_start, cursor, count))
            cursor += scheduled
        if segments:
            config = self.parallel_config
            assert config.tensor_parallel_size == 1 and config.pipeline_parallel_size == 1
            assert config.data_parallel_size == OWNERS and self.pcp_size == 1
            assert not self.dynamic_eplb and cursor == scheduler_output.total_num_scheduled_tokens
            self._lb_owner_plan = (config.data_parallel_rank, cursor, segments)
        return result

    @functools.wraps(model_forward)
    def forward(self, *args, **kwargs):
        global active
        plan = getattr(self, "_lb_owner_plan", None)
        if plan is None:
            return model_forward(self, *args, **kwargs)
        self._lb_owner_plan = None
        rank, size, segments = plan
        assert active is None
        active = {"size": size, "layer": 0, "buffer": None}
        try:
            result = model_forward(self, *args, **kwargs)
            assert active["layer"] == LAYERS, (rank, active["layer"], LAYERS)
            values = active["buffer"].cpu().numpy()
            import numpy as np
            output = Path(CAPTURE_DIR) / f"rank{rank:02d}.jsonl"
            with output.open("a", encoding="utf-8") as stream:
                for tag, request, prompt_start, offset, count in segments:
                    ids = values[:, offset:offset + count, :]
                    assert ids.min() >= 0 and ids.max() < OWNERS * EXPERTS_PER_OWNER
                    owners = np.bincount((ids // EXPERTS_PER_OWNER).reshape(-1), minlength=OWNERS)
                    row = dict(tag=tag, request=request, source_rank=rank,
                               start=prompt_start, tokens=count, owner_pairs=owners.tolist())
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            return result
        finally:
            active = None

    cls.execute_model = execute
    cls._prepare_inputs = prepare
    cls._model_forward = forward


def patch_moe(module):
    method = module.MoECommMethod
    fused = method.fused_experts

    @functools.wraps(fused)
    def capture(self, fused_experts_input):
        if active is not None:
            import torch
            assert isinstance(self, module.AlltoAllCommImpl), type(self)
            ids = fused_experts_input.topk_ids
            physical = fused_experts_input.routing.log2phy
            if physical is not None:
                ids = physical[ids]
            size, layer = active["size"], active["layer"]
            assert ids.ndim == 2 and ids.shape[1] == TOP_K and ids.shape[0] >= size
            assert layer < LAYERS
            if active["buffer"] is None:
                active["buffer"] = torch.empty(
                    (LAYERS, size, TOP_K), dtype=torch.int32, device=ids.device
                )
            active["buffer"][layer].copy_(ids[:size])
            active["layer"] += 1
        return fused(self, fused_experts_input)

    method.fused_experts = capture


class CaptureImports(importlib.abc.MetaPathFinder):
    patches = {
        "vllm_ascend.worker.model_runner_v1": patch_runner,
        "vllm_ascend.ops.fused_moe.moe_comm_method": patch_moe,
    }

    def find_spec(self, fullname, path=None, target=None):
        patch = self.patches.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        load_module = spec.loader.exec_module

        def load_and_patch(module):
            load_module(module)
            patch(module)

        spec.loader.exec_module = load_and_patch
        return spec


if CAPTURE_DIR:
    sys.meta_path.insert(0, CaptureImports())
