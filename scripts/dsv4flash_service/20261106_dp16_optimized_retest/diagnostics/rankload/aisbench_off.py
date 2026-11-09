# Optimized DP16 retest. No shared aisbench preset files are modified.
from mmengine.config import read_base
# Read-only reuse of the approved single-pass formatter; PYTHONPATH supplies its directory.
from literal_longbench_prompt import LiteralLongBenchPrompt

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as _base_models
    from ais_bench.benchmark.configs.datasets.longbenchv2.longbenchv2_gen_75fbba import LongBenchv2_datasets as _base_datasets
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer

models = [
    dict(
        _base_models[0],
        abbr="dsv4flash-dp16-longbench-off",
        path="/mnt/share/weight/DeepSeek-V4-Flash",
        model="dsv4flash_dp16_longbench_off",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=1,  # This aisbench version counts total attempts; 1 means no retries.
        api_key="",
        host_ip="141.61.45.118",
        host_port=8445,
        url="",
        max_out_len=1,
        batch_size=16,
        trust_remote_code=False,
        generation_kwargs=dict(temperature=0, ignore_eos=True, seed=1024),
    )
]
datasets = [
    dict(
        _base_datasets[0],
        abbr="longbenchv2-all503-midcut128k",
        path="/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp16_longbenchv2_aisbench_draft/data/longbenchv2_all503_midcut128k.json",
    )
]
# Original LongBench-v2 prompt, reader, and inferencer are preserved.
# No max_seq_len/truncate_prompt_tokens setting; no client-side truncation.
# CLI supplies --mode perf --debug and a fresh --work-dir for every round.

# Approved original-text fix: preserve literal placeholders inside original context/code text.
datasets[0]["infer_cfg"] = dict(
    _base_datasets[0]["infer_cfg"],
    prompt_template=dict(_base_datasets[0]["infer_cfg"]["prompt_template"], type=LiteralLongBenchPrompt),
)
