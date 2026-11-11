# DRAFT FOR REVIEW. No shared aisbench preset files are modified.
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as _base_models
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str_perf import gsm8k_datasets as _base_datasets
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer

models = [
    dict(
        _base_models[0],
        abbr="dsv4flash-gsm8k-off",
        path="/mnt/share/weight/DeepSeek-V4-Flash",
        model="dsv4flash_gsm8k_off",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=1,  # This aisbench version counts total attempts; 1 means no retries.
        api_key="",
        host_ip="127.0.0.1",
        host_port=8444,
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
        abbr="gsm8k-test",
        path="/mnt/share/x00893896/vllm_ascend_moonep_integration/benchmark/ais_bench/datasets/gsm8k",
    )
]
# Use all of the test split with the stock {question} performance template.
