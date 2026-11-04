# aisbench 原文完整性修复：配置变更确认

状态：等待用户确认，尚未应用候选配置、未重启实验。8卡与8444端口已释放。

## 已发现的问题及处理

第一台off服务已通过实际参数、8个worker环境/映射库核验，确认未启用MoonEP；128K长度与24GiB KV起服成功。最长130285-token请求成功输出1token，耗时约13.15秒，此验证请求不计正式结果。

aisbench第一轮完成316条请求，均有响应。但逐字核验发现3条代码样本的原文字面量 {idx} 被工具替换成了样本编号。原因是aisbench的 safe_format() 按entry各字段逐次 str.replace()，会再次处理刚插入context中的同名占位符。受影响的子集0基编号为144、151、152。

这一整轮已标记排除，不用于off/slot性能对照；所有原始JSONL、数据库、debug日志均保留。未启动Slot服务，也没有可报告的有效off/slot对照结果。

原始LongBench-v2和316条筛选数据没有修改。此问题是客户端prompt格式化导致的输入变化，与服务模式开关无关。

## 推荐变更：保留316条全文，单次替换模板

1. 新增本实验目录内 literal_longbench_prompt.py。对原模板结构做一次格式化，将context/问题/选项插入后，不再扫描这些原文中的花括号。
2. 两份aisbench配置把 prompt_template.type 换为 LiteralLongBenchPrompt；模板文字、数据、模型、并发、输出长度等都不变。
3. 两份压测sh的内部PYTHONPATH增加实验目录，以加载该本地类。仍不需要调用者设置任何环境变量。
4. 不修改aisbench公共源码，其他实验不受影响。
5. 已用mmengine加载两份候选配置、通过真实registry构造模板，并对316条prompt逐字核验全部一致，记录在 PROMPT_FIX_VALIDATION.json。
6. 如确认，应用候选配置后，按原计划重新执行off→slot、slot→off两对独立起服，每台3轮取后2轮。此前不完整输入的轮次不计入最终结果。

候选配置为 aisbench_off_proposed.py 与 aisbench_slot_proposed.py，当前活动配置尚未换用本修复；两份压测sh已设置待确认保护。

## 另已修正的执行问题

- 起服sh曾覆盖CANN的PYTHONPATH，导致acl导入失败。现已保留脚本内部source CANN得到的路径，off已成功起服。
- 本版本aisbench的retry表示总尝试次数。已从0修正为1，实现每条请求只尝试一次、失败不重试。首次0次尝试的预热未发送实际请求。
- 原始JSONL中可合法包含Unicode行分隔符；校验器现按物理LF逐行读取，避免误切JSON字符串。

这些记录与失败日志均保留在执行目录 data/executions/run_20261104_114303。

## 候选文件全文

下面由候选文件直接生成。此次需要确认的是模板类型与本地模块路径的变化。

### literal_longbench_prompt.py

```python
"""Experiment-only formatter: insert fields once; preserve literal braces in data."""
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.utils.prompt import PromptList

def _render_once(value, entry):
    if isinstance(value, str):
        return value.format(**entry)
    if isinstance(value, PromptList):
        return PromptList([_render_once(item, entry) for item in value])
    if isinstance(value, list):
        return [_render_once(item, entry) for item in value]
    if isinstance(value, dict):
        return {key: _render_once(item, entry) for key, item in value.items()}
    return value

class LiteralLongBenchPrompt(PromptTemplate):
    """Only the zero-shot LongBench template used by this experiment."""
    def generate_item(self, entry, output_field=None,
                      output_field_replace_token="", ice_field_replace_token=""):
        if self.ice_token is not None or self.sep_token is not None:
            raise ValueError("This experiment uses a zero-shot template without ICE/separator tokens.")
        if self.prompt_type != "meta":
            raise ValueError("Expected LongBench meta prompt template.")
        fields = dict(entry)
        if output_field is not None:
            fields[output_field] = output_field_replace_token
        encoded = self._encode_template(self.template, ice=False)
        return _render_once(encoded, fields)
```

### aisbench_off_proposed.py

```python
# DRAFT FOR REVIEW. No shared aisbench preset files are modified.
from mmengine.config import read_base
from literal_longbench_prompt import LiteralLongBenchPrompt

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as _base_models
    from ais_bench.benchmark.configs.datasets.longbenchv2.longbenchv2_gen_75fbba import LongBenchv2_datasets as _base_datasets
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer

models = [
    dict(
        _base_models[0],
        abbr="dsv4flash-longbench-off",
        path="/mnt/share/weight/DeepSeek-V4-Flash",
        model="dsv4flash_longbench_off",
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
        abbr="longbenchv2-full-le128k",
        path="/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench/data/longbenchv2_full_le128k.json",
    )
]
# Original LongBench-v2 prompt, reader, and inferencer are preserved.
# No max_seq_len/truncate_prompt_tokens setting; no client-side truncation.
# CLI supplies --mode perf --debug and a fresh --work-dir for every round.

# Proposed: preserve literal placeholders inside original context/code text.
datasets[0]["infer_cfg"] = dict(
    _base_datasets[0]["infer_cfg"],
    prompt_template=dict(_base_datasets[0]["infer_cfg"]["prompt_template"], type=LiteralLongBenchPrompt),
)
```

### aisbench_slot_proposed.py

```python
# DRAFT FOR REVIEW. No shared aisbench preset files are modified.
from mmengine.config import read_base
from literal_longbench_prompt import LiteralLongBenchPrompt

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as _base_models
    from ais_bench.benchmark.configs.datasets.longbenchv2.longbenchv2_gen_75fbba import LongBenchv2_datasets as _base_datasets
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer

models = [
    dict(
        _base_models[0],
        abbr="dsv4flash-longbench-slot",
        path="/mnt/share/weight/DeepSeek-V4-Flash",
        model="dsv4flash_longbench_slot",
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
        abbr="longbenchv2-full-le128k",
        path="/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_dp8_longbenchv2_aisbench/data/longbenchv2_full_le128k.json",
    )
]
# Original LongBench-v2 prompt, reader, and inferencer are preserved.
# No max_seq_len/truncate_prompt_tokens setting; no client-side truncation.
# CLI supplies --mode perf --debug and a fresh --work-dir for every round.

# Proposed: preserve literal placeholders inside original context/code text.
datasets[0]["infer_cfg"] = dict(
    _base_datasets[0]["infer_cfg"],
    prompt_template=dict(_base_datasets[0]["infer_cfg"]["prompt_template"], type=LiteralLongBenchPrompt),
)
```

### 两份压测sh的单行改动

```diff
-export PYTHONPATH="$BENCHMARK_DIR"
+export PYTHONPATH="$EXPERIMENT_DIR:$BENCHMARK_DIR"
```
