# 跨仓修复记录：背景、根因与解法

本仓（`vllm_ascend_moonep`，`vllm-ascend` 的 fork）能跑起来，还依赖几处**落在本仓之外**的修复。
它们目前只存在于本机的工作区里（各自上游未提交，见下表的「状态」），
`git submodule update`、换机器或同步上游都会把它们冲掉，所以在这里逐条记下：
现象是什么、为什么会这样、怎么改、怎么验证、还剩什么没做。

记录日期：2026-09-14 ~ 2026-09-15。环境：8 × Ascend950PR、CANN 9.2.0、
vllm-ascend `releases/v0.23.0`（本仓为其 fork）、vllm v0.23.0、ascend-moonep `urma_dev`。

本地仓库布局（下文路径均相对此根）：

```
/home/x00893896/vllm_ascend_moonep_integration/
├── vllm/                 # vllm-project/vllm，releases/v0.23.0
├── vllm-ascend/          # 上游 vllm-ascend，releases/v0.23.0（供对照）
├── vllm_ascend_moonep/   # 本仓（分支 moonep），内含子模块 csrc/third_party/catlass
├── ascend-moonep/        # gitcode jayhua/ascend-moonep（urma_dev），内含子模块 3rdparty/shmem
└── scripts/              # 本仓外部：服务化脚本，当前跑不起来、不作为入口
                          # （跑测入口是 vllm_ascend_moonep/scripts/ 下的驱动脚本）
```

## 索引

| # | 症状（可直接搜） | 所属仓 | 状态 |
|---|---|---|---|
| 1 | `error: simd vf function must be a free function or static member function` | `csrc/third_party/catlass`（本仓子模块） | 已实施并验证（子模块工作区，未提交上游） |
| 2 | `aclshmemx_init_attr_with_buffers failed with -100` | `ascend-moonep/3rdparty/shmem` | 已实施并验证（子模块工作区，未提交上游；新库已部署到两处） |
| 3 | `aclrtMemSetAccess(va=..., size=...) failed with 507899` | `ascend-moonep/csrc` + 两份 Python 层 | **一半已实施**（vendored Python 层）；C++ 侧待批准 |
| 4 | DSv4 的 W4A8 路径 `AttributeError`，找 `pad_and_split_input_ids` | **本仓** `moonep_comm.py` | 已实施（见 commit `fix(moonep): delegate pad_and_split_input_ids`） |
| 5 | `errcode:22007 / 507018 aicpu exception`，`kernelName=KvQuantSparseAttnSharedkvMetadata` | 上游 A5 稀疏注意力算子 + 模型配置 | **A5 侧待确认**（moonep/baseline 两条腿都一样，与本仓无关）；DP2+TP1 可绕开头数约束，**DSV4-Flash + MoonEP 已按此端到端跑通**（跨 engine 挂死根因是 `expandable_segments`，改 `False` 即可，见 §5 末） |

---

## 1. CANN 9.2.0 下 catlass 编译失败

**现象**

构建 `vllm_ascend` 的算子内核时报：

```
error: simd vf function must be a free function or static member function
```

报错落在 `chunk_gated_delta_rule_fwd_h` 等算子。

**背景**

本仓基于 vllm-ascend `releases/v0.23.0`，官方要求 `CANN == 9.1.0`，本机是 **CANN 9.2.0**
（毕昇 ccec 编译器 clang 15.0.5）。9.2.0 有一处非兼容变更：
`__simd_vf__`（`cce_simd_vf` 属性）修饰的函数必须是**自由函数或 static 成员函数**。

**根因**

本仓固定的 catlass 子模块（`41bf90da`，`v1.2.0-143-g41bf90da`）里，4 个头文件的 VF 函数写成了
**普通成员函数**。上游已在 catlass `c89fe73d`（commit 标题：`针对VF函数补充 static 修饰`，该 commit 对象
在本地子模块里可查）修掉，但该 commit 与本 pin 之间的分歧太大（改动横跨 fai/epilogue/tile 十几个头文件），
**不能直接 cherry-pick**。

**解法**

直接改子模块（这是唯一的低成本路径，子模块会因此保持 dirty）：

| 文件 | 改法 | 个数 |
|---|---|---|
| `include/catlass/epilogue/block/block_epilogue_fa_softmax_ascend950.hpp` | 类内定义加 `static` | 4 |
| `include/catlass/epilogue/block/block_epilogue_fa_rescale_o_ascend950.hpp` | 同上 | 3 |
| `include/catlass/epilogue/tile/tile_pertoken_dequant.hpp` | 同上 | 1 |
| `include/catlass/epilogue/block/block_epilogue_per_group_per_block.hpp` | ccec **不允许 VF 成员函数在类外定义**，需先把 3 个类外定义（`AivPerTensor` ×2、`AddBias`）搬进类内，再加 `static` | 3 |

前 3 个文件是在这四个文件里给函数签名补 `static`（共 **8 个**函数）：
softmax 的 4 个与 rescale_o 的 3 个签名形如 `__simd_vf__ inline void` → `__simd_vf__ inline static void`，
`tile_pertoken_dequant.hpp` 那 1 个没有 `inline`，是 `__simd_vf__ void` → `__simd_vf__ static void`；
第 4 处是一次「类外定义 → 类内定义」的结构调整（`AivPerTensor` ×2、`AddBias`，共 3 个），
用脚本做（见[附录 A](#附录-a重打-catlass-补丁的脚本)）。

**验证**

- 全量 ninja 通过（并行构建把报错淹没时，进 `csrc/build` 重跑增量 `ninja`，错误会出现在输出末尾）；
- `pip install -e . --no-build-isolation` 成功，装出 `0.1.dev4244+g56fc04ebe.d20260914`
  （setuptools-scm 就本仓 git describe 生成的版本串）；
- `import vllm_ascend` 正常。

**遗留与重打**

- 子模块处于 dirty 状态（`git -C csrc/third_party/catlass status --short` 显示 4 个 ` M`）；
- **`git submodule update` 会重置它，需要重新打补丁**；
- 根治办法：把 CANN 降到 9.1.0，或等上游升级 catlass 的固定版本。

**附带的两个经验**（都很费时间，所以写下来）

- 重装本仓必须 `pip install -e . --no-build-isolation`：主环境依赖已经装齐，走构建隔离会从
  pypi.org 重新拉 torch 等大包（实测 ~50 kB/s）。
- 安装会把 `numpy` 降到 1.26.4（vllm-ascend 的要求），与 `opencv-python-headless` 5.0 有软冲突，
  实测无影响。CANN 算子编译的详细日志在 `~/ascend/log/run/plog/`。

---

## 2. shmem 在 `ASCEND_RT_VISIBLE_DEVICES` 重映射下 init 返回 -100

**现象**

开启 `enable_moonep` 时，worker 起不来：

```
aclshmemx_init_attr_with_buffers failed with -100
```

`-100` = `ACLSHMEM_NOT_SUPPORTED`。shmem 自己的日志在 `/root/shmem/log/aclshmem_<pid>_*.log`。

**背景**

本机 8 张 Ascend950PR，卡 0/2/5/6 长期被其他任务占满，所以运行必然要用
`ASCEND_RT_VISIBLE_DEVICES` 挑空闲卡 → 该变量是**真子集**，**逻辑号 ≠ 物理号**。
另外 vLLM DP（`dp>1, tp=1`）会把每个 rank 进程收窄成单卡，同样造成逻辑号 ≠ 物理号。

**根因**

`ascend-moonep/3rdparty/shmem/src/host/init/shmem_init.cpp` 的 `query_buffer_property()`：
它把 `aclrtMemGetAllocationPropertiesFromHandle` 返回的 **物理** `location.id`
与 `aclrtGetDevice` 得到的 **逻辑** `current_device` 直接比较，不等就返回 `ACLSHMEM_NOT_SUPPORTED`。
重映射下这两个号**必然不等**，所以一选卡就 -100。

**解法（两段式，缺一不可）**

1. 把该校验从「硬失败」降级为 **WARN**（保留信息，不再中断）；
2. 校验通过后把 `property->location.id` **归一化回逻辑号** `current_device`。

第 2 步不能省：下游 `validate_buffer_mapping` 会拿这个 `location.id` 去调
`aclrtMemGetAllocationGranularity`，而那个接口要的是**逻辑**号；传物理号会在收窄后的上下文里
报 `107001`（`ACL_ERROR_RT_INVALID_DEVICEID`），最后还是以 -100 收场。

两处改动落在 `shmem_init.cpp:306-317`（注释 + `SHM_LOG_WARN` + `property->location.id = current_device;`）。

**重编与部署**

```bash
cd /home/x00893896/vllm_ascend_moonep_integration/ascend-moonep/3rdparty/shmem/build
make shmem -j
# 新的 libshmem.so 必须部署到两处：
#   3rdparty/shmem/install/shmem/lib/        <- shmem 自己的安装目录
#   ascend-moonep/ascend_moonep/lib/         <- _C.so 的 RUNPATH 是 $ORIGIN/lib，指向这里
```

旧库备份为 `libshmem.so.bak0914`（两处各一份）。当前的落地状态（已核实）：

| 位置 | `libshmem.so` | 备份 |
|---|---|---|
| `3rdparty/shmem/install/shmem/lib/` | 4109400 B，09-15 09:35 | `libshmem.so.bak0914`，4108888 B，09-14 21:26 |
| `ascend-moonep/ascend_moonep/lib/` | 4109400 B，09-15 09:35 | 同上 |

**关联知识**

- `libshmem.so` **自身没有 RPATH/RUNPATH**，它 NEEDED 的 `libshmem_utils.so` 要靠
  `LD_LIBRARY_PATH` 或 `ASCEND_MOONEP_BUILD_DIR`（vendored 包的 `__init__.py` 预加载机制）才能找到；
- CANN **没有公开的**「物理号 ↔ 逻辑号」转换符号，只能自己按 `ASCEND_RT_VISIBLE_DEVICES` 的列表换算。

**遗留**

改动只在子模块工作区（`git -C ascend-moonep status --short` 显示 ` m 3rdparty/shmem`），
未提交到上游；子模块一旦被重置，需要**重新打补丁 + 重编 + 重新部署到两处**。

---

## 3. moonep `aclrtMemSetAccess` 507899：设备号空间混用

**现象**

开启 `enable_moonep` 后，worker init 阶段直接崩：

```
ascend_moonep/api.py:233       create_vmm_physical([NvS, H], torch.bfloat16, device)
ascend_moonep/buffer_c.py:451  va, handle, shareable, size, fabric_bytes = acl_vmm_alloc_source(...)
RuntimeError: aclrtMemSetAccess(va=..., size=4194304) failed with 507899
```

`507899` = `ACL_ERROR_RT_DRV_INTERNAL_ERROR`，抛错点在
`ascend-moonep/csrc/acl_shared_buffer.cpp:401` 的 `acl_vmm_set_access()`。

**根因**

与第 2 条是**同一族问题**（物理号 / 逻辑号混用），但这次不是「跟谁比错了」，
而是**两个 CANN 接口要求的 id 空间本来就不一样**：

| CANN 接口 | 要哪个号 |
|---|---|
| `aclrtMemGetAllocationGranularity`、`aclrtMallocPhysical` | **逻辑**号（`ASCEND_RT_VISIBLE_DEVICES` 列表内的下标） |
| `aclrtMemSetAccess` 的 `aclrtMemLocation.id` | **物理**号（全局卡号） |

`ascend_moonep/api.py:195` 取的是 `torch.npu.current_device()`（逻辑号），一路原样传到
`acl_vmm_set_access`——那里直接把 `device_id` 填进 `desc.location.id`。于是设了
`ASCEND_RT_VISIBLE_DEVICES` 时，`set_access` 实际指向**另一张物理卡**
（例如 remap `6,7` 下传逻辑 0 → 打到物理 0，而物理 0 常被别人占满）→ 驱动内部错误。

**判据**（同一进程内实测，`ASCEND_RT_VISIBLE_DEVICES=6,7`、`DEV=0`）

| 调用 | 结果 |
|---|---|
| `acl_vmm_set_access(va, size, 0 /*逻辑号*/)` | `507899` |
| `acl_vmm_set_access(va, size, 6 /*物理号*/)` | **OK** |
| 不设 `ASCEND_RT_VISIBLE_DEVICES`（逻辑号 == 物理号） | 全链路 OK |
| `acl_vmm_alloc_source(size, 6)`（想让整条链都用物理号） | `107001`（granularity 那步非法） |

最后一行很关键：**不能**把物理号一路传下去，只能在需要物理号的那**一处**做转换。

**已经修好的一半（别重复劳动）**

本仓 **vendored** 的 Python 层已经带了修复，位于
`third_party/ascend_moonep/ascend_moonep/buffer_c.py`：

- `acl_vmm_access_device_id(device_id)`：按 `ASCEND_RT_VISIBLE_DEVICES` 把逻辑号翻成物理号；
- `_alloc_vmm_source_split_id(alloc_size, malloc_device, access_device)`：
  `acl_vmm_alloc_source` 的 Python 等价实现，malloc 用逻辑号、`set_access` 用物理号；
- `create_vmm_physical()` 里 **id 相等时仍走 C++ 实现**（不为常见路径引入分支差异），
  只在 id 不等时走上面那条；另外两处 `set_access` 调用点也全部做了翻译。

它最初的目的是 vLLM DP（`dp>1, tp=1` 把每个 rank 进程收窄成单卡）那个场景。

**2026-09-19 补记：本仓自己还有一处漏翻（slot 模式首次在 remap 卡上跑到 alias 时暴露）**

上面「三处都翻了」只覆盖 `buffer_c.py` **内部**的调用点。
`vllm_ascend/ops/fused_moe/moonep_balance.py` 的 `_packed_alias()` 是 `[epn+B]` alias 的
**另一份实现**（`create_full_projection_tensor` 是兄弟函数，不是同一个），它
`from ascend_moonep.buffer_c import acl_vmm_set_access` 之后把**逻辑号** `dev` 原样传下去：

```
moonep_comm.py:369 _LayerWeightState(...) → moonep_balance.py:473 _packed_alias
RuntimeError: aclrtMemSetAccess(va=..., size=536870912) failed with 507899
```

判据与原因：DP4 owner/slot 同一份 env、同样卡 4,5,6,7，**owner 模式全过**（它不建 alias）；
slot 模式在 alias 建 VA 处撞 507899，逻辑 0..3 被当物理号打到物理 0..3（`size=536870912`
= w13 在 epn=64 下的别名段）。**修法**与 `buffer_c` 完全一致：`set_access` 传
`acl_vmm_access_device_id(int(dev))`，`granularity`/`reserve`/`map` 与 tensor 的 `npu:{dev}`
仍用逻辑号；另加 `ImportError` 兜底（上游 buffer_c 没有该函数时退回 `int`，等价于未重映射）。
早先的 `smoke_slots.py`（R=2/3）跑在卡 0,1/0,1,2 上，逻辑号==物理号，所以一直没暴露。

**为什么仍然会撞 507899**

因为兄弟仓 `/home/x00893896/vllm_ascend_moonep_integration/ascend-moonep/ascend_moonep/buffer_c.py`
是**上游原样**（没有上述函数，`create_vmm_physical` 无条件调 `acl_vmm_alloc_source`），
而本机默认的 `import ascend_moonep` **解析到的正是它**（editable 安装）：

```bash
python -c "import ascend_moonep; print(ascend_moonep.__file__)"
# -> /home/x00893896/vllm_ascend_moonep_integration/ascend-moonep/ascend_moonep/__init__.py
```

所以**没有把 vendored 层放到 `PYTHONPATH` 最前**的运行，走的是未修复的那份。
（判据：复现的 traceback 行号 `buffer_c.py:451` 正是兄弟仓那份的行号；
vendored 那份的同一个调用在 518 行，而且外面有 id 是否相等的分支。）

**解法 A（运行期，推荐；路径已存在，端到端尚待验证）**

把 vendored 层放在 `PYTHONPATH` 最前，并用 `ASCEND_MOONEP_BUILD_DIR` 指到外部构建目录取
`_C*.so` 与 `lib/*.so`（本仓 `scripts/` 下的驱动脚本就是这么跑的）：

```bash
export PYTHONPATH=<repo>/third_party/ascend_moonep:$PYTHONPATH
export ASCEND_MOONEP_BUILD_DIR=/home/x00893896/vllm_ascend_moonep_integration/ascend-moonep/ascend_moonep

# 启动前自查：应当打印**本仓**的路径
python -c "import ascend_moonep, ascend_moonep._C as _C; print(ascend_moonep.__file__); print(_C.__file__)"
```

**解法 B（C++ 侧，已定位、未实施）**

若要让任何**直接调用 `_C`** 的路径也自洽，就在 `acl_vmm_set_access()`
（`ascend-moonep/csrc/acl_shared_buffer.cpp:401`）这一处把逻辑号翻成物理号——**只此一处**，
其余步骤（granularity、malloc_physical）继续用逻辑号。改完需要重编 `_C.so`
（`ascend_moonep/_C.cpython-312-x86_64-linux-gnu.so`，55 MB；**当前没有备份，动手前先备份**）。

> 本节只记录方案，**尚未实施**。这类改动按惯例：先确认、先备份。

**影响面**

只要可见设备是**真子集**（本机为避开他人占用的卡，这是必需的），moonep 链路的可用性就取决于
「用对的那份 Python 层」。与打点无关：关掉 `VLLM_ASCEND_MOE_PAR_STATS` 的对照跑同样崩在同一点。

---

## 4. DeepSeek-V4-Flash 用 moonep 拉起（唯一一处改动落在本仓）

**目标与链路**

用 moonep（SHMEM）通信拉起 `/mnt/data/DeepSeek-V4-Flash-0731`
（`DeepseekV4ForCausalLM`：fp8 块量化主干 + mxfp4 专家、E=256/K=6/H=4096、稀疏注意力 + HC + KV 压缩
+ DSPark + MTP）。链路：

```
DeepseekV4MoE → vllm FusedMoE →（0.23.0 走 REGISTERED_ASCEND_OPS）→ AscendFusedMoE
              → setup_moe_comm_method → enable_moonep 时 MoonEPCommImpl
量化方法：AscendW4A8MXFPDSDynamicFusedMoEMethod；moonep 的 dispatch/combine 只支持 bf16，
激活的 fp8 量化在专家侧 dispatch 之后做。
```

**现象与根因**

W4A8 量化路径下，`select_experts` → `_select_experts_with_fusion_ops` 需要 `input_ids`
（hash 路由融合算子要用），于是去调通信实现上的 `pad_and_split_input_ids`——
`MoonEPCommImpl` 没有这个方法，直接 `AttributeError`。
兄弟实现 `MC2` / `AllToAll` / `FusedMC2` 都有这个委托，MoonEP 漏了。

**解法**

在 `vllm_ascend/ops/fused_moe/moonep_comm.py` 的 `MoonEPCommImpl` 上补一个委托：

```python
def pad_and_split_input_ids(self, input_ids):
    return self.prepare_finalize.pad_and_split_input_ids(input_ids)
```

（`prepare_finalize` 是 `PrepareAndFinalizeWithAll2All`，本来就实现了这个接口。）

**约束（拉起这类模型时的前置条件）**

- `MOONEP_SLOT_MODE` 可开（W4A8MXFP 已打通到单层产品路径 bit-identical，端到端待 EP≥8 空卡；
  见 [moonep_slot_quant.md](moonep_slot_quant.md)）；
- 关闭 dynamic EPLB / mix_placement；
- token 数 ≤ `max_num_batched_tokens` 时走 SHMEM，超出会回落到 MC2 / all-to-all
  （所以 `method=` 列里可能混着出现，见 [moe_par.md](moe_par.md) 第 6 节）；
- 拉起前要有 `ASCEND_MOONEP_BUILD_DIR` 或 `LD_LIBRARY_PATH`（见第 2、3 条）；
- 用 `ASCEND_RT_VISIBLE_DEVICES` 选空闲卡。

**遗留**

- W4A8MXFP × moonep 的**正确性**还需要 A/B 对照（`scripts/moe_ab_test.py`，`MOONEP=0/1` 比 logprobs），
  尚未做；该模型的 A5 端到端当前另被**第 5 条**卡住（稀疏注意力只收每 rank 64/128 头）。
  MoE 侧的对照可先用 `Qwen3-30B-A3B`（TP=2，不走 DSA）：2026-09-18 已在 6/7 卡上跑完两条腿
  （`MOONEP=0/1`：两条腿的 top-5 token 集合一致，logprob 差多为 2e-4 量级、个别 0.12——
  该模型 logprob 落在 0.125 的粗网格上，属相邻格；贪心下 32 个 token 里 2 个不同，
  如 "20-year-old" / "22-year-old" 这种同分岔路），并用 `scripts/smoke_qwen_geo.py`
  在 6/7 卡上过了数值 harness（rel 0.0058 / 0.0056，判据 3e-2）；
- MTP / DSPark 尚未启用；
- 第 3 条不修（不改 `PYTHONPATH` 的前提下），这条链路在本机跑不起来。

---

## 5. A5 稀疏注意力只收 64/128 头：DeepSeek-V4-Flash 端到端被卡在这里

（记录日期 2026-09-18。A5 = Ascend950DT，CANN 9.2.0 / 驱动 25.6.rc1。）

**现象**

`MOONEP=1` 和 `MOONEP=0` 两条腿完全一样：跑过 profile、进入第一次真正生成（首个 attention
metadata）时挂，host plog 只有一句看不出原因的：

```
RUNTIME ... PrintAicpuErrorInfo: ... soName=7187735096501937334_5392192.so, funcName=RunCpuKernel,
            kernelName=KvQuantSparseAttnSharedkvMetadata, task_pos=0, aicpuKernelType=4
RUNTIME ... rtStreamSynchronizeWithTimeout:ErrCode=507018, desc=[aicpu exception]  (InnerCode 0x715002a)
device_error_proc: ... errcode:22007, msg:inner error   ← originResult[-1]，是算子自己 return false
```

**根因（真正的报错在 device 侧日志）**

```
/root/ascend/log/debug/device-<物理卡号>/device-<hostpid>_<时间戳>.log

[ERROR] AICPU(...) [kv_quant_sparse_attn_sharedkv_metadata_aicpu.cpp:103][CheckSingleParam]
        num_heads_q should be 64 or 128, but got 32
[ERROR] CCECPU(...) [ae_kernel_lib_aicpu.cc:251] call aicpu api RunCpuKernel in
        7187735096501937334_5392192.so failed, ret:4294967295
[ERROR] CCECPU(...) [aicpusd_event_manager.cpp:375] ... opName[KvQuantSparseAttnSharedkvMetadata],
        result[22007], originResult[-1]
```

两处校验卡住同一件事——**A5 的 DSA/量化稀疏注意力要求每个 rank 的 q 头数是 64 或 128**：

- metadata 算子（AICPU）
  `csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_kernel_aicpu/kv_quant_sparse_attn_sharedkv_metadata_aicpu.cpp:102`
- 消费者 attention 算子（AscendC，host 侧 tiling 校验）
  `csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check_feature.cpp:49`
  （`CheckFeatureAntiquantShape`：`n1Size_ != 64 && n1Size_ != 128`，`gSizeSupportList = {64,128}`）

而 `/mnt/weight/A5-weights/DeepSeek-V4-Flash` 是 `num_attention_heads=64`（`head_dim=512`、
`num_key_value_heads=1`），权重 149 GB / 单卡 HBM 98 GB → **TP 只能取 2** → 每 rank 32 头 → 两个算子都拒。
TP=1 能给出 64 头但 149 GB 放不下单卡。

**不是 moonep 的问题**：两条腿同一算子、同一 errcode、同一 `task_pos=0`；`enable_moonep` 只影响
MoE 通信，跟 attention metadata 无关。A2/A3 的同族算子里这段校验是**注释掉**的
（`csrc/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/sparse_attn_sharedkv_metadata_aicpu.cpp:99-105`，
只在 `layout_kv == "BSND"` 时查 64/32），A5 kernel 自己的调度数学对 `groupSize_` 也是通用的
（`mBaseSize_ = 64U / groupSize_ * groupSize_`，32 时得 64，不会退化成 0），所以挡人的只是校验——
但要放开得两个算子一起放开，且需要 op owner 确认 32 头在 A5 上算得对。

**无模型复现（不需要 vLLM 引擎，MB 级显存）**

```bash
export ASCEND_RT_VISIBLE_DEVICES=<空闲卡>
python scripts/probe_a5_meta_heads.py          # 直接调该算子，扫 num_heads_q=32/64/128
#   num_heads_q=32  → 立刻复现同一 errcode 22007 / 同一 soName
#   HEADS=32 SEQ=300 等换长度也一样 ⇒ 与输入无关，卡在参数校验
#   num_heads_q=64 / 128 → 算子通过（输出 1024 维，非零 40 / 108 个）
```

即 AICPU runtime、该 `.so`、本机 CANN 都是好的（早前把它判成「整个 AICPU 二进制在本机不能用」是误判：
那时的 `VllmQuantLightningIndexerMetadata` 失败是复现脚本自己传错了 `layout_key`，
device 日志里写得很清楚：`For layout_key, layout must be PA_BSND/TND/BSND!`）。

**规避**

- MoE / MoonEP 侧的 A/B 验证改用不走 DSA 的模型：`Qwen3-30B-A3B`（57 GB，TP=2，EP=2）。
  第 4 条遗留项的那个 A/B（`scripts/moe_ab_test.py`，`MOONEP=0/1` 比 logprobs）已在 6/7 卡上做完。
- DeepSeek-V4-Flash 在 A5 上要跑通：**TP 路径**得等上游把稀疏注意力放开到 32 头（或明确 A5 只支持
  每 rank 64/128 → 该模型在 A5 上要换配置/量化方案）；**DP 路径**已经跑通，见下面的 DP2 + EP 实测。

**DP2 + EP 这条路的实测（2026-09-18，6/7 卡）**

「开 DP2 是不是就没这个问题了」——头数这条：**是**；但这台机器上 DP2+EP2 会撞到另一个更底层的坑，
所以端到端还是跑不起来。

好消息（头数约束解除 + 权重放得下）：

- DP=2/TP=1 时 `n_local_heads = 64 // 1 = 64`，A5 的 64/128 校验天然满足（探针
  `HEADS=64` PASS；实例里也走过了第一层 attention，挂在后面的 MoE 跨卡 collective 上，不再是
  `task_pos=0` 的 metadata 算子）。
- 权重分布也合适：`Loading model weights took 78.1171 GB`（DP0/EP0 与 DP1/EP1 相同），
  比 TP=1 单卡要的 149 GB 少得多，98 GB HBM 放得下（gmu=0.9 时 profile 都过了）。

坏消息（跨卡 MoE collective 挂死，卡在 driver）——**根因已于 09-19 定位并修掉，见本节末「根因」段；
下面这段现场保留作证据**：

```
host: RuntimeError: ... current working operator name is HcclReduceScatter
      Execution_Error_Stream_Synchronize_Timeout(EE1002): Stream synchronize timeout.
      current task is type_name=KERNEL_AICPU, device_id=6/7, stream_id=26   （两张卡同一时刻）
      Call driver api halSqCqAllocate failed, drvRetCode=17, drvDevId=6/7
device-6 / device-7 日志（同上路径）：
  [ERROR] DRV(aicpu_scheduler) [drv_share_log.c:231][tsdrv] Alloc sq mem failed. Failed to sq mem init.
  [ERROR] DRV(aicpu_scheduler) [trs_sqcq.c:1038][tsdrv] [trs_local_sqcq_alloc] Ioctl ret. (ret=17)
  [ERROR] DRV(aicpu_scheduler) [trs_slave_urma.c:750][tsdrv] [trs_remote_urma_alloc_sqcq] Alloc sqcq ret. (ret=17)
  [ERROR] DRV(aicpu_scheduler) [trs_slave_event.c:273][tsdrv] [trs_remote_alloc_sq_cq_with_urma] Alloc sqcq with urma ret. (ret=17)
```

即：EP 组跨的是**两个独立 engine 进程**（DP 模式），这条 collective 走了 AICPU 的 remote URMA
队列，而本机 URMA 起不来（与 MoonEP 必须 `MOONEP_DISABLE_UDMA=1`、shmem/UDMA init 返回 -3 是同一类
问题），remote SQ/CQ 分配失败 → AICPU 任务永远不完成 → 1800s 级超时（示例里 `Stream synchronize
timeout` 报的是 65000ms 的内部等待）。挂死时两个 engine 的 worker 都卡在 91.7 GB 显存上不退出，
需要 `kill -9 <pid>` 收尸。

多进程 DP 的启动姿势（这份仓库里没写、踩出来的）：

- 单进程 `LLM(data_parallel_size=2)` 直接被拒（`vllm/entrypoints/llm.py:298`，提示"may hang"）。
  正确做法是 `examples/features/data_parallel/data_parallel_offline.py` 的姿态：一个 DP rank 一个
  进程，只设 `VLLM_DP_RANK / VLLM_DP_RANK_LOCAL / VLLM_DP_SIZE / VLLM_DP_MASTER_IP /
  VLLM_DP_MASTER_PORT`，**`data_parallel_size` 不再传给 `LLM()`**（DP 由 `ParallelConfig` 的
  env 分支读入，`config/parallel.py:826-830`）。
- `ASCEND_RT_VISIBLE_DEVICES` 要给**整组**卡（`"6,7"`）：引擎会按 `local_dp_rank` 自己切片
  （`vllm/v1/engine/utils.py:get_device_indices` → `[rank*world_size, ...)`），给单卡时 rank≥1 会
  `IndexError: list index out of range`。
- **cwd 不能是「集成根」**（不是「必须 /tmp」）：本机 `PYTHONPATH` 结尾有一个空段（CANN 的
  set_env.sh 在 `PYTHONPATH` 空时拼出的 `...opp/built-in/op_impl/ai_core/tbe:`），空段按 **cwd**
  解析（实测 `sys.path` 里那一段会跟着 cwd 变）。而集成根
  `/mnt/share/x00893896/vllm_ascend_moonep_integration` 下面躺着一份 vLLM 的 **checkout 根** `vllm/`
  （真包在 `vllm/vllm/__init__.py`），于是标准库 PathFinder 把 `<cwd>/vllm` 当成**空 namespace
  package** 直接返回（`origin=None`，就是报错里的 "(unknown location)"）；vllm 是 editable 的
  "empty" 安装、其 finder 用 `sys.meta_path.append(...)`（排在 PathFinder **之后**），兜不住 →
  `ImportError: cannot import name 'LLM' from 'vllm' (unknown location)`。
  **月面仓根目录、家目录、/tmp 都实测正常**；早期踩坑是因为 spawn 的子进程继承父进程 cwd，现在
  `scripts/moe_ab_test.py` 的 DP 模式已给每个子进程固定 `cwd=tempfile.gettempdir()`，所以 DP 模式
  在哪儿启动都行；单进程 TP 模式仍是同进程导入，别在集成根下跑即可。
  想彻底免疫尾空段：
  `export PYTHONPATH="$REPO/third_party/ascend_moonep:$REPO${PYTHONPATH:+:${PYTHONPATH%:}}"`。

**根因（2026-09-19 定位并修复，6/7 卡）：不是 URMA 坏，是 `expandable_segments`**

挂死与 URMA 无关，触发条件是 torch_npu 的 `expandable_segments` 分配器（vllm-ascend 的默认值）：

| 变量 | 值 | 结果 |
|---|---|---|
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True`（09-18 那次） | `HcclReduceScatter` → KERNEL_AICPU → `halSqCqAllocate drvRetCode=17` 挂死 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:False` | **端到端跑通**（`[dp] EXIT=OK`，两 rank SUCCESS） |

两次运行**其余全同**：同一对卡（6,7）、同一 DP2xTP1/EP=1（`ep_rank=0/2`、128 本地专家）、
同 `gmu=0.9`、MOONEP=0、同一份权重。即单变量 A/B。与 moonep 分支 commit `95ccdc7e` 的限制 #1②
完全一致（他们的最小复现：spawn chain + 28 GB 分配 + `expandable_segments:True` + URMA on →
首次跨 rank 同步死锁；原文 "Engine URMA users must set `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False`"）。

注意**本机纯 HCCL 复现不出来**：`/tmp/hccl_probe.py`（all_reduce/all_gather/reduce_scatter，含 30 GB
分配后再做一轮）在 `True` 下也全 PASS。所以这个坑是 **HCCL 走 AICPU remote URMA 队列 + 
expandable_segments 的大段/可搬迁虚拟地址** 叠加才触发的；DP2 时 EP 组跨两个 engine 进程，
collective 走的就是这条 AICPU+URMA 路。

对照组（回答「是不是本机 DP+EP 本来就挂」）：

- 纯 `torch.distributed` HCCL probe 六种配置全 PASS：6,7 纯通信；6,7 + 30 GB 分配（默认 /
  `expandable_segments:True` / `False`）；4,5；4,5,6,7。
- `Qwen3-30B-A3B` 走同一套多进程 DP harness（DP2xTP1），EP=1 与 EP=0 **都跑通**，
  两 rank top-5 logprobs 完全一致（Qwen3 显存小，`True` 下也没触发上面的叠加）。
  ⇒ 本机 DP+EP 跨 engine 这条路是通的。

**MoonEP 在 DSV4-Flash 上跑通的配置（2026-09-19，6/7 卡）**

```
ASCEND_RT_VISIBLE_DEVICES=6,7            # 整组；每个 DP 进程都给整组
VLLM_DP_RANK / VLLM_DP_RANK_LOCAL / VLLM_DP_SIZE / VLLM_DP_MASTER_IP / VLLM_DP_MASTER_PORT
CARDS=6,7 DP=2 TP=1 EP=1                 # LLM(..., tensor_parallel_size=1,
                                         #      enable_expert_parallel=True, gpu_memory_utilization=0.9)
additional_config={"enable_moonep": True}
MOONEP_DISABLE_UDMA=1                    # 本机必须；见下
PYTORCH_NPU_ALLOC_CONF=expandable_segments:False   # 必须
HCCL_BUFFSIZE=1024
```

证据：

- `[moonep-init r0/r1] stage=shmem-init-done`（`bootstrap=ipport`，即 `95ccdc7e` 的默认值）；
- MoE 打点（`VLLM_ASCEND_MOE_PAR_STATS=1`）两份 dump 各 688 行 = 43 层 × 16 步，
  **688/688 全是 `method=MoonEPCommImpl`**（没有悄悄退回 MC2/AllGather）；
- 重复跑一遍，两 rank 的 top-5 logprob 与第一次**逐位相同**（确定性，不是竞态）；
- 两 rank 输出可读文本，进程正常退出（对比 09-18 那次挂死要 `kill -9`、占 91.7 GB 显存）。

UDMA 仍开不起来（另一件事，与上面无关）：去掉 `MOONEP_DISABLE_UDMA` 后快速失败，
`RuntimeError: aclshmemx_init_attr_with_buffers failed with -3 (my_pe=1, n_pes=2, buffers=24)`
（`moonep_comm.py:393` → `shmem_runtime.py:265` → `buffer_c.py:159`），与 §2 的 shmem/设备号问题是同一族。
⇒ 本机 MoonEP 只能走 UDMA-off 路径。

另外两条也都实测通过（同一份 env，只换配置/启动器）：

- **DP4×TP1 EP=4（卡 4,5,6,7）**：`[dp] EXIT=OK`，四个 rank 全 SUCCESS，输出文本一致；
  即官方 950DT 配方里的 DP4 形状（`--data-parallel-size 4`、不带 TP）在本机 + MoonEP 下成立。
- **官方 `vllm serve` 路径**（不再是我那套 SPMD harness）：

  ```bash
  ASCEND_RT_VISIBLE_DEVICES=6,7 vllm serve <DSV4-Flash> \
    --data-parallel-size 2 --enable-expert-parallel \
    --max-model-len 4096 --max-num-batched-tokens 512 --gpu-memory-utilization 0.90 \
    --block-size 32 --enforce-eager --additional-config '{"enable_moonep": true}'
  ```

  `shmem-init-done` → 两个 `ApiServer` 都 `Application startup complete` → `POST /v1/completions`
  正常返回（两条 prompt 都是 16 token、finish_reason=length）。
  所以月面分支限制 #7 里那个「DP 要全量可见设备 + DP-offset shim」在**当前 vLLM 0.23.0 +
  本仓**下不再是拦路虎：只要 env 给整组卡，engine 自己按 local_dp_rank 切片就对了。
  （注意 `--enable-shared-expert-dp` 在 vLLM 0.23.0 CLI 上**不存在**，传了直接
  `unrecognized arguments` 退出码 2；官方文档里它是 `additional_config` 的键。）

**待评估**：baseline（MOONEP=0）两 rank 的 top-5 logprob 完全一致；MOONEP=1 时两 rank 有
0.001~0.3 的 logprob 差（prompt 2 的贪心输出文本也不同）。每次运行内部是确定的，所以不是竞态，
怀疑是 MoonEP 对称内存 padding/累加次序带来的数值差。交付前建议用仓库自带的 gsm8k 测评
量化 MOONEP=0/1 的精度差。

跑这个 A/B 的入口已经内置进 `scripts/moe_ab_test.py`（DP 模式下它自己是自己的 launcher，
一卡一进程，父进程只给子进程注入 `VLLM_DP_*` + `ASCEND_RT_VISIBLE_DEVICES`）：

```bash
cd /tmp    # 其实只需避开「集成根」（那下面有 vllm/ checkout）；DP 模式 launcher 已给子进程固定 cwd
DP=2 CARDS=6,7 MOONEP=0 MODEL=/mnt/weight/A5-weights/DeepSeek-V4-Flash python <repo>/scripts/moe_ab_test.py
DP=2 CARDS=6,7 MOONEP=1 MODEL=/mnt/weight/A5-weights/DeepSeek-V4-Flash python <repo>/scripts/moe_ab_test.py
```

产物 `/tmp/ab_{baseline,moonep}_<tag>_dp{0,1}.json` 与合并件 `..._dp_all.json`（含跨 rank 的
top-1 差值一行）；`AB_STATS=1` 顺带落 MoE 通信方式 dump，`AB_TIMEOUT`（默认 1800s）到点按
进程组收尸。默认每个 rank 跑全量 prompts（便于和单进程基线逐 rank 对拍），`AB_SLICE=1`
才切成官方那种「rank r 处理 PROMPTS[r::DP]」。
DSV4 走不了 `scripts/moe_par_bench.py` 的 TP 路径（会被 64/128 头约束挡住），要用这套 DP 入口灌数据集。

**遗留/待 op owner 确认**

1. `num_heads_q` 该传全局还是每 rank：metadata 的 QLI 兄弟算子
   （`npu_vllm_quant_lightning_indexer_metadata`，`vllm_ascend/attention/dsa_v1.py:830`）传的是全局
   `index_n_heads=64`，本算子传 `n_local_heads`（同文件 751 行）；但消费者算子 host 侧校验用的是
   q tensor 的实际头数（每 rank 32），所以即便 metadata 改成传全局，也过不了消费者那关。
2. 需要 device 侧日志才能定位的这类 AICPU 报错，别只看 host plog——`22007 inner error` 只表示
   kernel `return false`，原因在 `/root/ascend/log/debug/device-*/`（按次生成，不滚动，跑完立刻看）。

---

## 附录 A：重打 catlass 补丁的脚本

`block_epilogue_per_group_per_block.hpp` 那 3 个类外定义用下面这个一次性脚本搬进类内
（行号按该 pin 的**原文件**写死，`git submodule update` 之后原样可用）。
另外 8 个函数是手工补 `static`。核对方式（在子模块根目录执行，**应当没有输出**；
注意别用 `grep '__simd_vf__ static'` 当判据——多数签名中间夹着 `inline`，会漏掉）：

```bash
grep -rn '__simd_vf__' include/catlass/ | grep -v static
```

```python
# inline_vf_defs.py —— 在 catlass 子模块根目录下执行
path = 'include/catlass/epilogue/block/block_epilogue_per_group_per_block.hpp'
lines = open(path).readlines()  # 0-indexed

# 类内声明(0-idx 区间): A=[167,170), B=[171,174), C=[175,177)
# 类外块(0-idx): (首579, {584, }657), (首659, {664, }716), (首718, {722, }749)
pairs = [
    ((167, 170), (579, 584, 657)),
    ((171, 174), (659, 664, 716)),
    ((175, 177), (718, 722, 749)),
]

for (da, db), (bs, bb, be) in pairs:
    decl = ''.join(lines[da:db])
    sig = ''.join(lines[bs:bb])
    assert ('AivPerTensor' in decl) == ('AivPerTensor' in sig), decl
    assert ('AddBias' in decl) == ('AddBias' in sig), decl
    if 'CalcType* dst' in decl:
        assert 'CalcType* dst' in sig
    else:
        assert 'CalcType* dst' not in sig
    assert lines[bb].rstrip() == '{' and lines[be].rstrip() == '}'
    assert lines[bs].startswith('QBMM_BLOCK_EPILOGUE')
print("签名/边界校验通过")

ops = []
for (da, db), (bs, bb, be) in pairs:
    decl = list(lines[da:db])
    decl[-1] = decl[-1].rstrip()
    assert decl[-1].endswith(');')
    decl[-1] = decl[-1][:-1] + '\n'
    decl = [ln.replace('__simd_vf__ void', '__simd_vf__ static void', 1)
            if '__simd_vf__ void' in ln else ln for ln in decl]
    body = ['    {\n']
    for ln in lines[bb + 1:be]:
        body.append(('    ' + ln) if ln.strip() else ln)
    body.append('    }\n')
    ops.append((da, db, decl + body))
    ops.append((bs, be + 1, None))  # 删除整个类外块(含 QBMM 行、template 行、签名、函数体)

ops.sort(key=lambda t: t[0])
out, cur = [], 0
for a, b, new in ops:
    out.extend(lines[cur:a])
    if new is not None:
        out.extend(new)
    cur = b
out.extend(lines[cur:])

open(path, 'w').writelines(out)
print("写入完成, 原", len(lines), "行 -> 新", len(out), "行")
```

（这个脚本此前只存在于 `/tmp/inline_vf_defs.py`，而 `/tmp` 不保证长期保留；如需常备，
可考虑收进本仓，例如 `docs/` 或 `csrc/` 下的脚本目录。）

## 附录 B：启动前的三条自查

```bash
# 1) 用的哪一份 ascend_moonep Python 层？应当是本仓的 third_party/ascend_moonep
python -c "import ascend_moonep; print(ascend_moonep.__file__)"

# 2) shmem 的两个部署点是否都是重编后的库（大小 4109400 B、时间对得上）
ls -l /home/x00893896/vllm_ascend_moonep_integration/ascend-moonep/3rdparty/shmem/install/shmem/lib/libshmem.so \
      /home/x00893896/vllm_ascend_moonep_integration/ascend-moonep/ascend_moonep/lib/libshmem.so

# 3) catlass 的 4 个头文件还在 dirty 状态（补丁没被 submodule update 冲掉）
git -C /home/x00893896/vllm_ascend_moonep_integration/vllm_ascend_moonep \
    submodule status; git -C csrc/third_party/catlass status --short
```

## 相关文档

- [moe_par.md](moe_par.md)：MoE 负载采集与峰均比报告的使用方法（打点、驱动、报告）。
- 仓库根 `README.md`：环境搭建、MoonEP 的开关与性能优化点。
