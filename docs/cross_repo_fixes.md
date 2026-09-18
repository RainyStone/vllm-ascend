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
  尚未做；
- MTP / DSPark 尚未启用；
- 第 3 条不修（不改 `PYTHONPATH` 的前提下），这条链路在本机跑不起来。

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
