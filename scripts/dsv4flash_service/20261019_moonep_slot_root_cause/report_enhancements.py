"""Explanatory SVG diagrams and patch instructions for the report."""
from pathlib import Path
from html import escape as E
ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/"report"/"assets"
INK="#17334f"; MUTED="#536b81"; LINE="#dce5ef"
GREEN="#dcefe7"; GREEN_INK="#086454"; BLUE="#deebfa"; BLUE_INK="#245c9b"
GRAY="#edf0f4"; GRAY_INK="#627185"; RED="#a6333c"; LIGHT_RED="#fff1ed"
class SVG:
 def __init__(self,key,title,desc,height):
  self.key=key; self.h=height
  self.p=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" role="img" aria-labelledby="{key}-title {key}-desc">',
   f'<title id="{key}-title">{E(title)}</title><desc id="{key}-desc">{E(desc)}</desc>',
   '<defs><marker id="'+key+'-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8Z" fill="#71859a"/></marker></defs>',
   '<rect width="1000" height="'+str(height)+'" fill="#ffffff"/>',
   '<g font-family="Microsoft YaHei,Noto Sans CJK SC,Arial,sans-serif">']
 def rect(self,x,y,w,h,fill,stroke=None,r=8):
  self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"'+(f' stroke="{stroke}"' if stroke else '')+'/>')
 def text(self,x,y,s,size=18,color=INK,bold=False,anchor="start"):
  self.p.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{700 if bold else 400}">{E(s)}</text>')
 def lines(self,x,y,lines,size=18,color=INK,bold=False,anchor="start",gap=27):
  for i,s in enumerate(lines):self.text(x,y+i*gap,s,size,color,bold,anchor)
 def arrow(self,x1,y1,x2,y2):
  self.p.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="#71859a" stroke-width="2" marker-end="url(#{self.key}-arrow)"/>')
 def cell(self,x,y,w,h,s):
  fill,color=(GRAY,GRAY_INK) if s=="D" else ((BLUE,BLUE_INK) if s=="P2" else (GREEN,GREEN_INK))
  self.rect(x,y,w,h,fill,r=5); self.text(x+w/2,y+h/2+7,s,20,color,True,"middle")
 def finish(self): return "".join(self.p+["</g></svg>"])
def figure(key,title,desc,svg):
 ASSETS.mkdir(parents=True,exist_ok=True)
 (ASSETS/(key+".svg")).write_text(svg,encoding="utf-8")
 return f'<figure class="diagram" id="fig-{key}"><figcaption><strong>{E(title)}</strong><a href="assets/{key}.svg" target="_blank" rel="noopener">单独查看大图 ↗</a></figcaption><div class="diagram-scroll">{svg}</div><p class="caption">{E(desc)} 窄屏可横向滚动。</p></figure>'
def chunk_phase_diagram():
 desc="示意固定两轮初始偏移的影响。横向为 forward 顺序，不按真实耗时等比例绘制；后续 P / P1 / P2 属于连续的新请求。"
 s=SVG("chunk-phase","相同的两轮初始偏移，切 chunk 改变了它与周期的关系",desc,670)
 s.text(26,36,"同样晚两轮开始：不切分能重新同相，切两块却持续错位",25,bold=True)
 s.text(26,66,"P = 完整 16K prefill；P1 / P2 = 各 8K chunk；D = 本地无真实 token 的 dummy",17,MUTED)
 for y,title in [(100,"A  16K 输入 / 16K 预算：P → D，周期为 2"),(363,"B  16K 输入 / 8K 预算：P1 → P2 → D，周期为 3")]:
  s.rect(20,y,960,244,"#f8fafc",LINE)
  s.text(38,y+31,title,21,bold=True)
  for i in range(8):s.text(223+i*91,y+63,str(i+1),16,MUTED,anchor="middle")
  s.text(40,y+63,"全局轮次 →",16,MUTED)
  s.text(40,y+107,"先到 rank",18,bold=True); s.text(40,y+162,"其余 3 rank",18,bold=True)
  r0=["P","D"]*4 if y==100 else ["P1","P2","D","P1","P2","D","P1","P2"]
  r1=["D","D","P","D","P","D","P","D"] if y==100 else ["D","D","P1","P2","D","P1","P2","D"]
  for i,q in enumerate(r0):s.cell(182+i*91,y+77,82,42,q)
  for i,q in enumerate(r1):s.cell(182+i*91,y+132,82,42,q)
  s.p.append(f'<path d="M355 {y+71} V{y+183}" stroke="#aa6770" stroke-width="2" stroke-dasharray="5 5"/>')
  s.text(362,y+204,"↑ 其余 rank 第 3 轮才开始真实请求",16,RED)
  s.text(40,y+230,"2 mod 2 = 0 → 从第 3 轮起同相" if y==100 else "2 mod 3 = 2 → 稳态反复出现真实 / dummy 混合",18,GREEN_INK if y==100 else RED,True)
 s.text(26,643,"每轮通信顺序一致，不代表每个 rank 的请求阶段一致。",20,bold=True)
 return figure("chunk-phase","图 1｜切分为什么把启动偏移变成固定错位",desc,s.finish())
def kv_gap_diagram():
 desc="本次 9GiB KV、异步 batch queue、输出 1 token 配置的调度链。D 并不是额外的正常 decode。"
 s=SVG("kv-gap","先调度、后回收输出，使 KV 已释放的这一轮仍然执行 dummy",desc,390)
 s.text(26,36,"这一个 D，从哪里来？",25,bold=True);s.text(26,65,"请求 A 的最后一块在途；请求 B 已在等待队列。",18,MUTED)
 boxes=[(24,["① 先 schedule()","A 仍占 KV","B 分配空间失败"],BLUE),(220,["② 得到空调度","本轮真实 token = 0","executed = False"],LIGHT_RED),(416,["③ 再消费 A 输出","A 完成","KV 此刻释放"],GREEN),(612,["④ 本轮不重新调度","executed 仍为 False","于是执行 D"],LIGHT_RED),(808,["⑤ 下次循环","B 才分到 KV","开始 B 的 P1"],BLUE)]
 for x,lines,fill in boxes:
  s.rect(x,106,168,145,fill);s.lines(x+84,139,lines,16,INK,False,"middle",gap=40)
  if x<808:s.arrow(x+170,178,x+191,178)
 s.rect(24,278,952,83,"#f1f5fa")
 s.text(500,311,"因此：不切分为 P → D；切两块为 P1 → P2 → D。",21,INK,True,"middle")
 s.text(500,343,"20ms 补丁保留这个 D，让不同 rank 的 D 出现在同一轮。",18,GREEN_INK,False,"middle")
 return figure("kv-gap","图 2｜KV 回收顺序如何维持每请求后的空轮",desc,s.finish())
def fix_diagram():
 desc="本次突发负载的修复前后对照。重=至少一个 rank 有真实 8K 工作，轻=四个 rank 均 dummy；各重轮耗时并不相等。左侧分词时间来自一个默认窗口实测样本；下方为稳态机制示意，格宽不代表耗时。"
 s=SVG("fix-alignment","20ms 分词聚合窗口改变初始准入相位，使真实工作与 dummy 各自集中",desc,905)
 s.text(26,36,"20ms 补丁修复的是“何时一起开始”，并没有删除 dummy",24,bold=True)
 for x,before in [(24,True),(520,False)]:
  s.rect(x,62,456,758,"#fff8f5" if before else "#f3faf6",LINE)
  s.text(x+22,97,"修复前：默认 2ms 窗口" if before else "修复后：20ms 聚合窗口",23,RED if before else GREEN_INK,True)
  s.text(x+22,128,"首批 16 个文本请求，到达跨度约 11ms",16,MUTED)
  for i in range(16):s.rect(x+22+i*25,147,21,25,BLUE if (not before or i==0) else GREEN,r=3)
  s.lines(x+22,202,["分词拆成 1 + 15 两批","约 40.2ms 释放 1 个；477.5ms 释放 15 个"] if before else ["首批 16 个请求合为一批完成分词","聚合窗口 20ms ≠ 分词总耗时 20ms"],16,INK,False,gap=27)
  s.arrow(x+228,245,x+228,272);s.rect(x+20,283,416,80,"#ffffff",LINE)
  s.lines(x+228,313,["rank0 先获得真实请求","其它 rank 迟到前已提交两轮 D"] if before else ["请求集中就绪，随后快速分发","四个 rank 都赶上首轮真实调度"],18,INK,True,"middle",gap=29)
  s.arrow(x+228,374,x+228,398)
  s.text(x+22,426,"首次真实轮偏移：[0, 2, 2, 2]" if before else "首次真实轮偏移：[0, 0, 0, 0]",20,RED if before else GREEN_INK,True)
  s.text(x+22,457,"稳态三轮 →",16,MUTED)
  for j in range(3):s.text(x+175+j*90,457,str(j+1),16,MUTED,anchor="middle")
  for r in range(4):
   yy=476+r*44;s.text(x+22,yy+26,"rank "+str(r),17)
   seq=["P1","P2","D"] if not before or r==0 else ["P2","D","P1"]
   for j,q in enumerate(seq):s.cell(x+135+j*90,yy,80,35,q)
  for j,q in enumerate(["重","重","重"] if before else ["重","重","轻"]):
   xx=x+135+j*90;s.rect(xx,669,80,40,"#f5dcd5" if before else (GREEN if j<2 else GRAY),r=5)
   s.text(xx+40,697,q,21,RED if before else GREEN_INK,True,"middle")
  s.lines(x+22,744,["每 3 轮都有真实 8K 工作","完整样本：25 轮均含真实负载"] if before else ["每 3 轮只有 2 轮含真实工作","完整样本：16 重轮 + 7 全 dummy"],18,INK,True,gap=31)
 s.rect(24,838,952,47,BLUE);s.text(500,869,"真实 token 总量同为 524,288；补丁让 D 集中，保住跨 rank 并行度。",20,INK,True,"middle")
 return figure("fix-alignment","图 3｜修复前后：集中就绪 → 首轮对齐 → dummy 集中",desc,s.finish())
def code(s):return "<pre><code>"+E(s.strip())+"</code></pre>"
def patch_usage_html():
 init=r'''
# 在目标容器内执行；以下路径对应本报告的实际保存位置。
WORKSPACE=/home/x00893896/vllm_ascend_moonep_integration
VLLM_REPO="$WORKSPACE/vllm"
PATCH_FILE="$WORKSPACE/vllm-ascend/scripts/dsv4flash_service/20261019_moonep_slot_root_cause/patch/tokenizer_window.patch"

git -C "$VLLM_REPO" rev-parse HEAD
git -C "$VLLM_REPO" status --short
git -C "$VLLM_REPO" apply --stat "$PATCH_FILE"

# 只检查，不改文件：
if git -C "$VLLM_REPO" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "补丁已存在：跳过应用，直接进行环境配置。"
elif git -C "$VLLM_REPO" apply --check "$PATCH_FILE"; then
    echo "补丁尚未应用且检查通过：可以执行下一步应用命令。"
else
    echo "版本或工作区内容不匹配：先人工处理差异，不要强制应用。"
fi
'''
 apply=r'''
# 仅在上一段显示“尚未应用且检查通过”时执行：
git -C "$VLLM_REPO" apply --check "$PATCH_FILE" &&
git -C "$VLLM_REPO" apply "$PATCH_FILE"

# 应用后核对；反向检查成功说明整份补丁的修改已在工作区。
git -C "$VLLM_REPO" apply --reverse --check "$PATCH_FILE"
git -C "$VLLM_REPO" diff --check
git -C "$VLLM_REPO" diff --stat
git -C "$VLLM_REPO" status --short -- \
  vllm/envs.py vllm/utils/async_utils.py tests/utils_/test_async_tokenizer_window.py
'''
 enable=r'''
# 本次实验实际使用的 Python；其它部署改成启动 API 服务的那个解释器。
SERVICE_PYTHON="$WORKSPACE/scripts/20261017_scheduler_alignment/.venv/bin/python"
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH="$VLLM_REPO:$WORKSPACE/vllm-ascend:${PYTHONPATH:-}"
export VLLM_TOKENIZER_BATCH_WAIT_MS=20

# 从中性目录检查实际导入位置，避免同名目录影响解析。
cd /tmp
"$SERVICE_PYTHON" - <<'PY'
import inspect
from vllm import envs
from vllm.utils import async_utils

print("loaded:", async_utils.__file__)
print("window_ms:", envs.VLLM_TOKENIZER_BATCH_WAIT_MS)
assert envs.VLLM_TOKENIZER_BATCH_WAIT_MS == 20.0
source = inspect.getsource(async_utils.AsyncMicrobatchTokenizer.__init__)
assert "VLLM_TOKENIZER_BATCH_WAIT_MS" in source
print("patched tokenizer configuration: OK")
PY

# 然后使用原服务启动命令重启 API，让新进程继承以上环境。
'''
 tests=r'''
# 使用上一步已设置的 SERVICE_PYTHON、PYTHONPATH、CANN 环境：
cd "$WORKSPACE/scripts/20261017_scheduler_alignment"
ALIGNMENT_DISABLE_EXTRA=1 "$SERVICE_PYTHON" -m pytest \
  --confcutdir="$VLLM_REPO/tests/utils_" \
  "$VLLM_REPO/tests/utils_/test_async_tokenizer_window.py" \
  "$VLLM_REPO/tests/utils_/test_async_utils.py" -q
'''
 rollback=r'''
# 优先做配置回退：恢复默认行为，保留补丁的配置能力。
export VLLM_TOKENIZER_BATCH_WAIT_MS=2
# 或：unset VLLM_TOKENIZER_BATCH_WAIT_MS
# 再使用相同服务启动流程重启 API。

# 如确实要移除源代码补丁，先确认没有后续重叠修改，再执行：
git -C "$VLLM_REPO" apply --reverse --check "$PATCH_FILE" &&
git -C "$VLLM_REPO" apply --reverse "$PATCH_FILE"
'''
 return '''
<div class="patch-guide" id="patch-usage">
<h3>补丁怎么用：检查 → 应用 → 启用 → 验证 → 回退</h3>
<p>补丁和使用说明集中在报告根目录的直接子目录 <code>patch/</code>：<a href="../patch/tokenizer_window.patch">tokenizer_window.patch</a> · <a href="../patch/README.html">独立使用说明（HTML）</a> · <a href="../patch/README.txt">使用说明（纯文本）</a>。</p>
<div class="callout good"><p><strong>当前容器已经有补丁。</strong>2026-10-19 只读复查确认：vLLM HEAD 为 <code>0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665</code>，两处源码修改和新增测试均存在，<code>git apply --reverse --check</code> 通过。此环境直接从第 3 步配置和重启；下面完整流程供其它同版本环境复用。</p></div>
<p>这是 <strong>vLLM 仓库</strong>补丁，应用目标为集成工程的 <code>vllm/</code>，不是保存报告的 <code>vllm-ascend/</code>。补丁以本报告记录的 commit 为基准；其它版本必须先检查，不能仅凭参数同名假定兼容。执行前保存自己在这三个目标文件上的其它改动。</p>
<h4>1. 定位仓库和补丁，检查当前状态</h4>
'''+code(init)+'''
<h4>2. 只对尚未应用的工作区打补丁</h4>
'''+code(apply)+'''
<p class="small">新增测试是未跟踪文件，可能不会出现在普通 <code>git diff --stat</code> 中，因此同时检查 <code>git status</code>。不使用 <code>git reset --hard</code> 或覆盖整个源码文件来处理冲突。</p>
<h4>3. 设置 20ms，并确认服务会导入这份源码</h4>
'''+code(enable)+'''
<p>预期 <code>loaded</code> 指向 <code>.../vllm/vllm/utils/async_utils.py</code>，<code>window_ms</code> 为 <code>20.0</code>。本次通过 PYTHONPATH 使用源码，Python 补丁无需重新编译 NPU 算子或重装 Python。若服务采用其它虚拟环境或安装包，应以其实际导入路径为准。</p>
<div class="callout"><p><strong>设置环境变量之后必须重启 API 服务。</strong>上面检查的是新 Python 进程，不能据此认定已经运行的 API 已更新；在另一个 shell 中 export 不会改变旧进程。把变量设置在真正启动 API 的脚本 / 进程环境里，保持其余 DP、KV、chunk 和 slot 参数一致。若调用方显式传入 <code>batch_wait_timeout_s</code>，该显式值仍优先。</p></div>
<h4>4. 验证补丁行为，再验证调度相位</h4>
'''+code(tests)+'''
<p>归档环境已有相关测试依赖，原验证结果为 <strong>8 passed</strong>。性能回归还需使用正文的 n32/c16、L16K/B8K 配置，检查请求 ID、每轮 token 与 dummy：总真实 token=524288，首轮偏移 [0,0,0,0]，每 rank 8 请求，16 全真实 + 7 全 dummy。单测通过本身不等于任意业务流量一定对齐。</p>
<h4>5. 回退窗口，或移除补丁</h4>
'''+code(rollback)+'''
<p>反向检查失败时，应人工处理重叠修改；不要强制覆盖其它改动。移除代码后同样需要重启 API。仅把窗口回退为 2ms 就能恢复原默认聚合行为，通常足够进行 A/B 对照。</p>
<p class="small">应用、反向检查和回退已在原始基准源码的隔离临时目录中验证通过，并确认恢复原文件；没有对当前服务源码再次打补丁。验证记录：<a href="../data/patch_usage_validation.json">patch_usage_validation.json</a>。</p>
</div>'''
diagram_css="""
.diagram{margin:24px 0 28px;padding:16px;border:1px solid #dce5ef;border-radius:9px;background:white}
.diagram figcaption{display:flex;justify-content:space-between;gap:14px;align-items:baseline;margin-bottom:14px}
.diagram figcaption strong{font-size:17px;color:#17334f}.diagram figcaption a{font-size:12px;white-space:nowrap}
.diagram-scroll{overflow-x:auto;max-width:100%;border-radius:5px}.diagram svg{display:block;width:100%;height:auto;min-width:780px}
.diagram .caption{margin:12px 0 0}.patch-guide h4{font-size:17px;margin-top:28px}.patch-guide{border-top:1px solid var(--line);margin-top:25px}
@media(max-width:640px){.diagram{padding:10px}.diagram figcaption{display:block}.diagram figcaption a{display:inline-block;margin-top:5px}.diagram svg{min-width:780px}}
@media print{.diagram{break-inside:avoid;padding:5px;border:0}.diagram-scroll{overflow:visible}.diagram svg{min-width:0;width:100%}.diagram figcaption a{display:none}.patch-guide pre{break-inside:auto}}
"""
