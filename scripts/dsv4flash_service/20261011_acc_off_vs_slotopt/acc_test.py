#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""精度对比测试：一组固定提示词，贪心解码（temperature=0），把每条输出存 JSON。

用法: python3 acc_test.py <leg名>   # 产物 /tmp/acc_test/<leg名>.json
对比: 由 compare 阶段完成（off vs slot 逐条比对）。
提示词混合：可验证的算术/事实题 + 生成题，覆盖中英文。
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8444/v1/completions"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 绕过公司代理

# (提示词, 期望包含的答案片段或 None)——期望片段用于两腿各自的对错抽查
PROMPTS = [
    ("计算：17 × 23 = ", "391"),
    ("计算：144 ÷ 12 = ", "12"),
    ("1 + 2 + 3 + ... + 100 = ", "5050"),
    ("The capital of France is", "Paris"),
    ("中国的首都是", "北京"),
    ("水的化学式是", "H2O"),
    ("下列哪个是偶数：7, 12, 33, 45？只回答数字。", "12"),
    ("把下面这串数字倒序输出：3 1 4 1 5 9 2 6\n倒序是：", None),
    ("Write a Python function that adds two numbers:", None),
    ("简要说明什么是快速排序算法：", None),
    ("翻译成英文：今天天气很好\n译文：", None),
    ("从 1 数到 10，用逗号分隔：", None),
]
MAX_TOKENS = 64


def ask(prompt: str) -> dict:
    body = {
        "model": "test_auto",
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 1024,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(opener.open(req, timeout=600))
    return {
        "prompt": prompt,
        "text": r["choices"][0]["text"],
        "finish": r["choices"][0].get("finish_reason"),
        "completion_tokens": r["usage"]["completion_tokens"],
        "latency_s": round(time.time() - t0, 2),
    }


def main():
    leg = sys.argv[1]
    out = {"leg": leg, "ts": time.strftime("%F %T"), "results": []}
    for p, _ in PROMPTS:
        try:
            out["results"].append(ask(p))
            print(f"[{leg}] OK {p[:24]!r} -> {out['results'][-1]['text'][:40]!r}")
        except Exception as e:  # noqa: BLE001
            out["results"].append({"prompt": p, "error": str(e)})
            print(f"[{leg}] FAIL {p[:24]!r}: {e}")
    with open(f"/tmp/acc_test/{leg}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[{leg}] 完成：/tmp/acc_test/{leg}.json")


if __name__ == "__main__":
    main()
