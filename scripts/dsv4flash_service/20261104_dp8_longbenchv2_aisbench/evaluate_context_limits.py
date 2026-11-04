#!/usr/bin/env python3
"""CPU-only KV planner arithmetic; no model loading or NPU allocation.

Load only spec definitions and pure planning functions from the checked-out
sources. Reconstruct the A5 DSV4 Flash cache entries from config and constructors.
Results are planning estimates, not a service-start or runtime memory validation.
"""
import ast
import copy
import csv
import dataclasses
import enum
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Self, cast

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[3]

def load_defs(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names), (path, set(names)-{n.name for n in nodes})
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)

class Registry:
    @staticmethod
    def get_uniform_type_base_spec(spec):
        return ns['FullAttentionSpec'] if isinstance(spec, ns['FullAttentionSpec']) else type(spec)

ns = dict(__name__=__name__, dataclass=dataclasses.dataclass, field=dataclasses.field,
          fields=dataclasses.fields, replace=dataclasses.replace, copy=copy,
          Enum=enum.Enum, IntEnum=enum.IntEnum, Counter=Counter, defaultdict=defaultdict,
          prod=math.prod, math=math, cast=cast, Self=Self,
          cdiv=lambda a,b:(a+b-1)//b, round_up=lambda a,b:(a+b-1)//b*b,
          get_dtype_size=lambda dtype:dtype,
          torch=SimpleNamespace(float32=4, bfloat16=2, float16=2, float8_e4m3fn=1,
                                int8=1, uint8=1, dtype=int),
          KVCacheSpecRegistry=Registry,
          AscendDeviceType=SimpleNamespace(A5='A5'), get_ascend_device_type=lambda:'A5',
          logger=logging.getLogger('kv-estimate'))

def main():
    interface = WS/'vllm/vllm/v1/kv_cache_interface.py'
    custom = WS/'vllm-ascend/vllm_ascend/core/kv_cache_interface.py'
    utils = WS/'vllm/vllm/v1/core/kv_cache_utils.py'
    load_defs(interface, ['KVQuantMode', 'KVCacheSpec', 'AttentionSpec', 'FullAttentionSpec',
              'MLAAttentionSpec', 'SlidingWindowSpec', 'SlidingWindowMLASpec',
              'UniformTypeKVCacheSpecs', 'KVCacheGroupSpec', '_apply_alignment_padding'], ns)
    load_defs(custom, ['_get_c8_k_cache_dtype', '_get_c8_k_scale_cache_dtype',
              'AscendMLAAttentionSpec', 'AscendSlidingWindowMLASpec'], ns)
    load_defs(utils, ['_approximate_gcd', 'group_and_unify_kv_cache_specs',
              '_get_kv_cache_groups_uniform_groups', '_max_memory_usage_bytes_from_groups',
              '_bucket_layers_by_page_size'], ns)
    config = json.loads(Path('/mnt/share/weight/DeepSeek-V4-Flash/config.json').read_text())
    ratios = config['compress_ratios'][:config['num_hidden_layers']]
    n = Counter(ratios)
    rows = list(csv.DictReader((ROOT/'length_audit.csv').open()))
    lengths = [int(r['prompt_tokens']) for r in rows]
    print('LAYERS', dict(n), flush=True)
    results=[]
    for block in [128,64,32]:
        kv={}
        mla = ns['AscendMLAAttentionSpec']; swa=ns['AscendSlidingWindowMLASpec']
        c4_block=block//16; c128_block=block//8
        for i, ratio in enumerate(ratios):
            kv[f'{i}.swa'] = swa(block_size=block,num_kv_heads=1,head_size=640,dtype=1,
                                  sliding_window=128,cache_dtype_str='float8_e4m3fn',model_version='deepseek_v4')
            if ratio > 1:
                kv[f'{i}.mla'] = mla(block_size=block,num_kv_heads=1,head_size=640,dtype=1,
                                      compress_ratio=ratio,model_version='deepseek_v4',cache_dtype_str='float8_e4m3fn')
                kv[f'{i}.state'] = swa(block_size=c4_block if ratio==4 else c128_block,
                        num_kv_heads=1,head_size=2048 if ratio==4 else 1024,dtype=4,
                        sliding_window=8 if ratio==4 else 128,
                        page_size_padded=block*640)
            if ratio==4:
                kv[f'{i}.index'] = mla(block_size=block,num_kv_heads=1,head_size=128,dtype=1,
                        compress_ratio=4,model_version='deepseek_v4',cache_dtype_str='float8_e4m3fn',
                        scale_dim=1,scale_dtype=4)
                kv[f'{i}.index_state'] = swa(block_size=c4_block,num_kv_heads=1,head_size=512,dtype=4,
                        sliding_window=8,page_size_padded=block*132)
        grouped=ns['group_and_unify_kv_cache_specs'](kv)
        groups=ns['_get_kv_cache_groups_uniform_groups'](grouped)
        print('GROUPS', block, [(len(g.layer_names),g.kv_cache_spec.get_num_layer_tuples(),g.kv_cache_spec.get_page_sizes()) for g in groups],flush=True)
        buckets=ns['_bucket_layers_by_page_size'](groups)
        bytes_per_pool_block=sum(ps*len(slots) for ps,slots in buckets.items())
        for cap in [8192,16384,32768,65536,131072,262144,524288,1048575]:
            vc=SimpleNamespace(model_config=SimpleNamespace(max_model_len=cap+1),
                 scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
                 parallel_config=SimpleNamespace(decode_context_parallel_size=1,prefill_context_parallel_size=1))
            required=ns['_max_memory_usage_bytes_from_groups'](vc,groups)
            record=dict(input_cap=cap,max_model_len=cap+1,cache_block_size=block,
                        original_samples=sum(t<=cap for t in lengths),
                        one_max_length_request_kv_gib=required/2**30,
                        two_max_length_requests_kv_gib=required*2/2**30,
                        kv_pool_bytes_per_block=bytes_per_pool_block)
            results.append(record)
            if block==128: print('ESTIMATE',json.dumps(record),flush=True)
    payload=dict(assumptions=['DP8/TP1/EP8, A5 cache layout, 43 attention layers, no MTP',
                    'hybrid KV manager enabled, max-num-batched-tokens=8192',
                    'static cache planning only, not total NPU memory or a runtime guarantee'],
                 source_sha256={str(p.relative_to(WS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [interface,custom,utils]},
                 results=results)
    (ROOT/'context_limit_estimates.json').write_text(json.dumps(payload,indent=2)+'\n')

if __name__=='__main__': main()
