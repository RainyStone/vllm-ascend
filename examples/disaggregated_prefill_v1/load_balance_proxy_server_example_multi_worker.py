# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server Example
#
# This proxy server is designed to distribute requests between multiple
# "prefiller" and "decoder" backend servers for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances requests to multiple prefiller and decoder servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn vllm
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least one prefiller and one decoder backend running.
# These can be mock servers or actual vLLM servers.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 ... # Prefiller 1
#   vllm serve --host 0.0.0.0 --port 8101 ... # Prefiller 2
#   vllm serve --host 0.0.0.0 --port 8200 ... # Decoder 1
#   vllm serve --host 0.0.0.0 --port 8201 ... # Decoder 2
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each prefiller and decoder:
#
#   python load_balance_proxy_server_example.py \
#     --host 0.0.0.0 --port 9000 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201
#
# This will start the proxy on port 9000, load balancing between two prefiller
# and two decoder servers.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of prefiller
# and decoder instances.
#
# Step 5: Add or Remove Prefiller or Decoder Instances (Optional)
# ---------------------------------------------------------------
# You can add or remove prefiller or decoder instances after the proxy is started.
# For example, add 2 prefiller instances:
#
#   curl -X POST http://localhost:9000/instances/add \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "prefill",
#           "instances": ["127.0.0.1:8102", "127.0.0.1:8103"]
#         }'
#
# or remove 1 decoder instance:
#
#   curl -X POST http://localhost:9000/instances/remove \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "decode",
#           "instances": "127.0.0.1:8201"
#         }'
#
# This will return a JSON object with the adding or removing info
# and the current prefiller and decoder instances.
#
# When adding instances, if the instances are not started,
# the proxy will wait and try until the instances to be started
# or exceeding the number of attempts
#
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - The proxy will round-robin requests to balance load.
# - For production, ensure your backend servers are robust and secure.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import base64
import copy
import functools
import heapq
import ipaddress
import json
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from dynamic_bucket_load_balancer import DynamicBucketLoadBalancer, Task

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


# ------------------------------------------------------------
# 1. 常量与数据类
# ------------------------------------------------------------
MANAGER_CONFIG_ENV = "LB_PROXY_MANAGER_CONFIG"
ARGS_CONFIG_ENV = "LB_PROXY_ARGS"


@dataclass
class InstanceType:
    PREFILL: str = "prefill"
    DECODE: str = "decode"


TAINT_PRIORITY = 1e15


@dataclass
class ServerSnapshot:
    """Worker 端用于同步的服务器信息（不可变）。"""
    host: str
    port: int
    index: int  # 在调度器内部列表中的索引

    @property
    def key(self) -> str:
        return self._normalize_key(self.host, self.port)

    @staticmethod
    def _normalize_key(host: str, port: int) -> str:
        norm_host = host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return f"{norm_host}:{port}"


@dataclass
class InstanceInfo:
    """一次请求选择出的服务器信息。"""
    request_id: str
    prefiller_idx: int
    prefiller_score: float
    prefiller_key: str  # host:port 格式
    decoder_idx: int
    decoder_score: float
    decoder_key: str
    task_id: Optional[str] = None  # 动态桶产生的任务 ID


# ------------------------------------------------------------
# 2. 共享调度器（管理进程中的唯一实例）
# ------------------------------------------------------------
class SharedProxyScheduler:
    """跨 worker 共享的调度状态，所有方法均受 RLock 保护。"""

    def __init__(
        self,
        prefiller_instances: List[Tuple[str, int]],
        decoder_instances: List[Tuple[str, int]],
        enable_dynamic_bucket: bool = False,
        prefill_group_threshold: int = 32 * 1024,
        max_request_tokens: int = 128 * 1024,
        max_waiting_retries: int = 3,
        waiting_retry_interval: float = 10.0,
    ):
        self._lock = threading.RLock()

        # 服务器基本信息列表（不可变顺序）
        self.prefiller_hosts: List[Tuple[str, int]] = list(prefiller_instances)
        self.decoder_hosts: List[Tuple[str, int]] = list(decoder_instances)
        self.num_prefillers = len(self.prefiller_hosts)
        self.num_decoders = len(self.decoder_hosts)

        # 每台服务器的负载
        self.prefill_active_tokens: List[float] = [0.0] * self.num_prefillers
        self.prefill_active_kv_cache: List[float] = [0.0] * self.num_prefillers
        self.decode_active_tokens: List[float] = [0.0] * self.num_decoders

        # 全局请求计数
        self.request_num = 0

        # 分组逻辑
        self.enable_dynamic_bucket = enable_dynamic_bucket
        self.num_prefill_groups = 2 if enable_dynamic_bucket else 1

        if self.num_prefill_groups > 1 and self.num_prefillers < self.num_prefill_groups:
            raise ValueError("Number of prefiller servers must be >= number of groups")

        # 初始化 prefiller 堆：每个分组一个最小堆
        self.prefiller_heaps: List[List[Tuple[float, int]]] = []  # (priority, server_idx)
        self.server_idx_to_group: Dict[int, int] = {}
        # TODO self.prefill_group_load 可以删掉？
        self.prefill_group_load: Dict[int, float] = {}
        self._prefill_groups = self._build_groups(self.num_prefillers, self.num_prefill_groups)

        for group_id, idx_list in enumerate(self._prefill_groups):
            heap = [(0.0, idx) for idx in idx_list]
            heapq.heapify(heap)
            self.prefiller_heaps.append(heap)
            self.prefill_group_load[group_id] = 0.0
            for idx in idx_list:
                self.server_idx_to_group[idx] = group_id

        # 初始化 decoder 堆（单一堆）
        self.decoder_heap: List[Tuple[float, int]] = [
            (0.0, idx) for idx in range(self.num_decoders)
        ]
        heapq.heapify(self.decoder_heap)

        # Tainted（待移除）服务器
        self.tainted_prefillers: List[int] = []  # 存储索引
        self.tainted_decoders: List[int] = []

        # 动态桶负载均衡器
        self.bucket_load_balancer: Optional[DynamicBucketLoadBalancer] = None
        self._task_cache: Dict[str, Task] = {}  # task_id -> Task 对象
        self._task_to_group: Dict[str, int] = {}  # task_id -> group_idx

        if enable_dynamic_bucket:
            buckets = [(0, prefill_group_threshold), (prefill_group_threshold, max_request_tokens)]
            self.bucket_load_balancer = DynamicBucketLoadBalancer(
                buckets=buckets,
                affinity_strength=1.0
            )

        # 等待节点管理
        self.waiting_nodes: Dict[str, Tuple[str, Tuple[str, int], int]] = {}

        # Aborted 请求管理
        self.aborted_prefiller_requests: Dict[int, set] = {}

        # 等待重试配置
        self.max_waiting_retries = max_waiting_retries
        self.waiting_retry_interval = waiting_retry_interval

        logger.warning(
            f"Shared scheduler initialized with {self.num_prefillers} prefiller(s), "
            f"{self.num_decoders} decoder(s), groups={self.num_prefill_groups}, "
            f"dynamic_bucket={enable_dynamic_bucket}"
        )

    @staticmethod
    def _build_groups(n: int, num_groups: int) -> List[List[int]]:
        """将 n 个服务器索引均匀分成 num_groups 组。"""
        if n == 0:
            return [[] for _ in range(num_groups)]
        if n == 1:
            return [list(range(n))]
        base = n // num_groups
        rem = n % num_groups
        groups = []
        start = 0
        for i in range(num_groups):
            size = base + 1 if i < rem else base
            groups.append(list(range(start, start + size)))
            start += size
        return groups

    # ---- 基础辅助方法 ----
    def _update_prefiller_priority(self, server_idx: int) -> None:
        """重新计算并更新 prefiller 服务器在堆中的优先级（需持有锁）。"""
        group = self.server_idx_to_group[server_idx]
        priority = self.prefill_active_tokens[server_idx] + self.prefill_active_kv_cache[server_idx] * 0.3
        heap = self.prefiller_heaps[group]
        heap[:] = [(p, idx) for p, idx in heap if idx != server_idx]
        heapq.heapify(heap)
        heapq.heappush(heap, (priority, server_idx))

    def _update_decoder_priority(self, server_idx: int) -> None:
        """重新计算并更新 decoder 服务器在堆中的优先级（需持有锁）。"""
        priority = self.decode_active_tokens[server_idx]
        heap = self.decoder_heap
        heap[:] = [(p, idx) for p, idx in heap if idx != server_idx]
        heapq.heapify(heap)
        heapq.heappush(heap, (priority, server_idx))

    # ---- 对外 API ----
    def next_req_id(self) -> str:
        with self._lock:
            return str(uuid.uuid4())

    def calculate_prefill_tokens(self, request_length: int) -> float:
        return request_length / 4.0

    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        return length_score * 0.0345 + 120.0745

    def calculate_decode_scores(self, request_length: int) -> float:
        return float(request_length)

    def select_prefill_group(self, req_id: str, request_tokens: float, prefill_score: float) -> Tuple[int, Optional[str]]:
        """返回 group_idx 和可选的 task_id（如果启用动态桶）。"""
        with self._lock:
            if not self.enable_dynamic_bucket:
                return 0, None
            group_idx, task = self.bucket_load_balancer.dispatch_single_task(
                req_id, request_tokens, prefill_score
            )
            task_id = task.id
            self._task_cache[task_id] = task
            self._task_to_group[task_id] = group_idx
            return group_idx, task_id

    def select_prefiller(self, group_idx: int, token_count: float, task_id: Optional[str] = None) -> int:
        """选择 prefiller 服务器并更新负载。返回服务器索引。"""
        with self._lock:
            heap = self.prefiller_heaps[group_idx]
            if not heap:
                raise RuntimeError(f"No prefiller server available in group {group_idx}")
            _, chosen_idx = heapq.heappop(heap)
            self.prefill_active_tokens[chosen_idx] += token_count
            self.prefill_active_kv_cache[chosen_idx] += token_count
            self.prefill_group_load[group_idx] = self.prefill_group_load.get(group_idx, 0.0) + token_count
            self._update_prefiller_priority(chosen_idx)

            if task_id and self.enable_dynamic_bucket:
                task = self._task_cache.get(task_id)
                if task:
                    task.server_info = f"PREFILL-{chosen_idx}"
            return chosen_idx

    def release_prefiller(self, idx: int, token_count: float, task_id: Optional[str] = None) -> None:
        """释放 prefiller 服务器负载。"""
        with self._lock:
            if idx >= self.num_prefillers:
                return
            self.prefill_active_tokens[idx] -= token_count
            group_idx = self.server_idx_to_group.get(idx, 0)
            self.prefill_group_load[group_idx] = self.prefill_group_load.get(group_idx, 0.0) - token_count
            if task_id and self.enable_dynamic_bucket:
                task = self._task_cache.pop(task_id, None)
                if task:
                    self.bucket_load_balancer.release_task(task.id)
                self._task_to_group.pop(task_id, None)
            self._update_prefiller_priority(idx)

    def release_prefiller_kv(self, idx: int, token_count: float) -> None:
        """释放 prefiller 的 KV 缓存负载。"""
        with self._lock:
            if idx >= self.num_prefillers:
                return
            if self.prefill_active_kv_cache[idx] > 0:
                self.prefill_active_kv_cache[idx] -= token_count
            self._update_prefiller_priority(idx)

    def select_decoder(self, token_count: float) -> int:
        """选择 decoder 服务器并更新负载。返回服务器索引。"""
        with self._lock:
            if not self.decoder_heap:
                raise RuntimeError("No decoder server available")
            _, chosen_idx = heapq.heappop(self.decoder_heap)
            self.decode_active_tokens[chosen_idx] += token_count
            self._update_decoder_priority(chosen_idx)
            return chosen_idx

    def release_decoder(self, idx: int, token_count: float) -> None:
        """释放 decoder 服务器负载。"""
        with self._lock:
            if idx >= self.num_decoders:
                return
            self.decode_active_tokens[idx] -= token_count
            self._update_decoder_priority(idx)

    def get_snapshot(self) -> Dict[str, List[ServerSnapshot]]:
        """返回当前所有服务器的快照，供 worker 同步客户端。"""
        with self._lock:
            return {
                "prefill_instances": [
                    ServerSnapshot(host=h, port=p, index=i)
                    for i, (h, p) in enumerate(self.prefiller_hosts)
                    if i not in self.tainted_prefillers
                ],
                "decode_instances": [
                    ServerSnapshot(host=h, port=p, index=i)
                    for i, (h, p) in enumerate(self.decoder_hosts)
                    if i not in self.tainted_decoders
                ],
            }

    def get_config(self) -> Dict[str, Any]:
        """返回所有只读配置，供 worker 本地缓存。"""
        with self._lock:
            return {
                "enable_dynamic_bucket": self.enable_dynamic_bucket,
                "num_prefill_groups": self.num_prefill_groups,
                "num_prefillers": self.num_prefillers,
                "num_decoders": self.num_decoders,
            }

    def request_started(self) -> None:
        with self._lock:
            self.request_num += 1

    def request_finished(self) -> None:
        with self._lock:
            self.request_num = max(0, self.request_num - 1)

    def abort_prefiller_request(self, server_idx: int, request_id: str) -> None:
        with self._lock:
            if server_idx >= self.num_prefillers:
                return
            self.aborted_prefiller_requests.setdefault(server_idx, set()).add(request_id)

    def acquire_aborted_prefiller_requests(self, server_idx: int) -> List[str]:
        with self._lock:
            if server_idx >= self.num_prefillers:
                return []
            aborted = list(self.aborted_prefiller_requests.get(server_idx, set()))
            self.aborted_prefiller_requests[server_idx] = set()
            return aborted

    def healthcheck(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "prefill_instances": self.num_prefillers - len(self.tainted_prefillers),
                "decode_instances": self.num_decoders - len(self.tainted_decoders),
                "request_num": self.request_num,
            }

    # ---- 实例管理 ----
    def get_waiting_nodes(self) -> Dict[str, Tuple[str, Tuple[str, int], int]]:
        with self._lock:
            return dict(self.waiting_nodes)

    def add_instances_to_waiting(self, instance_type: str, instances: List[Tuple[str, int]]) -> List[str]:
        with self._lock:
            waiting_list = []
            for host, port in instances:
                key = self._normalize_key(host, port)
                if key not in self.waiting_nodes:
                    self.waiting_nodes[key] = (instance_type, (host, port), 0)
                    waiting_list.append(f"{host}:{port}")
            return waiting_list

    def mark_waiting_retry(self, key: str, retry_count: int) -> None:
        with self._lock:
            if key in self.waiting_nodes:
                inst_type, server, _ = self.waiting_nodes[key]
                self.waiting_nodes[key] = (inst_type, server, retry_count)

    def activate_waiting_instance(self, key: str) -> None:
        with self._lock:
            if key in self.waiting_nodes:
                inst_type, (host, port), _ = self.waiting_nodes.pop(key)
                if inst_type == InstanceType.PREFILL:
                    self._add_prefiller(host, port)
                else:
                    self._add_decoder(host, port)

    def drop_waiting_instance(self, key: str) -> None:
        with self._lock:
            self.waiting_nodes.pop(key, None)

    def _add_prefiller(self, host: str, port: int) -> None:
        """添加一个 prefiller 实例（需持有锁）。"""
        idx = self.num_prefillers
        self.prefiller_hosts.append((host, port))
        self.prefill_active_tokens.append(0.0)
        self.prefill_active_kv_cache.append(0.0)
        self.num_prefillers += 1

        # 重新构建分组
        self._rebuild_prefiller_groups()
        logger.warning(f"Added prefiller: {host}:{port}")

    def _add_decoder(self, host: str, port: int) -> None:
        """添加一个 decoder 实例（需持有锁）。"""
        idx = self.num_decoders
        self.decoder_hosts.append((host, port))
        self.decode_active_tokens.append(0.0)
        self.num_decoders += 1
        heapq.heappush(self.decoder_heap, (0.0, idx))
        logger.warning(f"Added decoder: {host}:{port}")

    def _rebuild_prefiller_groups(self) -> None:
        """重建 prefiller 分组和堆（需持有锁）。"""
        self._prefill_groups = self._build_groups(self.num_prefillers, self.num_prefill_groups)
        self.prefiller_heaps = []
        self.server_idx_to_group = {}
        self.prefill_group_load = {}

        for group_id, idx_list in enumerate(self._prefill_groups):
            heap = [(0.0, idx) for idx in idx_list]
            heapq.heapify(heap)
            self.prefiller_heaps.append(heap)
            self.prefill_group_load[group_id] = 0.0
            for idx in idx_list:
                self.server_idx_to_group[idx] = group_id

    def remove_instances(self, instance_type: str, instances: List[Tuple[str, int]]) -> bool:
        """标记实例为待移除（tainted）或直接移除。返回 True 表示需要等待。"""
        with self._lock:
            if not instances:
                return False

            if self.request_num > 0:
                # 有活跃请求，标记为 tainted
                if instance_type == InstanceType.PREFILL:
                    for host, port in instances:
                        idx = self._find_prefiller_idx(host, port)
                        if idx is not None and idx not in self.tainted_prefillers:
                            self.tainted_prefillers.append(idx)
                    self._reheapify_prefillers_with_taint()
                else:
                    for host, port in instances:
                        idx = self._find_decoder_idx(host, port)
                        if idx is not None and idx not in self.tainted_decoders:
                            self.tainted_decoders.append(idx)
                    self._reheapify_decoders_with_taint()
                logger.warning(f"Tainted {instance_type} instances: {instances}")
                return True
            else:
                # 无活跃请求，直接移除（简化实现）
                logger.warning(f"Directly removed {instance_type} instances: {instances}")
                return False

    def _find_prefiller_idx(self, host: str, port: int) -> Optional[int]:
        key = self._normalize_key(host, port)
        for i, (h, p) in enumerate(self.prefiller_hosts):
            if self._normalize_key(h, p) == key:
                return i
        return None

    def _find_decoder_idx(self, host: str, port: int) -> Optional[int]:
        key = self._normalize_key(host, port)
        for i, (h, p) in enumerate(self.decoder_hosts):
            if self._normalize_key(h, p) == key:
                return i
        return None

    def _reheapify_prefillers_with_taint(self) -> None:
        """将 tainted prefiller 的优先级设为最高，并重建堆。"""
        for group_idx, heap in enumerate(self.prefiller_heaps):
            new_heap = []
            for priority, idx in heap:
                if idx in self.tainted_prefillers:
                    new_heap.append((TAINT_PRIORITY, idx))
                else:
                    new_heap.append((priority, idx))
            heapq.heapify(new_heap)
            self.prefiller_heaps[group_idx] = new_heap

    def _reheapify_decoders_with_taint(self) -> None:
        """将 tainted decoder 的优先级设为最高，并重建堆。"""
        new_heap = []
        for priority, idx in self.decoder_heap:
            if idx in self.tainted_decoders:
                new_heap.append((TAINT_PRIORITY, idx))
            else:
                new_heap.append((priority, idx))
        heapq.heapify(new_heap)
        self.decoder_heap = new_heap

    def finalize_tainted_instances(self) -> None:
        """在无活跃请求时最终移除 tainted 实例。"""
        with self._lock:
            if self.request_num != 0:
                return
            # 简单实现：记录待移除并重启时需要重建
            # TODO 生产环境可在此实现真正的移除逻辑
            if self.tainted_prefillers:
                logger.warning(f"Finalizing tainted prefillers: {self.tainted_prefillers}")
                self.tainted_prefillers.clear()
            if self.tainted_decoders:
                logger.warning(f"Finalizing tainted decoders: {self.tainted_decoders}")
                self.tainted_decoders.clear()

    @staticmethod
    def _normalize_key(host: str, port: int) -> str:
        norm_host = host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return f"{norm_host}:{port}"


# ------------------------------------------------------------
# 3. 管理进程辅助
# ------------------------------------------------------------
shared_scheduler: Optional[SharedProxyScheduler] = None
global_args: Optional[argparse.Namespace] = None


class SchedulerManager(BaseManager):
    pass


def get_shared_scheduler() -> SharedProxyScheduler:
    if shared_scheduler is None:
        raise RuntimeError("Shared scheduler not initialized")
    return shared_scheduler


SchedulerManager.register("get_scheduler", callable=get_shared_scheduler)


def serialize_args(args: argparse.Namespace) -> dict:
    return {
        "port": args.port,
        "host": args.host,
        "prefiller_hosts": args.prefiller_hosts,
        "prefiller_ports": args.prefiller_ports,
        "decoder_hosts": args.decoder_hosts,
        "decoder_ports": args.decoder_ports,
        "max_retries": args.max_retries,
        "retry_delay": args.retry_delay,
        "max_waiting_retries": args.max_waiting_retries,
        "waiting_retry_interval": args.waiting_retry_interval,
        "prefill_group_threshold": args.prefill_group_threshold,
        "max_request_tokens": args.max_request_tokens,
        "enable_dynamic_bucket": args.enable_dynamic_bucket,
        "workers": args.workers,
    }


def deserialize_args(raw: dict) -> argparse.Namespace:
    args = argparse.Namespace(**raw)
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def start_shared_scheduler(args: argparse.Namespace) -> None:
    global shared_scheduler
    shared_scheduler = SharedProxyScheduler(
        prefiller_instances=args.prefiller_instances,
        decoder_instances=args.decoder_instances,
        enable_dynamic_bucket=args.enable_dynamic_bucket,
        prefill_group_threshold=args.prefill_group_threshold,
        max_request_tokens=args.max_request_tokens,
        max_waiting_retries=args.max_waiting_retries,
        waiting_retry_interval=args.waiting_retry_interval,
    )

    # 启动节点监听线程
    node_listener = threading.Thread(target=_node_listener_loop, daemon=True)
    node_listener.start()

    authkey = os.urandom(16)
    manager = SchedulerManager(address=("127.0.0.1", 0), authkey=authkey)
    server = manager.get_server()
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    host, port = server.address
    os.environ[MANAGER_CONFIG_ENV] = json.dumps({
        "host": host, "port": port,
        "authkey": base64.b64encode(authkey).decode("ascii")
    })
    os.environ[ARGS_CONFIG_ENV] = json.dumps(serialize_args(args))


def _node_listener_loop() -> None:
    """管理进程中的节点监听线程。"""
    import asyncio as async_io
    while True:
        scheduler = shared_scheduler
        if scheduler is None:
            time.sleep(1)
            continue

        waiting = scheduler.get_waiting_nodes()
        for key, (inst_type, (host, port), retries) in list(waiting.items()):
            is_valid = async_io.run(_check_instance_status(host, port))
            retries += 1
            if is_valid:
                scheduler.activate_waiting_instance(key)
                print(f"Instance {key} activated")
            elif retries >= global_args.max_waiting_retries:
                scheduler.drop_waiting_instance(key)
                print(f"Instance {key} dropped after {retries} retries")
            else:
                scheduler.mark_waiting_retry(key, retries)

        scheduler.finalize_tainted_instances()
        time.sleep(global_args.waiting_retry_interval)


async def _check_instance_status(host: str, port: int) -> bool:
    """检查实例是否可用。"""
    endpoint = "/models"
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
    try:
        import ipaddress as ip_mod
        url = f"http://{host}:{port}/v1"
        try:
            ip = ip_mod.ip_address(host)
            if isinstance(ip, ip_mod.IPv6Address):
                url = f"http://[{host}]:{port}/v1"
        except ValueError:
            pass
        async with httpx.AsyncClient(timeout=5.0, base_url=url) as client:
            response = await client.get(endpoint, headers=headers)
            response.raise_for_status()
            return True
    except (httpx.RequestError, httpx.HTTPStatusError):
        return False


def load_global_args_from_env() -> argparse.Namespace:
    global global_args
    raw = os.environ.get(ARGS_CONFIG_ENV)
    if raw is None:
        raise RuntimeError(f"{ARGS_CONFIG_ENV} not set")
    global_args = deserialize_args(json.loads(raw))
    return global_args


def connect_shared_scheduler() -> SharedProxyScheduler:
    config = os.environ.get(MANAGER_CONFIG_ENV)
    if config is None:
        raise RuntimeError(f"{MANAGER_CONFIG_ENV} not set")
    cfg = json.loads(config)
    manager = SchedulerManager(
        address=(cfg["host"], cfg["port"]),
        authkey=base64.b64decode(cfg["authkey"]),
    )
    manager.connect()
    return manager.get_scheduler()


# ------------------------------------------------------------
# 4. Worker 端运行时（每个 worker 进程一份）
# ------------------------------------------------------------
class WorkerRuntime:
    def __init__(self, scheduler: SharedProxyScheduler, args: argparse.Namespace):
        self.scheduler = scheduler
        self.args = args
        self.config = scheduler.get_config()  # 本地缓存配置
        # 本地客户端池，key = "host:port"
        self.prefiller_clients: Dict[str, httpx.AsyncClient] = {}
        self.decoder_clients: Dict[str, httpx.AsyncClient] = {}

    async def sync_clients(self) -> None:
        """同步本地客户端池与调度器的最新服务器列表。"""
        snapshot = self.scheduler.get_snapshot()

        # 同步 prefiller 客户端
        prefiller_targets = {s.key: (s.host, s.port) for s in snapshot["prefill_instances"]}
        await self._sync_group(self.prefiller_clients, prefiller_targets)

        # 同步 decoder 客户端
        decoder_targets = {s.key: (s.host, s.port) for s in snapshot["decode_instances"]}
        await self._sync_group(self.decoder_clients, decoder_targets)

    async def _sync_group(
        self,
        client_group: Dict[str, httpx.AsyncClient],
        targets: Dict[str, Tuple[str, int]],
    ) -> None:
        """同步一组客户端。"""
        stale = [key for key in client_group if key not in targets]
        for key in stale:
            client = client_group.pop(key)
            await client.aclose()
        for key, (host, port) in targets.items():
            if key not in client_group:
                client_group[key] = httpx.AsyncClient(
                    timeout=None,
                    base_url=self._build_base_url(host, port),
                    limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
                )

    @staticmethod
    def _build_base_url(host: str, port: int) -> str:
        import ipaddress as ip_mod
        url = f"http://{host}:{port}/v1"
        try:
            ip = ip_mod.ip_address(host)
            if isinstance(ip, ip_mod.IPv6Address):
                url = f"http://[{host}]:{port}/v1"
        except ValueError:
            pass
        return url

    def get_prefiller_client(self, key: str) -> httpx.AsyncClient:
        return self.prefiller_clients[key]

    def get_decoder_client(self, key: str) -> httpx.AsyncClient:
        return self.decoder_clients[key]

    async def close(self) -> None:
        for client in list(self.prefiller_clients.values()):
            await client.aclose()
        for client in list(self.decoder_clients.values()):
            await client.aclose()
        self.prefiller_clients.clear()
        self.decoder_clients.clear()


# ------------------------------------------------------------
# 5. FastAPI 应用与请求处理
# ------------------------------------------------------------
runtime: Optional[WorkerRuntime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime
    args = load_global_args_from_env()
    scheduler = connect_shared_scheduler()
    runtime = WorkerRuntime(scheduler, args)
    await runtime.sync_clients()
    print(
        f"Worker {os.getpid()} initialized with "
        f"{len(runtime.prefiller_clients)} prefiller(s) and "
        f"{len(runtime.decoder_clients)} decoder(s) client(s)."
    )
    yield
    await runtime.close()
    runtime = None


app = FastAPI(lifespan=lifespan)


def create_app():
    load_global_args_from_env()
    return app


async def listen_for_disconnect(request: Request) -> None:
    """Return if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancel_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait(
            [handler_task, cancel_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None
    return wrapper


async def send_request_to_service(
    client: httpx.AsyncClient,
    prefiller_idx: int,
    endpoint: str,
    req_data: dict,
    request_id: str,
    scheduler: SharedProxyScheduler,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    aborted_requests = scheduler.acquire_aborted_prefiller_requests(prefiller_idx)
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        "aborted_request": list(aborted_requests),
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    req_data["min_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(f"Attempt {attempt} failed for {endpoint}: {e}")
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for {endpoint}.")
                raise last_exc


async def stream_service_response_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as resp:
                resp.raise_for_status()
                first_chunk_sent = False
                async for chunk in resp.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {e}")
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                raise e
        except Exception as e:
            if "first_chunk_sent" in locals() and first_chunk_sent:
                logger.error(f"Stream interrupted after response started: {e}")
                return
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {e}")
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                raise e


async def _handle_select_instance(api: str, req_data: dict, request_length: int) -> InstanceInfo:
    await runtime.sync_clients()
    scheduler = runtime.scheduler

    # 计算 prefiller 分数
    if runtime.config["enable_dynamic_bucket"]:
        prefiller_score = scheduler.calculate_prefill_tokens(request_length)
    else:
        prefiller_score = scheduler.calculate_prefill_scores(request_length)

    request_id = scheduler.next_req_id()
    request_tokens = scheduler.calculate_prefill_tokens(request_length)
    group_idx, task_id = scheduler.select_prefill_group(request_id, request_tokens, prefiller_score)

    logger.warning(f'Selected group_idx: {group_idx}')

    prefiller_idx = scheduler.select_prefiller(group_idx, prefiller_score, task_id)

    # 获取服务器快照找到 key
    snapshot = scheduler.get_snapshot()
    prefiller_key = snapshot["prefill_instances"][prefiller_idx].key

    # 发送请求到 prefiller
    prefiller_client = runtime.get_prefiller_client(prefiller_key)
    response = await send_request_to_service(
        prefiller_client,
        prefiller_idx,
        api,
        req_data,
        request_id,
        scheduler,
        max_retries=runtime.args.max_retries,
        base_delay=runtime.args.retry_delay,
    )
    scheduler.release_prefiller(prefiller_idx, prefiller_score, task_id)

    response_json = response.json()
    kv_transfer_params = response_json.get("kv_transfer_params", {})
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params

    # 选择 decoder
    decoder_score = scheduler.calculate_decode_scores(request_length)
    decoder_idx = scheduler.select_decoder(decoder_score)
    decoder_key = snapshot["decode_instances"][decoder_idx].key

    logger.debug(f"Using prefiller {prefiller_key} and decoder {decoder_key}")
    return InstanceInfo(
        request_id=request_id,
        prefiller_idx=prefiller_idx,
        prefiller_score=prefiller_score,
        prefiller_key=prefiller_key,
        decoder_idx=decoder_idx,
        decoder_score=decoder_score,
        decoder_key=decoder_key,
        task_id=task_id,
    )


async def _handle_completions(api: str, request: Request):
    scheduler = runtime.scheduler
    scheduler.request_started()
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        instance_info = await _handle_select_instance(api, req_data, request_length)
        stream_flag = bool(req_data.get("stream", False))
        chat_flag = "messages" in req_data

        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        elif chat_flag:
            messages = req_data["messages"]
            origin_prompt = messages[0].get("content", "")
        else:
            origin_prompt = ""
        origin_max_tokens = req_data.get("max_tokens", 16)

        decoder_client = runtime.get_decoder_client(instance_info.decoder_key)

        async def generate_stream():
            nonlocal instance_info
            generated_token = ""
            released_kv = False
            retry_count = 0
            retry = True
            completion_tokens = 0
            try:
                while retry:
                    retry = False
                    async for chunk in stream_service_response_with_retry(
                        decoder_client,
                        api,
                        req_data,
                        request_id=instance_info.request_id,
                        max_retries=runtime.args.max_retries,
                        base_delay=runtime.args.retry_delay,
                    ):
                        if not released_kv and chunk:
                            scheduler.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
                            released_kv = True
                        try:
                            chunk_str = chunk.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            yield chunk
                            continue
                        if not chunk_str:
                            continue
                        if chunk_str.startswith("data: "):
                            chunk_str = chunk_str[len("data: "):]
                        try:
                            chunk_json = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            yield chunk
                            continue
                        choices = chunk_json.get("choices", [])
                        if not choices:
                            yield chunk
                            continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        content = delta.get("content") or message.get("content") or choice.get("text") or ""
                        generated_token += content

                        stop_reason = choice.get("stop_reason")
                        usage = chunk_json.get("usage", {})
                        completion_tokens = (
                            (completion_tokens + 1)
                            if stream_flag
                            else (completion_tokens + usage.get("completion_tokens"))
                        )
                        if stop_reason == "recomputed":
                            retry = True
                            retry_count += 1
                            if chat_flag:
                                messages[0]["content"] = origin_prompt + generated_token
                            else:
                                req_data["prompt"] = origin_prompt + generated_token
                            req_data["max_tokens"] = origin_max_tokens - completion_tokens + retry_count
                            tmp_request_length = len(json.dumps(req_data).encode("utf-8"))
                            instance_info = await _handle_select_instance(api, req_data, tmp_request_length)
                            break
                        if retry_count > 0 and not stream_flag:
                            if chat_flag:
                                choice["message"]["content"] = generated_token
                            else:
                                choice["text"] = generated_token
                            chunk = json.dumps(chunk_json).encode("utf-8")
                        yield chunk
            except Exception as e:
                logger.error(
                    f"Error during streaming from decoder {instance_info.decoder_key}: {e} "
                    f"the aborted request {instance_info.request_id} will be routing to the target "
                    "prefiller when new request is ready to dispatch to it"
                )
                scheduler.abort_prefiller_request(instance_info.prefiller_idx, instance_info.request_id)
                scheduler.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
            finally:
                scheduler.release_decoder(instance_info.decoder_idx, instance_info.decoder_score)
                scheduler.request_finished()

        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        return StreamingResponse(generate_stream(), media_type=media_type)
    except Exception as e:
        import traceback
        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


async def _handle_adjust_instances(adjust_mode: str, request: Request):
    try:
        req_data = await request.json()
        instance_type = req_data.get("type", "")
        instances = req_data.get("instances", [])
        if isinstance(instances, str):
            instances = [instances]
        instances = _trans_instances(instances)
        all_msg = f"{adjust_mode} {instance_type} instances: {instances}."

        if instance_type not in [InstanceType.PREFILL, InstanceType.DECODE]:
            return {
                "error": f"Instance type {instance_type} is not supported. "
                         f"Only support '{InstanceType.PREFILL}' and '{InstanceType.DECODE}'."
            }

        scheduler = runtime.scheduler
        if adjust_mode == "add":
            raw_instances = [(host, port) for host, port in instances]
            waiting_nodes = scheduler.add_instances_to_waiting(instance_type, raw_instances)
            if waiting_nodes:
                all_msg = (
                    f"{adjust_mode} {instance_type} instances: {raw_instances}. "
                    f"Instances {waiting_nodes} are waiting to be added."
                )
        elif adjust_mode == "remove":
            raw_instances = [(host, port) for host, port in instances]
            need_waiting = scheduler.remove_instances(instance_type, raw_instances)
            if need_waiting:
                all_msg = f"Instances {instances} are isolated and waiting to be removed."

        snapshot = scheduler.get_snapshot()
        return {
            "message": all_msg,
            "current_prefill_instances": [s.key for s in snapshot["prefill_instances"]],
            "current_decode_instances": [s.key for s in snapshot["decode_instances"]],
        }
    except Exception as e:
        logger.error(f"Failed to {adjust_mode} instances: {e}")
        raise e


def _trans_instances(instances: list) -> List[Tuple[str, int]]:
    result = []
    for instance in instances:
        if isinstance(instance, str):
            host, port = instance.split(":")
            result.append((host, int(port)))
        else:
            result.append(instance)
    return result


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return runtime.scheduler.healthcheck()


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await _handle_adjust_instances("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await _handle_adjust_instances("remove", request)


# ------------------------------------------------------------
# 6. 启动入口
# ------------------------------------------------------------
def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=0.001)
    parser.add_argument("--max-waiting-retries", type=int, default=3)
    parser.add_argument("--waiting-retry-interval", type=float, default=10)
    parser.add_argument("--prefill-group-threshold", type=int, default=32 * 1024)
    parser.add_argument("--max-request-tokens", type=int, default=128 * 1024)
    parser.add_argument("--enable-dynamic-bucket", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn worker processes")
    args = parser.parse_args(args_list)
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


if __name__ == "__main__":
    global_args = parse_args()
    start_shared_scheduler(global_args)

    import uvicorn
    module_name = Path(__file__).stem
    uvicorn.run(
        f"{module_name}:create_app",
        host=global_args.host,
        port=global_args.port,
        workers=global_args.workers,
        factory=True,
        app_dir=str(Path(__file__).resolve().parent),
    )
