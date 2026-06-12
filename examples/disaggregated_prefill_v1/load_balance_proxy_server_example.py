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
from typing import Any
from typing import List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from dynamic_bucket_load_balancer import DynamicBucketLoadBalancer, Task, Bucket, ServerInfo

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Add uvloop for faster event loop if available
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


@dataclass
class InstanceType:
    PREFILL: str = "prefill"
    DECODE: str = "decode"


TAINT_PRIORITY = 1e15


class ServerState:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        try:
            ip = ipaddress.ip_address(self.host)
            if isinstance(ip, ipaddress.IPv6Address):
                self.url = f"http://[{host}]:{port}/v1"
        except Exception:
            pass
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.active_tokens = 0
        self.active_kv_cache = 0  # Only for prefiller
        self.active_requests = 0  # Number of active requests
        self.aborted_requests = set()  # Track aborted requests
        # Removed individual server lock - will use global locks instead

    def __eq__(self, other):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        other_host = other.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return self_host == other_host and str(self.port) == str(other.port)

    def __hash__(self):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return hash((self_host, str(self.port)))

    def __repr__(self):
        return f"{self.host}:{self.port}"

@dataclass(order=True)
class ServerHeapItem:
    priority: float
    server_idx: int
    server: ServerState


class ProxyState:
    def __init__(self, prefiller_instances, decoder_instances):
        self.request_num = 0
        self.tainted_prefillers: list[ServerState] = []
        self.tainted_decoders: list[ServerState] = []
        self.node_listener = NodeListener(self)

        self.prefillers: list[ServerState] = [ServerState(h, p) for h, p in prefiller_instances]
        self.decoders: list[ServerState] = [ServerState(h, p) for h, p in decoder_instances]
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()
        # Removed selection locks - no longer needed for synchronous methods

        if global_args.enable_dynamic_bucket:
            self.num_prefill_groups = 2 # 启用动态分桶时的分组数量
        else:
            self.num_prefill_groups = 1 # 默认不分组

        # Initialize priority queues for efficient server selection
        # Each entry is (priority_score, server_index, server_reference)
        # Lower priority score = higher priority (less loaded)
        prefiller_heap_items = [ServerHeapItem(0.0, i, server) for i, server in enumerate(self.prefillers)]
        # TODO self.prefiller_heaps、self.server_idx_to_group_idx、self.prefill_group_load 等分桶变量在动态扩缩容时需要适配
        # 对Server进行分组
        self.prefiller_heaps: List[List[ServerHeapItem]] = self._group_servers(prefiller_heap_items, self.num_prefill_groups)
        self.server_idx_to_group_idx = {}
        # TODO self.prefill_group_load 可以删掉？
        self.prefill_group_load = {}
        # 堆化每一分组
        for idx, cur_heap in enumerate(self.prefiller_heaps):
            for server_item in cur_heap:
                self.server_idx_to_group_idx[server_item.server_idx] = idx
            heapq.heapify(cur_heap)
            self.prefill_group_load[idx] = 0

        logger.warning(f'Test==== len(self.prefiller_heaps): {len(self.prefiller_heaps)}')
        logger.warning(f'Test==== self.prefiller_heaps: {self.prefiller_heaps}')

        logger.warning(f'Test==== global_args.enable_dynamic_bucket: {global_args.enable_dynamic_bucket}')

        # add dynamic bucket load balancer
        self.bucket_load_balancer: Optional[DynamicBucketLoadBalancer] = None
        if global_args.enable_dynamic_bucket:
            self.prefill_group_threshold = global_args.prefill_group_threshold
            prefill_buckets = [(0, self.prefill_group_threshold),
                               (self.prefill_group_threshold, global_args.max_request_tokens)]

            if self.num_prefill_groups != len(prefill_buckets):
                raise ValueError("Number of prefill groups must match number of prefill buckets")

            self.bucket_load_balancer = DynamicBucketLoadBalancer(buckets=prefill_buckets)

        self.decoder_heap:List[ServerHeapItem] = [ServerHeapItem(0.0, i, server) for i, server in enumerate(self.decoders)]
        heapq.heapify(self.decoder_heap)

    @staticmethod
    def _group_servers(servers: List[ServerHeapItem], num_groups: int):
        """
        Group servers into num_groups groups.

        Args:
            servers (list): servers to be grouped.
            num_groups (int): num of groups.

        Returns:
            list[list]: grouped list of servers.

        Raises:
            ValueError: if num_groups <= 0.
        """
        if num_groups <= 0:
            raise ValueError("Num of group is illegal")

        if len(servers) < num_groups:
            raise ValueError("Number of servers must greater than or equal to number of groups")

        n = len(servers)
        if n == 0:
            return [[] for _ in range(num_groups)]
        elif n == 1:
            return [servers]

        base_size = n // num_groups
        remainder = n % num_groups

        groups = []
        start_index = 0
        for i in range(num_groups):
            group_size = base_size + 1 if i < remainder else base_size
            end_index = start_index + group_size
            groups.append(servers[start_index:end_index])
            start_index = end_index

        return groups

    @staticmethod
    def _add_new_server_to_heaps(server_heaps: List[List[ServerHeapItem]], new_server_heap_item:ServerHeapItem):
        """
        添加新的 Server 到 Server堆分组 中，并返回新的 Server堆分组 和 添加的Server所在堆分组索引
        """
        num_heaps = len(server_heaps)

        # 仅一个分组
        if num_heaps == 1:
            heapq.heappush(server_heaps[0], new_server_heap_item)
            return server_heaps, 0

        for idx, group in enumerate(server_heaps):
            # 当前分组比下一分组元素多，将新Server添加到下一分组
            if idx < num_heaps - 1 and len(server_heaps[idx]) > len(server_heaps[idx + 1]):
                heapq.heappush(server_heaps[idx + 1], new_server_heap_item)
                return server_heaps, idx + 1

        # 所有分组元素个数相同，添加到第一分组
        heapq.heappush(server_heaps[0], new_server_heap_item)
        return server_heaps, 0

    def _update_prefiller_priority(self, server_idx: int):
        """Update the priority of a prefiller server in the heap."""
        server = self.prefillers[server_idx]
        # Priority based on active_tokens and active_kv_cache
        priority = server.active_tokens + server.active_kv_cache * 0.3
        # Remove old entry and add new one
        group_idx = self.server_idx_to_group_idx[server_idx]

        self.prefiller_heaps[group_idx] = [server_heap_item for server_heap_item in self.prefiller_heaps[group_idx] if server_heap_item.server_idx != server_idx]
        self.prefiller_heaps[group_idx].append(ServerHeapItem(priority, server_idx, server))
        heapq.heapify(self.prefiller_heaps[group_idx])

    def _update_decoder_priority(self, server_idx: int):
        """Update the priority of a decoder server in the heap."""
        server = self.decoders[server_idx]
        priority = server.active_tokens
        # Remove old entry and add new one
        self.decoder_heap = [server_heap_item for server_heap_item in self.decoder_heap if server_heap_item.server_idx != server_idx]
        self.decoder_heap.append(ServerHeapItem(priority, server_idx, server))
        heapq.heapify(self.decoder_heap)

    def abort_prefiller_request(self, server_idx: int, request_id):  # Changed to synchronous
        """
        Mark a request as aborted. This will helps to release kv cache in
        prefiller node.
        """
        # No lock needed - atomic operation
        if server_idx >= len(self.prefillers):
            return
        self.prefillers[server_idx].aborted_requests.add(request_id)

    def acquire_aborted_prefiller_requests(self, server_idx: int):  # Changed to synchronous
        """
        Get the set of aborted requests and clear it.
        This is used to release kv cache in prefiller node.
        """
        # No lock needed - atomic operation
        if server_idx >= len(self.prefillers):
            return set()
        aborted_requests = self.prefillers[server_idx].aborted_requests.copy()
        self.prefillers[server_idx].aborted_requests.clear()
        return aborted_requests

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def select_prefiller(self, token_count, group_idx=0):  # Changed to synchronous
        # No lock needed - entire function is atomic
        if not self.prefillers:
            raise RuntimeError("No prefiller servers available")

        server_heap_item:ServerHeapItem = heapq.heappop(self.prefiller_heaps[group_idx])
        chosen_server_idx = server_heap_item.server_idx

        # Update the chosen server atomically
        self.prefillers[chosen_server_idx].active_tokens += token_count
        self.prefillers[chosen_server_idx].active_kv_cache += token_count
        self.prefill_group_load[group_idx] += token_count

        # Update priority and re-add to heap
        self._update_prefiller_priority(chosen_server_idx)

        return chosen_server_idx

    def release_prefiller(self, idx, token_count,task=None):  # Changed to synchronous
        # No lock needed - atomic operation
        if idx >= len(self.prefillers):
            return
        self.prefillers[idx].active_tokens -= token_count
        group_idx = self.server_idx_to_group_idx[idx]
        self.prefill_group_load[group_idx] -= token_count
        if global_args.enable_dynamic_bucket and task is not None:
            self.bucket_load_balancer.release_task(task.id)
        # Update priority queue after releasing
        self._update_prefiller_priority(idx)

    def release_prefiller_kv(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        if idx >= len(self.prefillers):
            return
        if self.prefillers[idx].active_kv_cache > 0:
            self.prefillers[idx].active_kv_cache -= token_count
        # Update priority queue after releasing
        self._update_prefiller_priority(idx)

    def select_decoder(self, token_count):  # Changed to synchronous
        # No lock needed - entire function is atomic
        if not self.decoder_heap:
            raise RuntimeError("No decoder servers available")

        server_heap_item = heapq.heappop(self.decoder_heap)
        chosen_server_idx = server_heap_item.server_idx

        # Update the chosen server atomically
        self.decoders[chosen_server_idx].active_tokens += token_count

        # Update priority and re-add to heap
        self._update_decoder_priority(chosen_server_idx)

        return chosen_server_idx

    def release_decoder(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        if idx >= len(self.decoders):
            return
        self.decoders[idx].active_tokens -= token_count
        # Update priority queue after releasing
        self._update_decoder_priority(idx)

    # Omni_infer's calculate_input_scores function
    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        input_score = length_score * 0.0345 + 120.0745
        return input_score

    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length

    def calculate_prefill_tokens(self, request_length: int) -> float:
        return request_length / 4.0

    def select_prefill_group(self, req_id: str, request_tokens, prefill_score) -> tuple[int, Task | None]:
        """
        Find Best group by request length and current load of groups
        """
        if global_args.enable_dynamic_bucket:
            group_idx, task = self.bucket_load_balancer.dispatch_single_task(req_id, request_tokens, prefill_score)
            return group_idx, task
        else:
            return 0, None

    async def add_instances(self, instance_type: str, instances: list[ServerState]) -> tuple[list[str], list[str]]:
        added_nodes, waiting_nodes = [], []
        for server in instances:
            is_valid = await self.node_listener.check_instance_status(server.client)
            if is_valid and instance_type == InstanceType.PREFILL:
                self.add_prefillers([server])
                added_nodes.append(str(server))
            elif is_valid and instance_type == InstanceType.DECODE:
                self.add_decoders([server])
                added_nodes.append(str(server))
            else:
                node = str(server)
                self.node_listener.waiting_nodes[node] = (instance_type, server, 0)
                waiting_nodes.append(node)
        return added_nodes, waiting_nodes

    def add_prefillers(self, instances: list[ServerState]) -> None:
        for server in instances:
            if server in self.tainted_prefillers:
                self.tainted_prefillers.remove(server)

                # 适配动态分桶
                for group_idx, heap in enumerate(self.prefiller_heaps):
                    re_heapify_flag = False
                    for server_heap_item in heap:
                        if server_heap_item.server == server:
                            server_heap_item.priority = 0
                            self.server_idx_to_group_idx[server_heap_item.server_idx] = group_idx
                            re_heapify_flag = True
                    if re_heapify_flag:
                        heapq.heapify(heap)
            elif server not in self.prefillers:
                self.prefillers.append(server)

                new_prefiller_server_idx = len(self.prefillers) - 1
                new_server_heap_item = ServerHeapItem(0, new_prefiller_server_idx, server)
                self.prefiller_heaps, group_idx = self._add_new_server_to_heaps(self.prefiller_heaps, new_server_heap_item)
                self.server_idx_to_group_idx[new_prefiller_server_idx] = group_idx

        self.print_status(f"Add prefiller instances: {instances}.")

    def add_decoders(self, instances: list[ServerState]) -> None:
        for server in instances:
            if server in self.tainted_decoders:
                self.tainted_decoders.remove(server)

                for server_heap_item in self.decoder_heap:
                    if server_heap_item.server == server:
                        server_heap_item.priority = 0
                heapq.heapify(self.decoder_heap)

            elif server not in self.decoders:
                self.decoders.append(server)
                # decoder_heap: [(priority_0, 0, server_0)] -> [(priority_0, 0, server_0), (0, 1, server_1)]
                heapq.heappush(self.decoder_heap, ServerHeapItem(0, len(self.decoders) - 1, server))
        self.print_status(f"Add decoder instances: {instances}.")

    def remove_prefillers(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False

        if self.request_num > 0:
            logger.warning(f"Start to taint prefill instances {instances}.")
            self._taint_prefillers(instances)
            return True

        instances_to_remove = set(instances)

        if len(self.prefillers) - len(instances_to_remove) < self.num_prefill_groups:
            logger.warning("Number of prefillers must greater than or equal to number of groups")
            return False

        #################################11111
        # TODO 需要适配，删除prefill时，由于被删除的prefill导致prefillers列表变化，需要更新桶的负载
        new_prefillers = []
        old_idx_to_new_idx = {}
        for idx, server in enumerate(self.prefillers):
            if server not in instances_to_remove:
                new_prefillers.append(server)
                old_idx_to_new_idx[idx] = len(new_prefillers) - 1
            else:
                if server.active_tokens!=0 or server.active_kv_cache!=0 or server.active_requests!=0:
                    logger.warning("Prefill server is not empty, please wait for all requests to be completed")
                    return False

        self.prefillers = new_prefillers

        old_idx_to_new_heap_item = {}
        for group_idx, heap in enumerate(self.prefiller_heaps):
            for server_item in heap:
                if server_item.server not in instances_to_remove:
                    old_idx_to_new_heap_item[server_item.server_idx] = ServerHeapItem(server_item.priority, old_idx_to_new_idx[server_item.server_idx], server_item.server)

        self.prefiller_heaps = self._group_servers(list(old_idx_to_new_heap_item.values()), self.num_prefill_groups)
        for group_idx, heap in enumerate(self.prefiller_heaps):
            for server_item in heap:
                self.server_idx_to_group_idx[server_item.server_idx] = group_idx
            heapq.heapify(heap)

        # TODO 需随之更新，self.prefill_group_load = {}，分数计算是否正确？
        for group_idx, heap in enumerate(self.prefiller_heaps):
            self.prefill_group_load[group_idx] = 0
            for server_item in heap:
                self.prefill_group_load[group_idx] += server_item.server.active_tokens

        # 更新桶的负载 TODO 待验证
        if global_args.enable_dynamic_bucket:
            tasks = self.bucket_load_balancer.tasks

            # 创建新桶，并清空原始桶的统计消息
            new_buckets: dict[int, Bucket] = copy.deepcopy(self.bucket_load_balancer.buckets)
            for bucket in new_buckets.values():
                bucket.task_count = 0
                bucket.total_load = 0.0

            # TODO 会有线程安全问题？
            # for task_queue in tasks.values():
            #     for task in list(task_queue.queue):
            #         if InstanceType.PREFILL == task.server_info[0]:
            #             # 更新 task 信息
            #             new_idx = old_idx_to_new_idx[task.server_info[1]]
            #             task.server_info[1] = new_idx
            #             new_group_idx = self.server_idx_to_group_idx[new_idx]
            #             task.bucket_idx = new_group_idx
            #             # 更新 bucket 信息
            #             new_buckets[new_group_idx].task_count += 1
            #             new_buckets[new_group_idx].total_load += task.length
            for task in tasks.values():
                if InstanceType.PREFILL == task.server_info.instance_type:
                    # 更新 task 信息
                    new_idx = old_idx_to_new_idx[task.server_info.instance_idx]
                    task.server_info.instance_idx = new_idx
                    new_group_idx = self.server_idx_to_group_idx[new_idx]
                    task.bucket_idx = new_group_idx
                    # 更新 bucket 信息
                    new_buckets[new_group_idx].task_count += 1
                    new_buckets[new_group_idx].total_load += task.load


            self.bucket_load_balancer.buckets = new_buckets

            self.print_status(f"Remove prefiller instances: {instances}.")
        return False
        #################################11111

    def remove_decoders(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False

        if self.request_num > 0:
            logger.warning(f"Start to taint decode instances {instances}.")
            self._taint_decoders(instances)
            return True

        instances_to_remove = set(instances)
        self.decoders = [server for server in self.decoders if server not in instances_to_remove]
        decoder_heap_copy:List[ServerHeapItem] = self.decoder_heap.copy()
        decoder_heap_copy.sort(key=lambda x: x.server_idx)  # sorted by key: decoder_idx
        decoder_heap = []
        idx = 0
        for server_heap_item in decoder_heap_copy:
            if server_heap_item.server not in instances_to_remove:
                decoder_heap.append(ServerHeapItem(server_heap_item.priority, idx, server_heap_item.server))
                idx += 1

        self.decoder_heap = decoder_heap
        heapq.heapify(self.decoder_heap)
        self.print_status(f"Remove decoder instances: {instances}.")
        return False

    def _taint_prefillers(self, instances: list[ServerState]) -> None:
        instances_to_taint = set(instances)
        for server in self.prefillers:
            if server in instances_to_taint and server not in self.tainted_prefillers:
                self.tainted_prefillers.append(server)

        for group_idx, heap in enumerate(self.prefiller_heaps):
            re_heapify_flag = False
            for server_item in heap:
                if server_item.server in instances_to_taint:
                    server_item.priority = TAINT_PRIORITY
                    re_heapify_flag = True
            if re_heapify_flag:
                heapq.heapify(heap)

    def _taint_decoders(self, instances: list[ServerState]) -> None:
        instances_to_taint = set(instances)
        for server in self.decoders:
            if server in instances_to_taint and server not in self.tainted_decoders:
                self.tainted_decoders.append(server)

        re_heapify_flag = False
        for server_item in self.decoder_heap:
            if server_item.server in instances_to_taint:
                server_item.priority = TAINT_PRIORITY
                re_heapify_flag = True
        if re_heapify_flag:
            heapq.heapify(self.decoder_heap)

    def print_status(self, msg: str) -> None:
        status = {
            "prefill_instances": [str(server) for server in self.prefillers],
            "decode_instances": [str(server) for server in self.decoders],
        }
        print(f"{msg} Status: {status}")

proxy_state: Optional[ProxyState] = None


class NodeListener:
    def __init__(self, proxy):
        self.proxy_state = proxy
        self.waiting_nodes: dict[str, tuple[str, Any, int]] = {}
        self.listening_thread = threading.Thread(target=self._node_listener, daemon=True)
        self.listening_thread.start()

    def _node_listener(self) -> None:
        while True:
            for node, (instance_type, server, check_times) in list(self.waiting_nodes.items()):
                is_valid = asyncio.run(self.check_instance_status(server.client))
                print(f"Checking instance {node}...")
                check_times += 1
                if is_valid:
                    if instance_type == InstanceType.PREFILL:
                        self.proxy_state.add_prefillers([server])
                    else:
                        self.proxy_state.add_decoders([server])
                    self.waiting_nodes.pop(node)
                elif check_times == global_args.max_waiting_retries:
                    print(f"Instance {node} was not added to the proxy.")
                    self.waiting_nodes.pop(node)
                else:
                    self.waiting_nodes[node] = (instance_type, server, check_times)

            if self.proxy_state.tainted_prefillers and not self.proxy_state.request_num:
                need_waiting = self.proxy_state.remove_prefillers(self.proxy_state.tainted_prefillers)
                if not need_waiting:
                    self.proxy_state.tainted_prefillers.clear()

            if self.proxy_state.tainted_decoders and not self.proxy_state.request_num:
                need_waiting = self.proxy_state.remove_decoders(self.proxy_state.tainted_decoders)
                if not need_waiting:
                    self.proxy_state.tainted_decoders.clear()
            time.sleep(global_args.waiting_retry_interval)

    @staticmethod
    async def check_instance_status(client: httpx.AsyncClient) -> bool:
        endpoint = "/models"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        try:
            response = await client.get(endpoint, headers=headers)
            response.raise_for_status()
            return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False


def parse_args(args_list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    parser.add_argument(
        "--max-waiting-retries", type=int, default=3, help="Maximum number of retries for waiting nodes to be started"
    )
    parser.add_argument(
        "--waiting-retry-interval",
        type=float,
        default=10,
        help="Check interval (seconds) for waiting nodes to be started",
    )
    parser.add_argument("--prefill-group-threshold",
                        type=int,
                        default=32 * 1024,
                        help="Threshold of prefill groups")
    parser.add_argument("--max-request-tokens",
                        type=int,
                        default=128 * 1024,
                        help="Max tokens of request")
    parser.add_argument("--enable-dynamic-bucket",
                        action="store_true",
                        default=False,
                        help="Enable dynamic bucket load Balancer")
    args = parser.parse_args(args_list)
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(global_args.prefiller_instances, global_args.decoder_instances)
    print(f"Initialized {len(proxy_state.prefillers)} prefill clients and {len(proxy_state.decoders)} decode clients.")
    yield
    for p in proxy_state.prefillers:
        await p.client.aclose()
    for d in proxy_state.decoders:
        await d.client.aclose()


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
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


app = FastAPI(lifespan=lifespan)


async def send_request_to_service(
        client: httpx.AsyncClient,
        prefiller_id: int,
        endpoint: str,
        req_data: dict,
        request_id: str,
        max_retries: int = 3,
        base_delay: float = 0.2,
):
    aborted_requests = proxy_state.acquire_aborted_prefiller_requests(prefiller_id)
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
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(f"Attempt {attempt} failed for {endpoint}: {str(e)}")
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
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                response.raise_for_status()
                first_chunk_sent = False
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return  # Success, exit after streaming
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                raise e
        except Exception as e:
            # If any chunk has been sent, do not retry, just log and drop
            if "first_chunk_sent" in locals() and first_chunk_sent:
                logger.error(f"Streaming to client interrupted after response started: {str(e)}")
                return
            else:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                    await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                else:
                    logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                    raise e


async def _handle_select_instance(api: str, req_data: Any, request_length: int):
    prefiller_score = 0
    # TODO asw 计算 prefiller_score 的逻辑在 num_prefill_groups 取不同值时不太一致
    if proxy_state.num_prefill_groups > 1:
        prefiller_score = proxy_state.calculate_prefill_tokens(request_length)
    else:
        prefiller_score = proxy_state.calculate_prefill_scores(request_length)
    logger.debug(f"Request length: {request_length}, Prefiller score: {prefiller_score}")
    request_id = await proxy_state.next_req_id()
    # Select prefiller
    request_tokens = proxy_state.calculate_prefill_tokens(request_length)
    group_idx, task = proxy_state.select_prefill_group(request_id, request_tokens, prefiller_score)

    logger.warning(f'Test =====selected group_idx: {group_idx}')

    prefiller_idx = proxy_state.select_prefiller(prefiller_score, group_idx)

    if global_args.enable_dynamic_bucket and task is not None:
        task.server_info = ServerInfo(InstanceType.PREFILL,prefiller_idx)

    prefiller = proxy_state.prefillers[prefiller_idx]
    # Send request to prefiller
    response = await send_request_to_service(
        prefiller.client,
        prefiller_idx,
        api,
        req_data,
        request_id,
        max_retries=global_args.max_retries,
        base_delay=global_args.retry_delay,
    )
    proxy_state.release_prefiller(prefiller_idx, prefiller_score,task)
    response_json = response.json()
    kv_transfer_params = response_json.get("kv_transfer_params", {})
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params
    # Select decoder
    decoder_score = proxy_state.calculate_decode_scores(request_length)
    logger.debug("Decoder score: %f", decoder_score)
    # Use the prefiller's kv_transfer_params to select decoder
    decoder_idx = proxy_state.select_decoder(decoder_score)
    decoder = proxy_state.decoders[decoder_idx]
    logger.debug("Using %s %s", prefiller.url, decoder.url)
    return InstanceInfo(
        request_id=request_id,
        prefiller_idx=prefiller_idx,
        prefiller_score=prefiller_score,
        prefiller=prefiller,
        decoder=decoder,
        decoder_idx=decoder_idx,
        decoder_score=decoder_score,
    )


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_idx: int
    prefiller_score: float
    prefiller: ServerState
    decoder_idx: int
    decoder_score: float
    decoder: ServerState


async def _handle_completions(api: str, request: Request):
    try:
        proxy_state.request_num += 1
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
        # refer to vLLM sampling_params: max_token default value
        origin_max_tokens = req_data.get("max_tokens", 16)

        async def generate_stream():
            nonlocal instance_info
            generated_token = ""
            released_kv = False
            retry_count = 0
            retry = True
            completion_tokens = 0
            # Only one await per chunk, minimal logic in loop
            try:
                while retry:
                    retry = False
                    async for chunk in stream_service_response_with_retry(
                            instance_info.decoder.client,
                            api,
                            req_data,
                            request_id=instance_info.request_id,
                            max_retries=global_args.max_retries,
                            base_delay=global_args.retry_delay,
                    ):
                        if not released_kv and chunk:
                            proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
                            released_kv = True
                        try:
                            chunk_str = chunk.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            logger.debug(f"Skipping chunk: {chunk}")
                            yield chunk
                            continue
                        if not chunk_str:
                            continue
                        if chunk_str.startswith("data: "):
                            chunk_str = chunk_str[len("data: "):]
                        try:
                            chunk_json = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            # if chunk is [done], skip it.
                            logger.debug(f"Skipping chunk: {chunk_str}")
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
                    f"Error during streaming from decoder {instance_info.decoder.url}: {str(e)} "
                    f"the aborted request {instance_info.request_id} will be routing to the target "
                    "prefiller when new request is ready to dispatch to it"
                )
                proxy_state.abort_prefiller_request(instance_info.prefiller_idx, instance_info.request_id)
                proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)

            # After streaming done, release tokens
            proxy_state.release_decoder(instance_info.decoder_idx, instance_info.decoder_score)

        # Determine the correct media type based on stream flag
        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        return StreamingResponse(generate_stream(), media_type=media_type)
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise
    finally:
        proxy_state.request_num -= 1


async def _handle_adjust_instances(adjust_mode: str, request: Request):
    try:
        req_data = await request.json()
        instance_type = req_data.get("type", "")
        instances = req_data.get("instances", [])
        if isinstance(instances, str):
            instances = [instances]
        instances = trans_instances(instances)
        all_msg = f"{adjust_mode} {instance_type} instances: {[str(server) for server in instances]}."

        if instance_type not in [InstanceType.PREFILL, InstanceType.DECODE]:
            return {
                "error": f"Instance type {instance_type} is not supported. "
                         f"Only support '{InstanceType.PREFILL}' and '{InstanceType.DECODE}'."
            }

        if adjust_mode == "add":
            added_nodes, waiting_nodes = await proxy_state.add_instances(instance_type, instances)
            if waiting_nodes:
                all_msg = (
                    f"{adjust_mode} {instance_type} instances: {added_nodes}. "
                    f"Instances {waiting_nodes} are waiting to be added."
                )
        elif adjust_mode == "remove":
            if instance_type == InstanceType.PREFILL:
                need_waiting = proxy_state.remove_prefillers(instances)
            else:
                need_waiting = proxy_state.remove_decoders(instances)

            if need_waiting:
                all_msg = f"Instances {instances} are isolated and waiting to be removed."
        return {
            "message": all_msg,
            "current_prefill_instances": [str(prefiller) for prefiller in proxy_state.prefillers],
            "current_decode_instances": [str(decoder) for decoder in proxy_state.decoders],
        }
    except Exception as e:
        logger.error(f"Failed to {adjust_mode} instances: {e}")
        raise e


def trans_instances(instances: list[str]) -> list[ServerState]:
    server_list = []
    for instance in instances:
        h, p = instance.split(":")
        server_list.append(ServerState(h, int(p)))
    return server_list


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
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders),
    }


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await _handle_adjust_instances("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await _handle_adjust_instances("remove", request)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
