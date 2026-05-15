# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server For External DP
#
# This proxy server is designed to distribute requests between multiple
# vLLM servers running in data parallel for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple vLLM instances.
#
# Features:
# - Load balances requests to multiple vLLM servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least two vLLM servers running in data parallel.
# These can be mock servers or actual vLLM servers.
# Note that this proxy also works with only one vLLM server running, but
# will fall back to direct request forwarding which is meaningless.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 --data-parallel-rank 0 ... # vLLM DP0
#   vllm serve --host 0.0.0.0 --port 8101 --data-parallel-rank 1 ... # vLLM DP1
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each vLLM DP Instance:
#
#   python dp_load_balance_proxy_server.py \
#     --host 0.0.0.0 --port 9000 \
#     --dp-hosts 127.0.0.1 127.0.0.1 \
#     --dp-ports 8100 8101 \
#
# This will start the proxy on port 9000, load balancing between two vLLM DP servers.
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
# This will return a JSON object with the status and the number of vLLM DP servers.
#
# Notes:
# - You can scale the number of vLLM data parallel size as needed.
# - The proxy will consider the length of requests to balance load.
# - For production, ensure your backend servers are robust and secure.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import functools
import heapq
import os
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, List, Optional

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


class ServerState:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.active_tokens = 0
        self.aborted_requests = set()  # Track aborted requests

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
    def __init__(self, server_instances):
        self.dp_servers: list[ServerState] = [ServerState(h, p) for h, p in server_instances]
        self.req_id_lock = asyncio.Lock()
        # Removed selection locks - no longer needed for synchronous methods

        if global_args.enable_dynamic_bucket:
            self.num_dp_groups = 2 # 启用动态分桶时的分组数量
        else:
            self.num_dp_groups = 1 # 默认不分组

        # Initialize priority queues for efficient server selection
        # Each entry is (priority_score, server_index, server_reference)
        # Lower priority score = higher priority (less loaded)
        dp_heap_items = [ServerHeapItem(0.0, i, server) for i, server in enumerate(self.dp_servers)]
        # 对Server进行分组
        self.dp_heaps: List[List[ServerHeapItem]] = self._group_servers(dp_heap_items, self.num_dp_groups)
        self.server_idx_to_group_idx = {}

        # 堆化每一分组
        for idx, cur_heap in enumerate(self.dp_heaps):
            for server_item in cur_heap:
                self.server_idx_to_group_idx[server_item.server_idx] = idx
            heapq.heapify(cur_heap)

        logger.warning(f'Test==== len(self.dp_heaps): {len(self.dp_heaps)}')
        logger.warning(f'Test==== self.dp_heaps: {self.dp_heaps}')

        logger.warning(f'Test==== global_args.enable_dynamic_bucket: {global_args.enable_dynamic_bucket}')

        # add dynamic bucket load balancer
        self.bucket_load_balancer: Optional[DynamicBucketLoadBalancer] = None
        if global_args.enable_dynamic_bucket:
            self.dp_group_threshold = global_args.dp_group_threshold
            dp_buckets = [(0, self.dp_group_threshold),
                               (self.dp_group_threshold, global_args.max_request_tokens)]

            if self.num_dp_groups != len(dp_buckets):
                raise ValueError("Number of dp groups must match number of dp buckets")

            self.bucket_load_balancer = DynamicBucketLoadBalancer(buckets=dp_buckets,
                                                                  affinity_strength=1.0  # todo: 待调整（0~1.0）
                                                                  )

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

    def _update_server_priority(self, server_idx: int):
        """Update the priority of a decoder server in the heap."""
        server = self.dp_servers[server_idx]
        priority = server.active_tokens
        # Remove old entry and add new one
        group_idx = self.server_idx_to_group_idx[server_idx]

        self.dp_heaps[group_idx] = [server_heap_item for server_heap_item in self.dp_heaps[group_idx] if
                                           server_heap_item.server_idx != server_idx]
        self.dp_heaps[group_idx].append(ServerHeapItem(priority, server_idx, server))
        heapq.heapify(self.dp_heaps[group_idx])

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def select_server(self, token_count, group_idx=0):  # Changed to synchronous
        # No lock needed - entire function is atomic
        if not self.dp_servers:
            raise RuntimeError("No decoder servers available")

        server_heap_item: ServerHeapItem = heapq.heappop(self.dp_heaps[group_idx])
        chosen_server_idx = server_heap_item.server_idx

        # Update the chosen server atomically
        self.dp_servers[chosen_server_idx].active_tokens += token_count

        # Update priority and re-add to heap
        self._update_server_priority(chosen_server_idx)

        return chosen_server_idx

    def release_server(self, idx: int, token_count,task=None):  # Changed to synchronous
        # No lock needed - atomic operation
        self.dp_servers[idx].active_tokens -= token_count
        if global_args.enable_dynamic_bucket and task is not None:
            self.bucket_load_balancer.release_task(task.id)
        # Update priority queue after releasing
        self._update_server_priority(idx)

    def calculate_request_score(self, request_length: int, max_tokens: int = 16, ignore_eos: bool = False) -> float:
        if ignore_eos:
            return request_length + max_tokens
        else:
            # Note that 0.5 is an empirical value here because we don't know
            # the actual number of tokens generated before EOS.
            return request_length + 0.5 * max_tokens

    def calculate_request_tokens(self, request_length: int) -> float:
        return request_length / 4.0

    def select_dp_group(self, req_id: str, request_tokens, priority_score) -> tuple[int, Task | None]:
        """
        Find Best group by request length and current load of groups
        """
        if global_args.enable_dynamic_bucket:
            group_idx, task = self.bucket_load_balancer.dispatch_single_task(req_id, request_tokens, priority_score)
            return group_idx, task
        else:
            return 0, None


proxy_state: Optional[ProxyState] = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--dp-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--dp-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )

    parser.add_argument("--dp-group-threshold",
                        type=int,
                        default=32 * 1024,
                        help="Threshold of dp groups")
    parser.add_argument("--max-request-tokens",
                        type=int,
                        default=128 * 1024,
                        help="Max tokens of request")
    parser.add_argument("--enable-dynamic-bucket",
                        action="store_true",
                        default=False,
                        help="Enable dynamic bucket load Balancer")

    args = parser.parse_args()
    if len(args.dp_hosts) != len(args.dp_ports):
        raise ValueError("Number of dp hosts must match number of dp ports")
    args.server_instances = list(zip(args.dp_hosts, args.dp_ports))
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(global_args.server_instances)
    print(f"Initialized {len(proxy_state.dp_servers)} dp server clients.")
    yield
    for p in proxy_state.dp_servers:
        await p.client.aclose()


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


async def _select_instance(api: str, req_data: Any, request_length: int):
    # refer to vLLM sampling_params: max_token default value
    max_tokens = req_data.get("max_tokens", 16)
    ignore_eos = req_data.get("ignore_eos", False)
    priority_score = 0
    if proxy_state.num_dp_groups > 1:
        priority_score = proxy_state.calculate_request_tokens(request_length)
    else:
        priority_score = proxy_state.calculate_request_score(request_length, max_tokens=max_tokens, ignore_eos=ignore_eos)

    logger.debug(
        f"Request length: {request_length}, max tokens: {max_tokens}, "
        f"ignore_eos: {ignore_eos}, Priority score: {priority_score}"
    )
    request_id = await proxy_state.next_req_id()
    # Select dp server based on priority score
    request_tokens = proxy_state.calculate_request_tokens(request_length)
    group_idx, task = proxy_state.select_dp_group(request_id, request_tokens, priority_score)

    logger.warning(f'Test =====selected group_idx: {group_idx}')

    server_idx = proxy_state.select_server(priority_score, group_idx)

    if global_args.enable_dynamic_bucket and task is not None:
        task.server_info = ServerInfo("DP",server_idx)

    logger.warning(f'Test =====chosen_server_idx: {server_idx}')

    chosen_server = proxy_state.dp_servers[server_idx]
    logger.debug(f"Choose server {chosen_server.url} to process request {request_id}")
    return InstanceInfo(
        request_id=request_id, server_idx=server_idx, priority_score=priority_score, server_state=chosen_server
    )


@dataclass
class InstanceInfo:
    request_id: str
    server_idx: int
    priority_score: float
    server_state: ServerState


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        instance_info = await _select_instance(api, req_data, request_length)

        async def generate_stream():
            nonlocal instance_info
            # Only one await per chunk, minimal logic in loop
            try:
                async for chunk in stream_service_response_with_retry(
                    instance_info.server_state.client,
                    api,
                    req_data,
                    request_id=instance_info.request_id,
                    max_retries=global_args.max_retries,
                    base_delay=global_args.retry_delay,
                ):
                    yield chunk
            except Exception as e:
                logger.error(
                    f"Error during streaming from server {instance_info.server_state.url}: {str(e)}, "
                    f"the aborted request is: {instance_info.request_id}."
                )

            # After streaming done, release tokens
            proxy_state.release_server(instance_info.server_idx, instance_info.priority_score)

        return StreamingResponse(generate_stream(), media_type="application/json")
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in external dp proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


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
        "dp_instances": len(proxy_state.dp_servers),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
