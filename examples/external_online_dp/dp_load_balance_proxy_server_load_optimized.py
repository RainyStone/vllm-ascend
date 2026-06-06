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
import os
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, List, Optional, Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from dynamic_bucket_load_balancer_load_optimized import DynamicBucketLoadBalancer, Task, ServerInfo
from load_collector.factory import create_load_collector
from load_collector.base import LoadUpdateConfig
from load_collector.metric_load_calculator import KvCacheAwareCalculator
from load_collector.token_estimator import create_token_estimator, TokenEstimator

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
    def __init__(self, host, port,total_kv_blocks: int, block_size: int,idx: int, max_num_seqs: int):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.idx = idx  # 在 dp_servers 中的索引
        self.realtime_load = 0.0  # 0~1 之间的归一化负载
        self.inflight_tokens = 0.0  # 请求进来时处理的token数 (即prompt的token数，一定程度上可反应prefill阶段的负载)

        self.total_kv_blocks = total_kv_blocks  # 推理后端总 KV block 数
        self.block_size = block_size  # 每个 block 的 token 数
        self.max_num_seqs = max_num_seqs  # batch_size大小
        self.latest_metrics: Dict[str, float] = {}  # 最近一次抓取的 metrics 数据
        self.calculator = None  # 绑定的负载计算器


    def __eq__(self, other):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        other_host = other.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return self_host == other_host and str(self.port) == str(other.port)

    def __hash__(self):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return hash((self_host, str(self.port)))

    def __repr__(self):
        return f"{self.host}:{self.port}"

class ProxyState:
    def __init__(
            self,
            server_instances,     # [(host, port, total_blocks, block_size, max_num_seqs), ...]
            collect_config: Optional[LoadUpdateConfig] = None,
            token_estimator: Optional[TokenEstimator] = None,
    ):
        # 修改：按 5 元组展开，传入 total_kv_blocks / block_size / idx / max_num_seqs
        self.dp_servers: list[ServerState] = [
            ServerState(h, p, total_kv_blocks=b, block_size=bs, idx=i,max_num_seqs=ms) for i, (h, p, b, bs, ms) in enumerate(server_instances)
        ]

        self.token_estimator = token_estimator or create_token_estimator("char")
        self.req_id_lock = asyncio.Lock()

        if global_args.enable_dynamic_bucket:
            self.num_dp_groups = 2 # 启用动态分桶时的分组数量
        else:
            self.num_dp_groups = 1 # 默认不分组

        # 对Server进行分组
        self.dp_groups: List[List[ServerState]] = self._group_servers(self.dp_servers, self.num_dp_groups)

        # 记录每个服务器所属的组索引（可选，用于 release_server 快速定位）
        self.server_idx_to_group_idx = {}
        for group_idx, group in enumerate(self.dp_groups):
            for server in group:
                self.server_idx_to_group_idx[server.idx] = group_idx

        logger.warning(f'Test==== self.dp_groups: {self.dp_groups}')
        logger.warning(f'Test==== global_args.enable_dynamic_bucket: {global_args.enable_dynamic_bucket}')

        # 初始化 DynamicBucketLoadBalancer
        self.bucket_load_balancer: Optional[DynamicBucketLoadBalancer] = None
        if global_args.enable_dynamic_bucket:
            self.dp_group_threshold = global_args.dp_group_threshold
            dp_buckets = [(0, self.dp_group_threshold),
                               (self.dp_group_threshold, global_args.max_request_tokens)]

            if self.num_dp_groups != len(dp_buckets):
                raise ValueError("Number of dp groups must match number of dp buckets")

            self.bucket_load_balancer = DynamicBucketLoadBalancer(buckets=dp_buckets,
                                                                  sensitivity=100.0,
                                                                  affinity_strength=1.0)

        # 初始化负载采集器
        self.collect_config = collect_config or LoadUpdateConfig()
        self.collectors = []
        for server in self.dp_servers:
            calculator = KvCacheAwareCalculator(server.total_kv_blocks, server.block_size,max_num_seqs=server.max_num_seqs)
            collector = create_load_collector("vllm", server.client, calculator)
            server.calculator = calculator
            self.collectors.append(collector)

        self._load_update_task = None

    def _update_bucket_loads_from_metrics(self):
        """从负载采集模块获取每个服务器的负载，然后聚合到桶"""
        if global_args.enable_dynamic_bucket:
            group_loads = []
            for group_idx, group in enumerate(self.dp_groups):
                # 获取该组所有服务器的负载
                loads = []
                for server in group:
                    # 这里需要从 server 获取实时负载，可以是在 ServerState 中由采集器更新的属性
                    loads.append(server.realtime_load)
                avg_load = sum(loads) / len(loads) if loads else 0.0
                group_loads.append(avg_load)

            self.bucket_load_balancer.update_bucket_loads(group_loads)

    async def update_load_once(self) -> float:
        """执行一轮负载更新，返回本次采集耗时（秒）。"""
        start = asyncio.get_event_loop().time()
        # 单个后端的采集协程
        async def update_one(idx, server):
            collector = self.collectors[idx]
            try:
                metrics = await asyncio.wait_for(collector.fetch_metrics(server.url.replace("/v1", "")), timeout=5.0)
                if metrics is not None:
                    server.latest_metrics = metrics
                    server.realtime_load = collector.load_calculator.calculate(metrics)
                else:
                    # 采集失败，设为 fallback 高负载以避免分配
                    server.latest_metrics = {}
                    server.realtime_load = self.collect_config.fallback_load * self.collect_config.scale_factor
                    logger.warning(f'Failed to fetch metrics for server: {server}, reason: fetched metrics is None')
            except asyncio.TimeoutError:
                server.latest_metrics = {}
                server.realtime_load = self.collect_config.fallback_load * self.collect_config.scale_factor
                logger.warning(f'Failed to fetch metrics for server: {server}, reason: Timeout')

        # 并行采集所有后端
        results = await asyncio.gather(
            *[update_one(i, s) for i, s in enumerate(self.dp_servers)],
            return_exceptions=True
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Load update failed for server {self.dp_servers[i]}: {result}")

        # 更新桶的负载（仅当启用动态分桶时）
        self._update_bucket_loads_from_metrics()

        elapsed = asyncio.get_event_loop().time() - start
        return elapsed

    async def start_load_updater(self):
        """启动后台负载更新任务（并行采集、请求超时、固定间隔）"""

        async def _update_loop():
            while True:
                elapsed = await self.update_load_once()
                # 精确控制间隔，扣除采集耗时
                await asyncio.sleep(max(0.0, self.collect_config.interval_seconds - elapsed))

        self._load_update_task = asyncio.create_task(_update_loop())

    async def stop_load_updater(self):
        if self._load_update_task:
            self._load_update_task.cancel()
            try:
                await self._load_update_task
            except asyncio.CancelledError:
                pass

    @staticmethod
    def _group_servers(servers: List[ServerState], num_groups: int):
        """ Group servers into num_groups groups. """
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

    def next_req_id(self):
        return str(uuid.uuid4())

    def select_server(self,group_idx,estimated_tokens):
        if not self.dp_servers:
            raise RuntimeError("No decoder servers available")

        group:List[ServerState] = self.dp_groups[group_idx]
        if not group:
            raise RuntimeError(f"No servers in group {group_idx}")

        # 新增：请求感知的评分函数
        def server_score(s: ServerState):
            # if s.calculator and s.latest_metrics and estimated_tokens > 0:
            #     req_load = s.calculator.calculate(s.latest_metrics, estimated_tokens)
            # else:
            #     req_load = s.realtime_load
            req_load = s.realtime_load
            # 按 inflight_tokens 惩罚，系数 1e-6 可调 TODO 惩罚系数根据 total_block、block_size计算每个token的比例？
            return req_load + s.inflight_tokens * 1e-6

        chosen_server = min(group, key=server_score)
        chosen_server.inflight_tokens += estimated_tokens
        return chosen_server.idx

    def release_server(self, idx: int,  release_tokens: float, req_id):  # Changed to synchronous
        # No lock needed - atomic operation
        server = self.dp_servers[idx]
        server.inflight_tokens = max(0.0, server.inflight_tokens - release_tokens)

        if global_args.enable_dynamic_bucket and req_id is not None:
            self.bucket_load_balancer.release_task(req_id)

    def estimate_input_tokens(self, req_data: dict) -> float:
        """使用配置的估算策略计算输入 token 数"""
        return self.token_estimator.estimate(req_data)

    def calculate_request_score(self, estimated_tokens: float, max_tokens: int = 16, ignore_eos: bool = False) -> float:
        if ignore_eos:
            return estimated_tokens + max_tokens
        else:
            # Note that 0.5 is an empirical value here because we don't know
            # the actual number of tokens generated before EOS.
            return estimated_tokens + 0.5 * max_tokens

    def calculate_request_tokens(self, estimated_tokens: float) -> float:
        return estimated_tokens

    def select_dp_group(self, req_id: str, request_tokens, priority_score) -> tuple[int, Task | None]:
        """
        Find Best group by request length and current load of groups
        """
        if global_args.enable_dynamic_bucket:
            group_idx, task = self.bucket_load_balancer.dispatch_task(Task(req_id, request_tokens, priority_score))
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

    # 新增：KV cache 容量配置
    parser.add_argument("--dp-total-blocks",
                        type=int,
                        nargs="+",
                        required=True,
                        help="Total KV cache blocks for each DP backend. Must match --dp-hosts count.")
    parser.add_argument("--dp-block-size",
                        type=int,
                        nargs="+",
                        required=True,
                        help="KV cache block size (tokens per block) for each DP backend. ")
    parser.add_argument("--max-num-seqs",
                        type=int,
                        nargs="+",
                        default=None,
                        help="vLLM max_num_seqs for each DP backend, used for compute pressure normalization. "
                             "Must match --dp-hosts count, or provide exactly one value for all.")

    args = parser.parse_args()
    n=len(args.dp_hosts)
    if len(args.dp_ports) != n:
        raise ValueError("Number of dp hosts must match number of dp ports")

    # 新增：校验并补全 dp-total-blocks / dp-block-size
    if len(args.dp_total_blocks) != n:
        raise ValueError("--dp-total-blocks count must match --dp-hosts count")

    if len(args.dp_block_size) == 1:
        args.dp_block_size = args.dp_block_size * n
    elif len(args.dp_block_size) != n:
        raise ValueError("--dp-block-size count must match --dp-hosts count, or provide exactly one value")

    if len(args.max_num_seqs) == 1:
        args.max_num_seqs = args.max_num_seqs * n
    elif len(args.max_num_seqs) != n:
        raise ValueError("--max-num-seqs count must match --dp-hosts count, or provide exactly one value")

    args.server_instances = [
        (h, p, b, bs, ms) for h, p, b, bs, ms in zip(args.dp_hosts, args.dp_ports, args.dp_total_blocks, args.dp_block_size,args.max_num_seqs)
    ]

    logger.warning(f"Test===========args.server_instances: {args.server_instances}")

    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(global_args.server_instances, LoadUpdateConfig(interval_seconds=1.0))
    print(f"Initialized {len(proxy_state.dp_servers)} dp server clients.")
    await proxy_state.start_load_updater()
    yield
    await proxy_state.stop_load_updater()
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


async def _select_instance(api: str, req_data: Any, estimated_tokens: float):
    # refer to vLLM sampling_params: max_token default value
    max_tokens = req_data.get("max_tokens", 16)
    ignore_eos = req_data.get("ignore_eos", False)
    priority_score = 0
    if proxy_state.num_dp_groups > 1:
        priority_score = proxy_state.calculate_request_tokens(estimated_tokens)
    else:
        priority_score = proxy_state.calculate_request_score(estimated_tokens, max_tokens=max_tokens,ignore_eos=ignore_eos)

    logger.debug(
        f"Estimated tokens: {estimated_tokens}, max tokens: {max_tokens}, "
        f"ignore_eos: {ignore_eos}, Priority score: {priority_score}"
    )
    request_id = proxy_state.next_req_id()
    # Select dp server based on priority score
    request_tokens = proxy_state.calculate_request_tokens(estimated_tokens)
    group_idx, task = proxy_state.select_dp_group(request_id, request_tokens, priority_score)

    logger.warning(f'Test =====selected group_idx: {group_idx}')

    server_idx = proxy_state.select_server(group_idx,estimated_tokens)

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
        estimated_tokens = proxy_state.estimate_input_tokens(req_data)
        instance_info = await _select_instance(api, req_data, estimated_tokens)

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
            finally:
                # After streaming done, release tokens
                proxy_state.release_server(instance_info.server_idx,estimated_tokens,instance_info.request_id)

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
