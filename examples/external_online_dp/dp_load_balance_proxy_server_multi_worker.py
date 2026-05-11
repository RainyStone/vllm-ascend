import argparse
import asyncio
import base64
import functools
import heapq
import json
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# 引入动态桶负载均衡器
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
class ServerSnapshot:
    """Worker 端用于同步的服务器信息（不可变）。"""
    host: str
    port: int
    index: int               # 在调度器内部列表中的索引

    @property
    def key(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class InstanceInfo:
    """一次请求选择出的服务器信息。"""
    request_id: str
    server_idx: int
    priority_score: float
    server_key: str          # host:port，用于查找本地客户端
    task_id: Optional[str] = None   # 动态桶产生的任务 ID


# ------------------------------------------------------------
# 2. 共享调度器（管理进程中的唯一实例）
# ------------------------------------------------------------
class SharedProxyScheduler:
    """跨 worker 共享的调度状态，所有方法均受 RLock 保护。"""

    def __init__(self, server_instances: List[Tuple[str, int]],
                 enable_dynamic_bucket: bool = False,
                 dp_group_threshold: int = 32 * 1024,
                 max_request_tokens: int = 128 * 1024):
        self._lock = threading.RLock()

        # 服务器基本信息列表（不可变顺序）
        self.servers: List[Tuple[str, int]] = list(server_instances)
        self.num_servers = len(self.servers)

        # 每台服务器的负载
        self.active_tokens: List[float] = [0.0] * self.num_servers

        # 分组逻辑
        self.enable_dynamic_bucket = enable_dynamic_bucket
        self.num_dp_groups = 2 if enable_dynamic_bucket else 1

        if self.num_dp_groups > 1 and self.num_servers < self.num_dp_groups:
            raise ValueError("Number of servers must be >= number of groups")

        # 初始化堆：每个分组一个最小堆
        self.dp_heaps: List[List[Tuple[float, int]]] = []  # (priority, server_idx)
        self.server_idx_to_group: Dict[int, int] = {}
        self._groups = self._build_groups(self.num_servers, self.num_dp_groups)

        for group_id, idx_list in enumerate(self._groups):
            heap = [(0.0, idx) for idx in idx_list]
            heapq.heapify(heap)
            self.dp_heaps.append(heap)
            for idx in idx_list:
                self.server_idx_to_group[idx] = group_id

        # 动态桶负载均衡器
        self.bucket_load_balancer: Optional[DynamicBucketLoadBalancer] = None
        if enable_dynamic_bucket:
            buckets = [(0, dp_group_threshold), (dp_group_threshold, max_request_tokens)]
            self.bucket_load_balancer = DynamicBucketLoadBalancer(
                buckets=buckets,
                affinity_strength=1.0
            )
            # 内部任务缓存，用于跨方法关联 task_id
            self._task_cache: Dict[str, Task] = {}
            self._task_server_map: Dict[str, int] = {}  # task_id -> server_idx

        logger.info(f"Shared scheduler initialized with {self.num_servers} servers, "
                       f"groups={self.num_dp_groups}, dynamic_bucket={enable_dynamic_bucket}")

    @staticmethod
    def _build_groups(n: int, num_groups: int) -> List[List[int]]:
        """将 n 个服务器索引均匀分成 num_groups 组。"""
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
    def _update_priority(self, server_idx: int) -> None:
        """重新计算并更新服务器在堆中的优先级（需持有锁）。"""
        group = self.server_idx_to_group[server_idx]
        priority = self.active_tokens[server_idx]
        heap = self.dp_heaps[group]
        # 移除旧条目并添加新条目
        heap[:] = [(p, idx) for p, idx in heap if idx != server_idx]
        heapq.heapify(heap)
        heapq.heappush(heap, (priority, server_idx))

    # ---- 对外 API ----
    def next_req_id(self) -> str:
        with self._lock:
            return str(uuid.uuid4())

    def calculate_request_tokens(self, request_length: int) -> float:
        return request_length / 4.0

    def calculate_request_score(self, request_length: int, max_tokens: int = 16, ignore_eos: bool = False) -> float:
        if ignore_eos:
            return request_length + max_tokens
        return request_length + 0.5 * max_tokens

    def select_dp_group(self, req_id: str, request_tokens: float, priority_score: float) -> Tuple[int, Optional[str]]:
        """返回 group_idx 和可选的 task_id（如果启用动态桶）。"""
        with self._lock:
            if not self.enable_dynamic_bucket:
                return 0, None
            group_idx, task = self.bucket_load_balancer.dispatch_single_task(
                req_id, request_tokens, priority_score
            )
            task_id = task.id
            self._task_cache[task_id] = task
            return group_idx, task_id

    def select_server(self, group_idx: int, token_count: float, task_id: Optional[str] = None) -> int:
        """选择服务器并更新负载。返回服务器索引。"""
        with self._lock:
            heap = self.dp_heaps[group_idx]
            if not heap:
                raise RuntimeError(f"No server available in group {group_idx}")
            _, chosen_idx = heapq.heappop(heap)
            self.active_tokens[chosen_idx] += token_count
            self._update_priority(chosen_idx)

            if task_id and self.enable_dynamic_bucket:
                task = self._task_cache.get(task_id)
                if task:
                    task.server_info = f"DP-{chosen_idx}"  # 存储服务器标记
                    self._task_server_map[task_id] = chosen_idx
            return chosen_idx

    def release_server(self, idx: int, token_count: float, task_id: Optional[str] = None) -> None:
        """释放服务器负载，并可选地释放动态桶任务。"""
        with self._lock:
            self.active_tokens[idx] -= token_count
            self._update_priority(idx)
            if task_id and self.enable_dynamic_bucket:
                task = self._task_cache.pop(task_id, None)
                if task:
                    self.bucket_load_balancer.release_task(task.id)
                self._task_server_map.pop(task_id, None)

    def get_snapshot(self) -> List[ServerSnapshot]:
        """返回当前所有服务器的快照，供 worker 同步客户端。"""
        with self._lock:
            return [
                ServerSnapshot(host=host, port=port, index=i)
                for i, (host, port) in enumerate(self.servers)
            ]

    def healthcheck(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "dp_instances": self.num_servers,
            }


# ------------------------------------------------------------
# 3. 管理进程辅助
# ------------------------------------------------------------
shared_scheduler: Optional[SharedProxyScheduler] = None
# global_args: Optional[argparse.Namespace] = None


class SchedulerManager(BaseManager):
    pass


def get_shared_scheduler() -> SharedProxyScheduler:
    if shared_scheduler is None:
        raise RuntimeError("Shared scheduler not initialized")
    return shared_scheduler


SchedulerManager.register("get_scheduler", callable=get_shared_scheduler)


def serialize_args(args) -> dict:
    return {
        "port": args.port,
        "host": args.host,
        "dp_hosts": args.dp_hosts,
        "dp_ports": args.dp_ports,
        "max_retries": args.max_retries,
        "retry_delay": args.retry_delay,
        "dp_group_threshold": args.dp_group_threshold,
        "max_request_tokens": args.max_request_tokens,
        "enable_dynamic_bucket": args.enable_dynamic_bucket,
        "workers": args.workers,
    }


def deserialize_args(raw: dict) -> argparse.Namespace:
    args = argparse.Namespace(**raw)
    args.server_instances = list(zip(args.dp_hosts, args.dp_ports))
    return args


def start_shared_scheduler(args: argparse.Namespace) -> None:
    global shared_scheduler
    shared_scheduler = SharedProxyScheduler(
        server_instances=args.server_instances,
        enable_dynamic_bucket=args.enable_dynamic_bucket,
        dp_group_threshold=args.dp_group_threshold,
        max_request_tokens=args.max_request_tokens,
    )

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


def load_global_args_from_env() -> argparse.Namespace:
    raw = os.environ.get(ARGS_CONFIG_ENV)
    if raw is None:
        raise RuntimeError(f"{ARGS_CONFIG_ENV} not set")
    return deserialize_args(json.loads(raw))


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
        # 本地客户端池，key = "host:port"
        self.dp_clients: Dict[str, httpx.AsyncClient] = {}

    async def sync_clients(self) -> None:
        snapshot = self.scheduler.get_snapshot()
        current_keys = {s.key for s in snapshot}

        # 移除不再存在的客户端
        stale = [key for key in self.dp_clients if key not in current_keys]
        for key in stale:
            client = self.dp_clients.pop(key)
            await client.aclose()

        # 为新增服务器创建客户端
        for s_info in snapshot:
            if s_info.key not in self.dp_clients:
                base_url = self._build_base_url(s_info.host, s_info.port)
                self.dp_clients[s_info.key] = httpx.AsyncClient(
                    timeout=None,
                    base_url=base_url,
                    limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
                )

    @staticmethod
    def _build_base_url(host: str, port: int) -> str:
        import ipaddress
        url = f"http://{host}:{port}/v1"
        try:
            ip = ipaddress.ip_address(host)
            if isinstance(ip, ipaddress.IPv6Address):
                url = f"http://[{host}]:{port}/v1"
        except ValueError:
            pass
        return url

    def get_client(self, key: str) -> httpx.AsyncClient:
        return self.dp_clients[key]

    async def close(self) -> None:
        for client in self.dp_clients.values():
            await client.aclose()
        self.dp_clients.clear()


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
    logger.info(f"Worker {os.getpid()} initialized with "
          f"{len(runtime.dp_clients)} DP clients.")
    yield
    await runtime.close()
    runtime = None


app = FastAPI(lifespan=lifespan)


def create_app():
    load_global_args_from_env()
    return app


async def listen_for_disconnect(request: Request) -> None:
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
                logger.warning(f"Attempt {attempt} failed for {endpoint}: {e}")
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for {endpoint}.")
                raise e
        except Exception as e:
            if "first_chunk_sent" in locals() and first_chunk_sent:
                logger.error(f"Stream interrupted after response started: {e}")
                return
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt} failed for {endpoint}: {e}")
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for {endpoint}.")
                raise e


async def _select_instance(api: str, req_data: dict, request_length: int) -> InstanceInfo:
    await runtime.sync_clients()   # 确保客户端池最新
    scheduler = runtime.scheduler

    max_tokens = req_data.get("max_tokens", 16)
    ignore_eos = req_data.get("ignore_eos", False)

    # 计算优先级分数
    if runtime.args.enable_dynamic_bucket:
        priority_score = scheduler.calculate_request_tokens(request_length)
    else:
        priority_score = scheduler.calculate_request_score(
            request_length, max_tokens=max_tokens, ignore_eos=ignore_eos
        )

    request_id = scheduler.next_req_id()
    request_tokens = scheduler.calculate_request_tokens(request_length)
    group_idx, task_id = scheduler.select_dp_group(request_id, request_tokens, priority_score)
    server_idx = scheduler.select_server(group_idx, priority_score, task_id)

    # 获取服务器快照，找到对应的 host:port
    snapshot = scheduler.get_snapshot()
    server_key = snapshot[server_idx].key

    logger.debug(f"Request {request_id} dispatched to server {server_key} (group {group_idx})")
    return InstanceInfo(
        request_id=request_id,
        server_idx=server_idx,
        priority_score=priority_score,
        server_key=server_key,
        task_id=task_id,
    )


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        instance_info = await _select_instance(api, req_data, request_length)

        client = runtime.get_client(instance_info.server_key)

        async def generate_stream():
            nonlocal instance_info
            try:
                async for chunk in stream_service_response_with_retry(
                    client,
                    api,
                    req_data,
                    request_id=instance_info.request_id,
                    max_retries=runtime.args.max_retries,
                    base_delay=runtime.args.retry_delay,
                ):
                    yield chunk
            except Exception as e:
                logger.error(f"Error streaming from {instance_info.server_key}: {e}")
            finally:
                # 无论成功失败，释放服务器负载
                runtime.scheduler.release_server(
                    instance_info.server_idx,
                    instance_info.priority_score,
                    instance_info.task_id,
                )

        media_type = "application/json" if not req_data.get("stream") else "text/event-stream; charset=utf-8"
        return StreamingResponse(generate_stream(), media_type=media_type)
    except Exception:
        import traceback
        exc_info = sys.exc_info()
        logger.error(f"Error in {api} endpoint")
        logger.error("".join(traceback.format_exception(*exc_info)))
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
    return runtime.scheduler.healthcheck()


# ------------------------------------------------------------
# 6. 启动入口
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--dp-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--dp-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=0.001)
    parser.add_argument("--dp-group-threshold", type=int, default=32 * 1024)
    parser.add_argument("--max-request-tokens", type=int, default=128 * 1024)
    parser.add_argument("--enable-dynamic-bucket", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of uvicorn worker processes")
    args = parser.parse_args()
    if len(args.dp_hosts) != len(args.dp_ports):
        raise ValueError("Number of dp hosts must match number of dp ports")
    args.server_instances = list(zip(args.dp_hosts, args.dp_ports))
    return args


if __name__ == "__main__":
    args = parse_args()
    start_shared_scheduler(args)

    import uvicorn
    module_name = Path(__file__).stem
    uvicorn.run(
        f"{module_name}:create_app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        factory=True,
        app_dir=str(Path(__file__).resolve().parent),
    )