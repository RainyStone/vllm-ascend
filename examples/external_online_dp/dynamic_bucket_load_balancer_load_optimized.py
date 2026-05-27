import math
from collections import namedtuple
from typing import List, Optional, AnyStr

ServerInfo = namedtuple('ServerInfo', ['instance_type', 'instance_idx'])

class Task:
    """模拟任务(仅包含长度信息)"""

    def __init__(self, task_id: AnyStr, task_length: int, task_load: float):
        self.id = task_id
        self.length = task_length
        self.bucket_idx = -1
        self.load = task_load
        self.server_info: ServerInfo = ServerInfo("Unknown", -1)

    def __repr__(self):
        return f"Task(id={self.id}, length={self.length}, load={self.load},instance_type={self.server_info.instance_type}, instance_idx={self.server_info.instance_idx})"

class Bucket:
    """桶：现在只保留长度范围，不再维护动态负载"""
    def __init__(self, bucket_ranges):
        self.min_length = bucket_ranges[0]
        self.max_length = bucket_ranges[1]

class DynamicBucketLoadBalancer:
    def __init__(self, buckets, sensitivity=100.0, affinity_strength=1.0, log_func=print, all_neighbor=False):
        self.num_buckets = len(buckets)
        self.buckets = {idx: Bucket(r) for idx, r in enumerate(buckets)}
        self.sensitivity = sensitivity
        self.affinity_strength = affinity_strength
        self.log_func = log_func
        self.all_neighbor = all_neighbor

        # 外部实时负载（由代理定期更新）
        self.current_bucket_loads: List[float] = [0.0] * self.num_buckets

        self.base_probability_threshold = 1 / (self.num_buckets * 2.0)
        self._log_info(f"Load Balance base_probability_threshold: {self.base_probability_threshold:.2f}")

        # 任务追踪（仅用于释放时的清理，但释放不再需要调整负载，所以可以移除）
        self.tasks = {}   # 保留仅用于调试，可删除

    def _log_info(self, msg, *args, **kwargs):
        if self.log_func:
            self.log_func(msg, *args, **kwargs)

    def update_bucket_loads(self, loads: List[float]):
        """由代理定期调用，传入每个桶的最新负载（归一化值，例如平均 kv_cache_usage）"""
        if len(loads) != self.num_buckets:
            raise ValueError("Length of loads must equal number of buckets")
        self.current_bucket_loads = loads[:]

    def _get_standard_bucket_index(self, task_length):
        for idx, bucket in self.buckets.items():
            if bucket.min_length <= task_length < bucket.max_length:
                return idx
        return self.num_buckets - 1

    def _get_neighbor_indices(self, bucket_idx):
        if self.all_neighbor:
            return list(range(self.num_buckets))
        neighbors = []
        if bucket_idx > 0:
            neighbors.append(bucket_idx - 1)
        if bucket_idx < self.num_buckets - 1:
            neighbors.append(bucket_idx + 1)
        return neighbors

    def _calculate_length_affinity(self, task_length, target_bucket_idx):
        bucket = self.buckets[target_bucket_idx]
        bucket_min = bucket.min_length
        bucket_max = bucket.max_length
        bucket_center = (bucket_min + bucket_max) / 2.0
        bucket_half_width = (bucket_max - bucket_min) / 2.0
        if bucket_half_width <= 0:
            return 1.0
        distance = abs(task_length - bucket_center)
        normalized_distance = distance / bucket_half_width
        affinity = math.exp(-self.affinity_strength * normalized_distance)
        return max(0.0, min(affinity, 1.0))

    def _calculate_redirect_probability(self, task_length, standard_idx, neighbor_idx):
        # 使用外部实时负载
        standard_load = self.current_bucket_loads[standard_idx]
        neighbor_load = self.current_bucket_loads[neighbor_idx]

        # 负载差距概率
        if standard_load <= 0 or neighbor_load <= 0:
            load_probability = 0.0
        else:
            load_ratio = standard_load / neighbor_load
            if load_ratio <= 1.0:
                load_probability = 0.0
            else:
                raw = math.log(load_ratio)
                load_probability = 1 - math.exp(-self.sensitivity * raw)
                load_probability = max(0.0, min(load_probability, 1.0))

        # 长度亲和因子
        affinity = self._calculate_length_affinity(task_length, neighbor_idx)

        return load_probability * affinity

    def dispatch_task(self, task):
        """分配任务，返回 (bucket_idx, task)"""
        standard_idx = self._get_standard_bucket_index(task.length)
        neighbors = self._get_neighbor_indices(standard_idx)

        best_neighbor = None
        best_prob = 0.0
        for nb in neighbors:
            # 只有邻居负载更低时才考虑重定向
            if self.current_bucket_loads[nb] < self.current_bucket_loads[standard_idx]:
                prob = self._calculate_redirect_probability(task.length, standard_idx, nb)
                if prob > best_prob:
                    best_prob = prob
                    best_neighbor = nb

        final_idx = standard_idx
        if best_neighbor is not None and best_prob > self.base_probability_threshold:
            final_idx = best_neighbor
            self._log_info(f"Task {task.id} redirected from {standard_idx} to {final_idx} (prob={best_prob:.3f})")

        task.bucket_idx = final_idx
        # 不再更新桶负载，只记录任务
        self.tasks[task.id] = task
        return final_idx, task

    def release_task(self, task_id):
        # 桶负载不由任务释放驱动，仅清理记录
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False