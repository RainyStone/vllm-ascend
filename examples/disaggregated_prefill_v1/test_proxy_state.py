import math
import unittest
import load_balance_proxy_server_example
from load_balance_proxy_server_example import ProxyState, parse_args, ServerHeapItem


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_static_buckets(self):
        args_list = [
            '--port', '6999',
            "--host", "localhost",
            "--prefiller-hosts", "127.0.0.1", "127.0.0.2", "127.0.0.3",
            "--prefiller-ports", "8001", "8002", "8003",
            "--decoder-hosts", "128.0.0.1", "128.0.0.2",
            "--decoder-ports", "9001", "9002",

            "--num-prefill-groups", "2"

        ]
        load_balance_proxy_server_example.global_args = parse_args(args_list)
        args = load_balance_proxy_server_example.global_args
        proxy_state = ProxyState(args.prefiller_instances, args.decoder_instances)

        # 添加请求 test_req_id_0
        req_id = "test_req_id_0"
        request_length = 8000
        prefiller_score = proxy_state.calculate_prefill_tokens(request_length)

        request_tokens = request_length / 4.0
        group_idx, task = proxy_state.select_prefill_group(req_id,request_tokens,math.ceil(prefiller_score))
        self.assertEqual(0, group_idx)
        self.assertIsNone(task)


        chosen_prefiller_idx = proxy_state.select_prefiller(prefiller_score, group_idx)
        self.assertEqual(0, chosen_prefiller_idx)
        self.assertEqual(prefiller_score,proxy_state.prefillers[chosen_prefiller_idx].active_tokens)
        self.assertEqual(prefiller_score, proxy_state.prefillers[chosen_prefiller_idx].active_kv_cache)
        self.assertEqual(prefiller_score, proxy_state.prefill_group_load[group_idx])

        priority = proxy_state.prefillers[chosen_prefiller_idx].active_tokens + proxy_state.prefillers[chosen_prefiller_idx].active_kv_cache * 0.3
        for server_item in proxy_state.prefiller_heaps[group_idx]:
            if server_item.server_idx == chosen_prefiller_idx:
                self.assertEqual(priority,server_item.priority)

        # 添加请求 test_req_id_1
        req_id = "test_req_id_1"
        request_length = 6000
        prefiller_score = proxy_state.calculate_prefill_tokens(request_length)

        request_tokens = request_length / 4.0
        group_idx, task = proxy_state.select_prefill_group(req_id, request_tokens, math.ceil(prefiller_score))
        self.assertEqual(1, group_idx)
        self.assertIsNone(task)

        chosen_prefiller_idx = proxy_state.select_prefiller(prefiller_score, group_idx)
        self.assertEqual(2, chosen_prefiller_idx)
        self.assertEqual(prefiller_score, proxy_state.prefillers[chosen_prefiller_idx].active_tokens)
        self.assertEqual(prefiller_score, proxy_state.prefillers[chosen_prefiller_idx].active_kv_cache)
        self.assertEqual(prefiller_score, proxy_state.prefill_group_load[group_idx])

        priority = proxy_state.prefillers[chosen_prefiller_idx].active_tokens + proxy_state.prefillers[chosen_prefiller_idx].active_kv_cache * 0.3
        for server_item in proxy_state.prefiller_heaps[group_idx]:
            if server_item.server_idx == chosen_prefiller_idx:
                self.assertEqual(priority, server_item.priority)

        # 添加请求 test_req_id_2
        req_id_2 = "test_req_id_2"
        request_length_2 = 6000
        prefiller_score_2 = proxy_state.calculate_prefill_tokens(request_length_2)

        request_tokens_2 = request_length_2 / 4.0
        group_idx_2, task_2 = proxy_state.select_prefill_group(req_id_2, request_tokens_2, math.ceil(prefiller_score_2))
        self.assertEqual(0, group_idx_2)
        self.assertIsNone(task_2)

        pre_active_tokens = proxy_state.prefillers[1].active_tokens
        pre_active_kv_cache = proxy_state.prefillers[1].active_kv_cache
        pre_prefill_group_load = proxy_state.prefill_group_load[group_idx_2]

        chosen_prefiller_idx_2 = proxy_state.select_prefiller(prefiller_score_2, group_idx_2)

        self.assertEqual(1, chosen_prefiller_idx_2)
        self.assertEqual(prefiller_score_2 + pre_active_tokens,proxy_state.prefillers[chosen_prefiller_idx_2].active_tokens)
        self.assertEqual(prefiller_score_2 + pre_active_kv_cache,proxy_state.prefillers[chosen_prefiller_idx_2].active_kv_cache)
        self.assertEqual(prefiller_score_2 + pre_prefill_group_load, proxy_state.prefill_group_load[group_idx_2])

        priority = proxy_state.prefillers[chosen_prefiller_idx_2].active_tokens + proxy_state.prefillers[
            chosen_prefiller_idx_2].active_kv_cache * 0.3
        for server_item in proxy_state.prefiller_heaps[group_idx_2]:
            if server_item.server_idx == chosen_prefiller_idx_2:
                self.assertEqual(priority, server_item.priority)

        # 添加请求 test_req_id_3
        req_id_3 = "test_req_id_3"
        request_length_3 = 6000
        prefiller_score_3 = proxy_state.calculate_prefill_tokens(request_length_3)

        request_tokens_3 = request_length_3 / 4.0
        group_idx_3, task_3 = proxy_state.select_prefill_group(req_id_3, request_tokens_3, math.ceil(prefiller_score_3))
        self.assertEqual(1, group_idx_3)
        self.assertIsNone(task_3)

        pre_active_tokens = proxy_state.prefillers[1].active_tokens
        pre_active_kv_cache = proxy_state.prefillers[1].active_kv_cache
        pre_prefill_group_load = proxy_state.prefill_group_load[group_idx_3]

        chosen_prefiller_idx_3 = proxy_state.select_prefiller(prefiller_score_3, group_idx_3)

        self.assertEqual(2, chosen_prefiller_idx_3)
        self.assertEqual(prefiller_score_3 + pre_active_tokens,
                         proxy_state.prefillers[chosen_prefiller_idx_3].active_tokens)
        self.assertEqual(prefiller_score_3 + pre_active_kv_cache,
                         proxy_state.prefillers[chosen_prefiller_idx_3].active_kv_cache)
        self.assertEqual(prefiller_score_3 + pre_prefill_group_load, proxy_state.prefill_group_load[group_idx_3])

        priority = proxy_state.prefillers[chosen_prefiller_idx_3].active_tokens + proxy_state.prefillers[
            chosen_prefiller_idx_3].active_kv_cache * 0.3
        for server_item in proxy_state.prefiller_heaps[group_idx_3]:
            if server_item.server_idx == chosen_prefiller_idx_3:
                self.assertEqual(priority, server_item.priority)

        # 释放请求 "test_req_id_2" active_tokens
        pre_active_tokens = proxy_state.prefillers[chosen_prefiller_idx_2].active_tokens
        pre_active_kv_cache = proxy_state.prefillers[chosen_prefiller_idx_2].active_kv_cache
        pre_prefill_group_load = proxy_state.prefill_group_load[group_idx_2]

        proxy_state.release_prefiller(chosen_prefiller_idx_2, prefiller_score_2, task_2)
        self.assertEqual(pre_active_tokens-prefiller_score_2, proxy_state.prefillers[chosen_prefiller_idx_2].active_tokens)
        self.assertEqual(0,
                         proxy_state.prefillers[chosen_prefiller_idx_2].active_tokens)
        # TODO 注意，prefill_group_load 这里暂没有考虑 active_kv_cache
        self.assertEqual(pre_prefill_group_load - prefiller_score_2,proxy_state.prefill_group_load[group_idx_2])
        self.assertEqual(pre_active_kv_cache, proxy_state.prefillers[chosen_prefiller_idx_2].active_kv_cache)
        self.assertNotEqual(0, proxy_state.prefillers[chosen_prefiller_idx_2].active_kv_cache)

        priority = proxy_state.prefillers[chosen_prefiller_idx_2].active_kv_cache * 0.3
        for server_item in proxy_state.prefiller_heaps[group_idx_2]:
            if server_item.server_idx == chosen_prefiller_idx_2:
                self.assertEqual(priority, server_item.priority)

        # 释放请求 "test_req_id_2" active_kv_cache
        proxy_state.release_prefiller_kv(chosen_prefiller_idx_2, prefiller_score_2)
        self.assertEqual(0,proxy_state.prefillers[chosen_prefiller_idx_2].active_tokens)
        self.assertEqual(0, proxy_state.prefillers[chosen_prefiller_idx_2].active_kv_cache)

        for server_item in proxy_state.prefiller_heaps[group_idx_2]:
            if server_item.server_idx == chosen_prefiller_idx_2:
                self.assertEqual(0, server_item.priority)

    def test_add_prefillers(self):
        args_list = [
            '--port', '6999',
            "--host", "localhost",
            "--prefiller-hosts", "127.0.0.1", "127.0.0.2", "127.0.0.3",
            "--prefiller-ports", "8001", "8002", "8003",
            "--decoder-hosts", "128.0.0.1", "128.0.0.2",
            "--decoder-ports", "9001", "9002",

            "--num-prefill-groups", "2"

        ]
        load_balance_proxy_server_example.global_args = parse_args(args_list)
        args = load_balance_proxy_server_example.global_args
        proxy_state = ProxyState(args.prefiller_instances, args.decoder_instances)





if __name__ == '__main__':
    unittest.main()
