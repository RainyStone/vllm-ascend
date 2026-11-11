# SPDX-License-Identifier: Apache-2.0
"""CPU checks for omitted planning histograms and real-token accounting."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

SOURCE=Path(__file__).resolve().parents[3]/"vllm_ascend/ops/fused_moe/moonep_comm.py"
tree=ast.parse(SOURCE.read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_route_histogram")
namespace={"torch":torch}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),"exec"),namespace)
histogram=namespace["_route_histogram"]

class HistogramTests(unittest.TestCase):
    def setUp(self):
        self.routes=torch.tensor([[0,1],[1,2],[3,3],[3,3]],dtype=torch.int32)
        self.compat=torch.zeros(4,dtype=torch.int32)
    def test_disabled_does_no_histogram_work(self):
        with patch.object(torch,"histc",side_effect=AssertionError("unneeded kernel")):
            result=histogram(self.routes,4,2,self.compat,False)
        self.assertIs(result,self.compat)
        self.assertTrue(torch.equal(self.compat,torch.zeros_like(self.compat)))
    def test_disabled_reuses_same_buffer(self):
        for valid in (1,2,4):
            result=histogram(self.routes,4,valid,self.compat,False)
            self.assertEqual(result.data_ptr(),self.compat.data_ptr())
    def test_diagnostics_exclude_capacity_tail(self):
        result=histogram(self.routes,4,2,self.compat,True)
        self.assertEqual(result.tolist(),[1,2,1,0])
    def test_invalid_tail_experts_are_not_counted(self):
        self.routes[2:].fill_(999)
        result=histogram(self.routes,4,2,self.compat,True)
        self.assertEqual(result.tolist(),[1,2,1,0])
    def test_full_capacity_counts_every_route(self):
        result=histogram(self.routes,4,4,self.compat,True)
        self.assertEqual(result.sum().item(),8)
    def test_zero_tokens(self):
        result=histogram(self.routes,4,0,self.compat,True)
        self.assertEqual(result.tolist(),[0,0,0,0])

if __name__=="__main__":unittest.main()
