# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
import unittest
import torch
from nanovllm.experimental.wide_serving.attention import PagedKVCache,PagedAttentionBackend,cache_bytes,store_kv


class CacheTests(unittest.TestCase):
    def test_cache_memory(self):
        specs=[dict(layer_id=i,num_kv_heads=11,head_dim=64) for i in range(18)]
        self.assertEqual(cache_bytes(specs,1,1048576),int(49.5*2**30))

    def test_disjoint_sequences_and_layers(self):
        specs=[dict(layer_id=i,num_kv_heads=11,head_dim=64) for i in (0,4)]
        cache=PagedKVCache(specs,2,512,device='cpu',dtype=torch.float32)
        _,c=cache.context(255,2,prefill=True)
        self.assertEqual(c.slot_mapping.tolist(),[255,256,767,768])
        k=torch.randn(4,11,64);v=torch.randn_like(k)
        store_kv(k,v,*cache.layers[0],c.slot_mapping)
        torch.testing.assert_close(cache.layers[0][0].flatten(0,1)[c.slot_mapping],k)
        self.assertNotEqual(cache.layers[0][0].data_ptr(),cache.layers[4][0].data_ptr())

    def test_chunked_causal_equivalence(self):
        torch.manual_seed(1)
        specs=[dict(layer_id=0,num_kv_heads=3,head_dim=8)]
        def make():
            c=PagedKVCache(specs,2,16,device='cpu',dtype=torch.float32,block_size=4)
            return c,PagedAttentionBackend(c,implementation='reference')
        q=torch.randn(2,9,6,8);k=torch.randn(2,9,3,8);v=torch.randn_like(k)
        full,b=make();_,ctx=full.context(0,9,prefill=True)
        expected=b(q.flatten(0,1),k.flatten(0,1),v.flatten(0,1),0,ctx).view(2,9,6,8)
        chunk,b=make();out=[]
        for start,end in ((0,4),(4,8),(8,9)):
            _,ctx=chunk.context(start,end-start,prefill=end-start>1)
            out.append(b(q[:,start:end].flatten(0,1),k[:,start:end].flatten(0,1),
                v[:,start:end].flatten(0,1),0,ctx).view(2,end-start,6,8))
        torch.testing.assert_close(torch.cat(out,dim=1),expected)

    def test_bounds(self):
        c=PagedKVCache([dict(layer_id=0,num_kv_heads=1,head_dim=4)],1,8,device='cpu')
        with self.assertRaises(ValueError):c.context(8,1,prefill=False)


if __name__=='__main__':unittest.main()
