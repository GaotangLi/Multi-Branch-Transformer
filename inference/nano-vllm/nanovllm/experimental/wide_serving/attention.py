# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Paged KV attention for uneven GQA shards and independent branch caches.

FA4 or FA3 (sgl_kernel) supplies the production CUDA attention; an intentionally slow FP32 SDPA
reference is available for correctness only. No quadratic reference is silently
substituted into a timed CUDA run. Cache positions are absolute token positions.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BatchContext:
    is_prefill: bool
    cu_seqlens_q: torch.Tensor
    seqused_k: torch.Tensor
    page_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


def cache_bytes(specs, batch_size, capacity_tokens, block_size=256, element_size=2):
    if batch_size < 1 or capacity_tokens < 1 or block_size < 1:
        raise ValueError('Positive batch, capacity, and block size required')
    pages = batch_size * math.ceil(capacity_tokens / block_size)
    return 2 * pages * block_size * element_size * sum(
        s['num_kv_heads'] * s['head_dim'] for s in specs)


class PagedKVCache:
    def __init__(self, specs, batch_size, capacity_tokens, *, device='cuda',
                 dtype=torch.bfloat16, block_size=256, memory_fraction=.95):
        self.device = torch.device(device)
        self.batch_size, self.capacity_tokens = batch_size, capacity_tokens
        self.block_size = block_size
        self.pages_per_seq = math.ceil(capacity_tokens / block_size)
        self.num_pages = batch_size * self.pages_per_seq
        self.specs = [dict(s) for s in specs]
        if len({s['layer_id'] for s in specs}) != len(specs):
            raise ValueError('Each physical attention layer needs its own cache')
        self.nbytes = cache_bytes(specs,batch_size,capacity_tokens,block_size,torch.empty((),dtype=dtype).element_size())
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(self.device)
            if self.nbytes > free * memory_fraction:
                raise MemoryError(f'KV cache needs {self.nbytes/2**30:.2f}GiB, free {free/2**30:.2f}GiB')
        self.page_table = torch.arange(self.num_pages,device=self.device,dtype=torch.int32).view(batch_size,-1)
        self.layers = {}
        for s in specs:
            shape = (self.num_pages,block_size,s['num_kv_heads'],s['head_dim'])
            # Zero-filled: paged kernels load whole KV tiles, and 0*NaN from uninitialized slots past
            # seqused_k would poison the output even though those slots get softmax weight 0.
            self.layers[s['layer_id']] = (torch.zeros(shape,device=self.device,dtype=dtype),
                                          torch.zeros(shape,device=self.device,dtype=dtype))

    def context(self, start, length, *, prefill):
        if start < 0 or length < 1 or start+length > self.capacity_tokens:
            raise ValueError('Query exceeds the allocated KV capacity')
        b, device = self.batch_size, self.device
        positions = torch.arange(start,start+length,device=device,dtype=torch.long).repeat(b)
        offsets = torch.arange(b,device=device,dtype=torch.long).repeat_interleave(length)
        slots = offsets*(self.pages_per_seq*self.block_size)+positions
        cu = torch.arange(b+1,device=device,dtype=torch.int32)*length
        used = torch.full((b,),start+length,device=device,dtype=torch.int32)
        context = BatchContext(prefill,cu,used,self.page_table,slots,length,
                               start+length if prefill else self.capacity_tokens)
        return positions, context


_store_kernel = None


def _get_store_kernel():
    global _store_kernel
    if _store_kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def store(key, value, kc, vc, slots, stride_k:tl.constexpr,
                  stride_v:tl.constexpr, width:tl.constexpr, block:tl.constexpr):
            row=tl.program_id(0).to(tl.int64)
            slot=tl.load(slots+row).to(tl.int64)
            col=tl.arange(0,block)
            mask=(col<width)&(slot>=0)
            k=tl.load(key+row*stride_k+col,mask=mask,other=0)
            v=tl.load(value+row*stride_v+col,mask=mask,other=0)
            tl.store(kc+slot*width+col,k,mask=mask)
            tl.store(vc+slot*width+col,v,mask=mask)
        _store_kernel=store
    return _store_kernel


def store_kv(k,v,kc,vc,slots):
    if k.shape != v.shape or k.shape[0] != slots.numel():
        raise ValueError('KV input and slot shape mismatch')
    if k.stride(-1)!=1 or k.stride(-2)!=k.shape[-1] or v.stride(-1)!=1 or v.stride(-2)!=v.shape[-1]:
        k,v=k.contiguous(),v.contiguous()
    if not k.is_cuda:
        live=slots>=0
        kc.view(-1,*k.shape[1:])[slots[live]]=k[live]
        vc.view(-1,*v.shape[1:])[slots[live]]=v[live]
        return
    import triton
    width=k.shape[1]*k.shape[2]
    _get_store_kernel()[(k.shape[0],)](k,v,kc,vc,slots,k.stride(0),v.stride(0),width,
                                      triton.next_power_of_2(width))


class PagedAttentionBackend:
    def __init__(self, cache, *, implementation='fa4', decode_splits=0):
        if implementation not in ('fa4','fa3','reference'):
            raise ValueError('Unknown attention backend')
        self.cache=cache
        self.implementation=implementation
        self.decode_splits=decode_splits
        self._flash=None
        if implementation=='fa3':
            self._sms=torch.cuda.get_device_properties(cache.device).multi_processor_count

    def __call__(self,q,k,v,layer_id,context):
        self.store(k,v,layer_id,context)
        return self.attend(q,layer_id,context)

    def store(self,k,v,cache_id,context):
        kc,vc=self.cache.layers[cache_id]
        store_kv(k,v,kc,vc,context.slot_mapping)

    def attend(self,q,cache_id,context):
        """Attention over an already written cache; q may have any multiple of its KV heads (shared banks)."""
        kc,vc=self.cache.layers[cache_id]
        if self.implementation=='fa3':
            return self._fa3(q,kc,vc,context)
        if self.implementation=='reference':
            return self.reference(q,kc,vc,context)
        if not q.is_cuda:
            raise RuntimeError('FA4 backend requires CUDA; reference must be explicitly selected')
        if self._flash is None:
            from flash_attn.cute import flash_attn_varlen_func
            self._flash=flash_attn_varlen_func
        result=self._flash(q,kc,vc,cu_seqlens_q=context.cu_seqlens_q,
            seqused_k=context.seqused_k,page_table=context.page_table,
            max_seqlen_q=context.max_seqlen_q,max_seqlen_k=context.max_seqlen_k,
            softmax_scale=q.shape[-1]**-.5,causal=True,
            num_splits=1 if context.is_prefill else self.decode_splits)
        return result[0] if isinstance(result,tuple) else result

    def _fa3(self,q,kc,vc,context):
        # Decode splits depend only on (sequences, KV heads, capacity), never on Q heads, so a call with
        # folded branch heads splits exactly like the per-layer call.
        if self._flash is None:
            from sgl_kernel.flash_attn import flash_attn_with_kvcache
            self._flash=flash_attn_with_kvcache
        b=context.seqused_k.numel()
        splits=1 if context.is_prefill else max(1,min(math.ceil(self.cache.capacity_tokens/128),128,
                                                       math.ceil(2*self._sms/(b*kc.shape[2]))))
        out=self._flash(q.view(b,-1,*q.shape[1:]),kc,vc,cache_seqlens=context.seqused_k,page_table=context.page_table,
                        softmax_scale=q.shape[-1]**-.5,causal=True,num_splits=splits)
        return (out[0] if isinstance(out,tuple) else out).reshape(q.shape)

    @staticmethod
    def reference(q,kc,vc,context):
        outputs=[]
        cu=context.cu_seqlens_q.cpu().tolist()
        lengths=context.seqused_k.cpu().tolist()
        for i,n in enumerate(lengths):
            qi=q[cu[i]:cu[i+1]]
            ki=kc[context.page_table[i].long()].flatten(0,1)[:n]
            vi=vc[context.page_table[i].long()].flatten(0,1)[:n]
            repeat=q.shape[1]//ki.shape[1]
            ki=ki.repeat_interleave(repeat,dim=1)
            vi=vi.repeat_interleave(repeat,dim=1)
            query_positions=torch.arange(n-len(qi),n,device=q.device)
            mask=torch.arange(n,device=q.device)[None,:]<=query_positions[:,None]
            out=F.scaled_dot_product_attention(qi.transpose(0,1).float(),ki.transpose(0,1).float(),
                vi.transpose(0,1).float(),attn_mask=mask,dropout_p=0)
            outputs.append(out.transpose(0,1).to(q.dtype))
        return torch.cat(outputs,dim=0)
