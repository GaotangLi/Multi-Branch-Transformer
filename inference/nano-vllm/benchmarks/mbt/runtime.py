# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Serving benchmark for one export (entry: run_bench.py): exact serial/batched parity check, then throughput per mode.

Both paths retain chunk128, identical model operations, eager prefill and graph
decode. The candidate reuses immutable prefill contexts, reset slot indices and
a prompt input buffer. It cannot qualify through a tolerance relaxation.
"""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time

from profiling import (compare_timelines, export_torch_trace, profile_request,
                       profile_summary, sample_statistics)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('smoke', 'throughput', 'profile'), required=True)
    p.add_argument('--export', required=True)
    p.add_argument('--data-prefix', default=None, help='Megatron token stream; or use --prompt-manifest')
    p.add_argument('--prompt-manifest', default=None, help='strict JSONL prompt manifest (see prompt_manifest.py)')
    p.add_argument('--manifest-source', default=None, help='select only manifest records with this source')
    p.add_argument('--expected-tokenizer-sha256', default=None)
    p.add_argument('--modes', default='serial,streams',
                   help='comma list starting with serial; others from streams (independent KV), batched (shared KV)')
    p.add_argument('--prefill-token-budget', type=int, default=0,
                   help='if >0: per-sequence prefill chunk = max(128, budget//C) rounded down to 128')
    p.add_argument('--sampling', choices=('greedy', 'sample'), default='greedy')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=0)
    p.add_argument('--sampling-seed', type=int, default=0)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--concurrency', type=int, choices=(1, 2, 4, 8, 16, 32, 64, 128), required=True)
    p.add_argument('--variant', choices=('baseline', 'preallocated', 'both'), default='both')
    p.add_argument('--prompt-length', type=int, default=4096)
    p.add_argument('--output-length', type=int, default=1024)
    p.add_argument('--system-only-extrapolation', action='store_true')
    p.add_argument('--prefill-chunk-size', type=int, choices=(128,), default=128)
    p.add_argument('--prompt-rounds', type=int, default=4)
    p.add_argument('--offset-tokens', type=int, default=0)
    p.add_argument('--repeats', '--rounds', dest='repeats', type=int, default=3)
    p.add_argument('--window-seconds', type=float, default=60.)
    p.add_argument('--warmup-rounds', type=int, default=2)
    p.add_argument('--profile-rounds', type=int, default=3)
    p.add_argument('--timeline-samples', type=int, default=5)
    p.add_argument('--torch-profiler-steps', type=int, default=0)
    return p


def validate_args(args):
    if args.prompt_length < 1 or args.output_length < 2 or args.prompt_rounds < 2:
        raise ValueError('positive prompt and >=2 outputs/prompt rounds required')
    if args.repeats < 3 or args.profile_rounds < 3 or args.timeline_samples < 3:
        raise ValueError('at least three throughput repeats, profile cohorts, and timeline samples required')
    if args.warmup_rounds < 1 or args.offset_tokens < 0:
        raise ValueError('positive warmup and nonnegative data offset required')
    if not math.isfinite(args.window_seconds) or args.window_seconds < 60:
        raise ValueError('throughput windows must be at least 60 finite seconds')
    if not 0 <= args.torch_profiler_steps <= min(32, args.output_length-1):
        raise ValueError('optional torch profiler must be bounded to <=32 decode steps')
    if (args.data_prefix is None) == (args.prompt_manifest is None):
        raise ValueError('exactly one of --data-prefix or --prompt-manifest is required')
    modes = parse_modes(args.modes)
    if args.variant != 'baseline' and (args.sampling != 'greedy' or args.prefill_token_budget or modes != ['serial', 'streams']):
        raise ValueError('preallocated variant supports only the original greedy chunk128 serial/streams workload')
    if args.prefill_token_budget < 0:
        raise ValueError('prefill token budget must be nonnegative')


def parse_modes(text):
    modes = [mode.strip() for mode in text.split(',') if mode.strip()]
    if not modes or modes[0] != 'serial' or len(set(modes)) != len(modes) \
            or any(mode not in ('serial', 'streams', 'batched') for mode in modes):
        raise ValueError('--modes must start with serial and use unique entries from streams,batched')
    return modes


def prefill_chunk(args):
    """Per-sequence chunk: fixed --prefill-chunk-size, or a total per-forward token budget shared by C sequences."""
    budget = args.prefill_token_budget
    if not budget:
        return args.prefill_chunk_size
    per_sequence = max(128, (budget // args.concurrency) // 128 * 128)
    return min(per_sequence, args.prompt_length)


def context_contract(prompt_length, output_length, trained_context, acknowledgement):
    total = prompt_length+output_length
    extrapolated = total > trained_context
    if extrapolated and not acknowledgement:
        raise ValueError('requested tokens exceed trained context; explicit --system-only-extrapolation required')
    return dict(prompt_length=prompt_length, output_length=output_length, total_tokens=total,
                trained_context=trained_context, context_extrapolated=extrapolated,
                system_only_extrapolation_acknowledged=bool(acknowledgement),
                model_config_changed=False, rope_changed=False,
                quality_claim='none; systems workload only' if extrapolated else 'no quality evaluation')


class PreallocatedState:
    """Cache only immutable prefill metadata; mutable decode context is separate."""
    def __init__(self, runner):
        self.runner, self.cache, self.contexts, self.reset_slots = runner, None, {}, {}

    def refresh(self):
        if self.runner.cache is not self.cache:
            self.cache = self.runner.cache
            self.contexts.clear()
            self.reset_slots.clear()

    def context(self, start, length):
        self.refresh()
        key = (start, length)
        if key not in self.contexts:
            self.contexts[key] = self.cache.context(start, length, prefill=True)
        return self.contexts[key]

    def slots(self, prompt_length):
        import torch
        self.refresh()
        if prompt_length not in self.reset_slots:
            offsets = torch.arange(self.cache.batch_size, device=self.runner.device, dtype=torch.long)
            self.reset_slots[prompt_length] = offsets*(self.cache.pages_per_seq*self.cache.block_size)+prompt_length
        return self.reset_slots[prompt_length]


@contextmanager
def preallocated_runner(runner):
    """Instance-only adapter, restored on exit; no frozen source changes."""
    import torch
    state = PreallocatedState(runner)
    original_prefill, original_reset, original_release = runner.prefill, runner.reset_decode, runner.release_cache

    def release():
        # Do not retain an entire old KV allocation through a metadata cache.
        state.contexts.clear()
        state.reset_slots.clear()
        state.cache = None
        original_release()

    @torch.inference_mode()
    def prefill(tokens, *, chunk_size=128):
        runner.model.validate_router_caches()
        if tokens.ndim != 2 or tokens.shape[0] != runner.cache.batch_size:
            raise ValueError('tokens must be [batch, prompt length]')
        if tokens.shape[1] >= runner.cache.capacity_tokens:
            raise ValueError('reserve cache space for generated tokens')
        runner._decode_ready = False
        batch, length = tokens.shape
        for start in range(0, length, chunk_size):
            chunk = tokens[:, start:start+chunk_size].contiguous()
            positions, context = state.context(start, chunk.shape[1])
            hidden = runner.model(chunk.flatten(), positions, context)
        last = hidden.view(batch, -1, runner.config.hidden_size)[:, -1]
        logits = runner.model.compute_logits(last)
        first = logits.argmax(-1)
        runner._finish_prefill(first, length)
        return logits, first

    @torch.inference_mode()
    def reset(first_token, prompt_length):
        if not 0 <= prompt_length < runner.cache.capacity_tokens:
            raise ValueError('decode reset exceeds cache capacity')
        runner.decode_cursor = prompt_length
        runner.decode_input.copy_(first_token)
        runner.decode_positions.fill_(prompt_length)
        runner.decode_context.seqused_k.fill_(prompt_length+1)
        runner.decode_context.slot_mapping.copy_(state.slots(prompt_length))
        runner._decode_ready = True

    runner.prefill, runner.reset_decode, runner.release_cache = prefill, reset, release
    try:
        yield state
    finally:
        runner.prefill, runner.reset_decode, runner.release_cache = original_prefill, original_reset, original_release
        state.contexts.clear()
        state.reset_slots.clear()
        state.cache = None


def make_backend(runner, prompts, args, variant):
    import torch
    from closed_loop.backend import FixedCohortBackend
    from closed_loop.protocol import RoundResult

    class PreallocatedBackend(FixedCohortBackend):
        def __init__(self, *positional, **kwargs):
            super().__init__(*positional, **kwargs)
            self.prompt_staging = torch.empty_like(self.prompt_bank[0], device=self.runner.device)

        def stage_prompt(self, prompt):
            self.prompt_staging.copy_(prompt)
            return self.prompt_staging

        def observe(self, index):
            # The inherited observation path calls prompt.to(device). Route
            # that operation through the actual candidate staging buffer too.
            original_bank = self.prompt_bank
            backend = self
            class StagedPrompt:
                def __init__(self, value):
                    self.value = value

                def to(self, device):
                    if device != backend.runner.device:
                        raise ValueError('qualification staging target changed')
                    return backend.stage_prompt(self.value)

            class StagedBank:
                def __len__(self):
                    return len(original_bank)

                def __getitem__(self, index):
                    return StagedPrompt(original_bank[index])

            self.prompt_bank = StagedBank()
            try:
                return super().observe(index)
            finally:
                self.prompt_bank = original_bank

        @torch.inference_mode()
        def run_round(self, index):
            self.assert_execution()
            tokens = self.stage_prompt(self.prompt_bank[index % len(self.prompt_bank)])
            _, first = self.runner.prefill(tokens, chunk_size=self.chunk_size)
            last = self.runner.decode(self.output_length-1)
            if tuple(first.shape) != (self.concurrency,) or tuple(last.shape) != (self.concurrency,):
                raise RuntimeError('generated token batch changed')
            self.synchronize()
            return RoundResult(self.concurrency, self.concurrency*self.output_length,
                               decode_graph_hits=self.output_length-1 if self.use_graph else 0,
                               decode_eager_calls=0 if self.use_graph else self.output_length-1,
                               prefill_eager_calls=math.ceil(self.prompt_length/self.chunk_size))

    cls = PreallocatedBackend if variant == 'preallocated' else FixedCohortBackend
    return cls(runner, prompts, output_length=args.output_length,
               chunk_size=prefill_chunk(args), use_graph=True, sampling_seed=args.sampling_seed, page_size=128, decode_splits=0)


def qualify_backend(backend, variant, write, baseline_references=None, modes=('serial', 'streams')):
    import torch
    from closed_loop.backend import compare_observations, router_state
    initial_state = router_state(backend.runner.model)
    references = {}
    for mode in modes:
        backend.prepare(mode)
        fresh_a, fresh_b, replaced_a = backend.observe(0), backend.observe(1), backend.observe(0)
        write(dict(kind='qualification_check', variant=variant, mode=mode, check='kv_replacement',
                   graph_reused_without_recapture=True,
                   **compare_observations(fresh_a, replaced_a, atol=0., rtol=0.)))
        if initial_state != router_state(backend.runner.model):
            raise AssertionError('router weights/bias/cache/backend changed')
        write(dict(kind='qualification_check', variant=variant, mode=mode,
                   check='router_state_unchanged', status='PASS', state=initial_state))
        references[mode] = (fresh_a, fresh_b)
        if baseline_references is not None:
            for index, actual in enumerate((fresh_a, fresh_b)):
                write(dict(kind='qualification_check', variant=variant, mode=mode,
                           check='baseline_vs_candidate', prompt_index=index,
                           **compare_observations(baseline_references[mode][index], actual, atol=0., rtol=0.)))
        # Recapture without observation hooks and check the actual timed request
        # method, including candidate prompt staging and graph reuse, on A/B/A.
        backend.prepare(mode)
        graph = backend.runner.graph
        for index in (0, 1, 0):
            result = backend.run_round(index)
            torch.testing.assert_close(backend.runner.decode_output.cpu(), references[mode][index]['tokens'][:, -1],
                                       atol=0., rtol=0.)
            if backend.runner.graph is not graph or result.decode_graph_hits != backend.output_length-1:
                raise AssertionError('actual request path changed the prepared decode graph')
        write(dict(kind='qualification_check', variant=variant, mode=mode,
                   check='actual_request_path_A_B_A', status='PASS', final_tokens_exact=True,
                   prompt_staging_tested=variant == 'preallocated', graph_reused_without_recapture=True))
    for mode in modes[1:]:
        for index in range(2):
            write(dict(kind='qualification_check', variant=variant, check=f'serial_vs_{mode}', prompt_index=index,
                       **compare_observations(references['serial'][index], references[mode][index], atol=0., rtol=0.)))
    write(dict(kind='qualification', variant=variant, status='PASS', atol=0., rtol=0.,
               prompt_samples_per_mode=2, replacement_pattern='A/B/A', generated_tokens_exact=True,
               expert_ids_exact=True, decode_graph=True, prefill_graph=False,
               baseline_parity_checked=baseline_references is not None))
    return references


def measure_throughput(backend, args, variant, write):
    import torch
    from closed_loop.protocol import run_window
    rows = []
    all_modes = parse_modes(args.modes)
    for repeat in range(args.repeats):
        shift = repeat % len(all_modes)  # rotate the measurement order across repeats
        modes = all_modes[shift:] + all_modes[:shift]
        for order_index, mode in enumerate(modes):
            backend.prepare(mode)
            for index in range(args.warmup_rounds):
                backend.run_round(index)
            torch.cuda.reset_peak_memory_stats(backend.runner.device)
            row = dict(kind='measurement', variant=variant, mode=mode, concurrency=args.concurrency,
                       repeat=repeat, order_index=order_index, decode_graph=True, prefill_graph=False,
                       requested_branch_execution=mode,
                       actual_branch_execution=backend.runner.model.config.branch_execution,
                       throughput_graph_instrumented=False, local_kv_bytes=backend.runner.cache.nbytes,
                       **run_window(backend, args.concurrency, args.output_length, args.window_seconds))
            row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(backend.runner.device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(backend.runner.device))
            write(row)
            rows.append(row)
    summaries = {mode: sample_statistics([row['output_tokens_per_second'] for row in rows if row['mode']==mode])
                 for mode in all_modes}
    decode = {mode: sample_statistics([row['decode_tokens_per_second'] for row in rows if row['mode']==mode])
              for mode in all_modes if all('decode_tokens_per_second' in row for row in rows if row['mode']==mode)}
    speedups = {mode: summaries[mode]['mean']/summaries['serial']['mean'] for mode in all_modes[1:]}
    decode_speedups = {mode: decode[mode]['mean']/decode['serial']['mean'] for mode in all_modes[1:]
                       if mode in decode and 'serial' in decode}
    record = dict(kind='summary', variant=variant, concurrency=args.concurrency, by_mode=summaries,
                  decode_by_mode=decode, speedups=speedups, decode_speedups=decode_speedups, status='PASS')
    if all_modes == ['serial', 'streams']:
        record['speedup'] = speedups['streams']
    write(record)


def measure_profile(backend, args, variant, write, output_dir):
    from closed_loop.backend import trace_decode
    traces, summaries = {}, {}
    profile_modes = parse_modes(args.modes)
    for mode in profile_modes:
        backend.prepare(mode)
        for index in range(args.warmup_rounds):
            backend.run_round(index)
        rows = []
        for index in range(args.profile_rounds):
            row = profile_request(backend, index)
            write(dict(row, variant=variant))
            rows.append(row)
        summaries[mode] = profile_summary(rows)
        write(dict(summaries[mode], variant=variant))
        traces[mode] = []
        for sample in range(args.timeline_samples):
            row = dict(trace_decode(backend), kind='stage_trace', variant=variant,
                       concurrency=args.concurrency, sample=sample,
                       decode_position=args.prompt_length,
                       sampling='independent warmed diagnostic graphs at the first cached decode position')
            traces[mode].append(row)
            write(row)
        if args.torch_profiler_steps:
            trace_path = output_dir/f'{variant}-{mode}-torch-trace.json'
            write(dict(export_torch_trace(backend, trace_path, args.torch_profiler_steps), variant=variant, mode=mode))
    for mode in profile_modes[1:]:
        speedups = {name: summaries['serial']['per_request'][name]['mean']/summaries[mode]['per_request'][name]['mean']
                    for name in ('prefill_wall_ms', 'decode_total_wall_ms', 'request_total_wall_ms')}
        write(dict(kind='profile_comparison', variant=variant, concurrency=args.concurrency, mode=mode,
                   component_speedups=speedups, **compare_timelines(traces['serial'], traces[mode])))


def check_storage(output):
    """Optional scratch contract. With MBT_SCRATCH_ROOT set, the output directory and every
    compiler cache must sit under it and be writable; this keeps runs off a small shared home quota.
    Unset, the benchmark writes wherever --output-dir points."""
    target=Path(output).resolve()
    root=os.environ.get('MBT_SCRATCH_ROOT')
    if root:
        root=Path(root).resolve(strict=True)
        for key in ('TMPDIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH'):
            cache=Path(os.environ[key]).resolve(strict=True)
            if not cache.is_relative_to(root) or not os.access(cache,os.W_OK):
                raise RuntimeError(f'{key} must be writable new scratch')
        if not target.is_relative_to(root):
            raise ValueError('benchmark output belongs on scratch')
    target.parent.mkdir(parents=True,exist_ok=True)
    return target


def read_prompt(prefix, batch, length, vocab_size, offset_tokens=0):
    import numpy as np
    import torch
    prefix=Path(prefix)
    with Path(str(prefix)+'.idx').open('rb') as stream:
        if stream.read(9)!=b'MMIDIDX\x00\x00' or struct.unpack('<Q',stream.read(8))[0]!=1:
            raise ValueError('unsupported Megatron indexed dataset')
        code=struct.unpack('<B',stream.read(1))[0]
    dtype={1:'u1',2:'i1',3:'<i2',4:'<i4',5:'<i8',8:'<u2'}.get(code)
    if dtype is None:
        raise ValueError('benchmark requires integer token data')
    source=np.memmap(str(prefix)+'.bin',dtype=dtype,mode='r')
    # Fixed disjoint ranges at each length, no sample shuffle or random tokens.
    if offset_tokens < 0 or len(source)<offset_tokens+batch*length:
        raise ValueError('validation prefix is too short')
    selected=np.array(source[offset_tokens:offset_tokens+batch*length],dtype=np.int64).reshape(batch,length)
    if selected.min()<0 or selected.max()>=vocab_size:
        raise ValueError('token IDs outside checkpoint vocabulary')
    evidence=dict(dataset=str(prefix.resolve()),dtype=dtype,offset_tokens=offset_tokens,
                  tokens=batch*length,sha256=hashlib.sha256(selected.tobytes()).hexdigest())
    return torch.from_numpy(selected),evidence


def write_record(output, record):
    record = dict(record, recorded_utc=datetime.now(timezone.utc).isoformat())
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    with output.open('a') as stream:
        stream.write(line+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def source_hashes():
    from nanovllm.experimental.wide_serving import runner
    paths = list((Path(__file__).parent/'closed_loop').glob('*.py')) + list(Path(runner.__file__).parent.glob('*.py'))
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def run(args):
    validate_args(args)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('requires a single-GPU process')
    import torch
    from closed_loop.protocol import WindowFailure
    from nanovllm.experimental.wide_serving.runner import ServingRunner
    output = check_storage(str(Path(args.output_dir)/'results.jsonl'))
    with output.open('x'):
        pass
    write = lambda row: write_record(output, row)
    requested = ['baseline', 'preallocated'] if args.variant == 'both' else [args.variant]
    # Candidate-only still needs a freshly qualified baseline reference.
    variants = ['baseline'] + (['preallocated'] if 'preallocated' in requested else [])
    qualified, rejected, runner, failure = [], [], None, None
    baseline_passed = False
    phase, variant = 'load', None
    try:
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        runner_cls = ServingRunner
        if json.loads((Path(args.export)/'config.json').read_text()).get('format_version') == 2:
            from nanovllm.experimental.mbt_serving.runner import MBTServingRunner as runner_cls  # v2 MBT exports
        runner = runner_cls(args.export, branch_execution='serial', moe_backend='auto',
                            elementwise_backend='triton', router_backend='fused_post8',
                            greedy_head_backend='replicated', qk_backend='packed', device='cuda:0')
        context = context_contract(args.prompt_length, args.output_length, runner.config.max_position_embeddings,
                                   args.system_only_extrapolation)
        runner.set_sampling(args.sampling, temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, seed=args.sampling_seed)
        if args.prompt_manifest is not None:
            from prompt_manifest import load_manifest
            prompts, evidence = load_manifest(args.prompt_manifest, length=args.prompt_length,
                                              count=args.prompt_rounds*args.concurrency,
                                              vocab_size=runner.config.vocab_size,
                                              expected_tokenizer_sha256=args.expected_tokenizer_sha256,
                                              source=args.manifest_source)
        else:
            prompts, evidence = read_prompt(args.data_prefix, args.prompt_rounds*args.concurrency,
                                           args.prompt_length, runner.config.vocab_size, args.offset_tokens)
        prompts = prompts.reshape(args.prompt_rounds, args.concurrency, args.prompt_length)
        manifest = Path(args.export)/'manifest.json'
        own_sources = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), Path(__file__).with_name('profiling.py'))}
        props = torch.cuda.get_device_properties(runner.device)
        try:
            driver = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total',
                                     '--format=csv,noheader'], check=True, capture_output=True,
                                    text=True, timeout=15).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            driver = f'unavailable: {error}'
        write(dict(kind='environment', schema_version=1, phase=args.phase, concurrency=args.concurrency,
                   workload=vars(args), context_contract=context, data=evidence, model=runner.config.to_dict(),
                   export=str(Path(args.export).resolve()), export_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   export_manifest=json.loads(manifest.read_text()), load_report=runner.load_report,
                   source_sha256=dict(source_hashes(), **own_sources), slurm_job=os.environ.get('SLURM_JOB_ID'),
                   host=os.uname().nodename, torch=torch.__version__, cuda=torch.version.cuda,
                   nvidia_smi=driver, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   gpu=dict(name=props.name, uuid=str(getattr(props, 'uuid', 'unavailable')), total_memory=props.total_memory),
                   gpus=1, tp=1, ep=1, pp=1, cpu_threads=2, parity_policy=dict(atol=0., rtol=0., tokens='exact', expert_ids='exact'),
                   scheduler='fixed-C zero-think-time synchronous cohorts; prompt selection, staging, prefill, decode, resets inside throughput windows',
                   counter_semantics='output token numerator includes first prefill-generated token; excludes prompt tokens; graph hits count batched decode replays',
                   graph_policy=dict(decode=True, prefill=False, fallback='forbidden'),
                   sampling=('greedy' if args.sampling == 'greedy' else
                             dict(kind='sample', temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                  seed=args.sampling_seed, implementation='FlashInfer deterministic, after the decode graph',
                                  reseed='per request: seed + prompt index, identical across modes')),
                   lengths='fixed; EOS ignored; identical deterministic prompt bank for every mode/variant',
                   modes=parse_modes(args.modes),
                   attention_kernels='FA3 (sgl_kernel) paged prefill and decode, page size 128',
                   prefill_chunk=dict(per_sequence=prefill_chunk(args), token_budget=args.prefill_token_budget or None),
                   candidate='reuse immutable chunk128 prefill metadata, reset slots, prompt staging; model arithmetic/shape unchanged',
                   scientific_limit='same-export runtime equivalence only; no native-checkpoint quality fidelity or context-extension claim'))
        baseline_references = None
        for variant in variants:
            manager = preallocated_runner(runner) if variant == 'preallocated' else nullcontext()
            with manager:
                backend = make_backend(runner, prompts, args, variant)
                phase = 'qualification'
                try:
                    references = qualify_backend(backend, variant, write,
                                                 baseline_references if variant == 'preallocated' else None,
                                                 modes=parse_modes(args.modes))
                except AssertionError as error:
                    if variant == 'baseline':
                        raise
                    rejected.append(variant)
                    write(dict(kind='candidate_rejection', variant=variant, status='FAIL', error=repr(error),
                               policy='strict tolerance unchanged; candidate receives no performance measurements'))
                    continue
                if variant in requested:
                    qualified.append(variant)
                if variant == 'baseline':
                    baseline_passed = True
                    baseline_references = references
                if variant not in requested or args.phase == 'smoke':
                    continue
                phase = args.phase
                if args.phase == 'throughput':
                    measure_throughput(backend, args, variant, write)
                else:
                    measure_profile(backend, args, variant, write, output.parent)
    except Exception as error:
        failure = error
        underlying = error.cause if isinstance(error, WindowFailure) else error
        write(dict(kind='failure', status='ERROR', phase=phase, variant=variant,
                   concurrency=args.concurrency, error=repr(underlying),
                   oom=isinstance(underlying, (torch.OutOfMemoryError, MemoryError)),
                   partial_counters=error.counters if isinstance(error, WindowFailure) else None,
                   policy='stop; no smaller workload, fallback, tolerance change, or retry'))
    finally:
        teardown_complete = False
        if runner is not None:
            runner.close()
            teardown_complete = True
        write(dict(kind='completion', phase=args.phase,
                   status='ERROR' if failure else ('PARTIAL' if rejected else 'PASS'),
                   requested_variants=requested, qualified_variants=qualified, rejected_variants=rejected,
                   baseline_qualification_passed=baseline_passed, teardown_complete=teardown_complete))
    if failure is not None:
        raise failure


def main():
    p = parser()
    args = p.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        p.error(str(error))
    run(args)


if __name__ == '__main__':
    main()
