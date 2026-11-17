"""Real native-model causality check followed by normal train/save/resume.

The test supplies an explicit lifecycle subclass; it never modifies model or
training-module functions. Run with torchrun using the mbt.train arguments.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from mbt import train as entry
from mbt.runtime import TrainingLifecycle, preserve_rng
from mbt.training_config import require


class NativeProbeLifecycle(TrainingLifecycle):
    def tensor_summary(self, value):
        """Bounded diagnostic samples; checkpoint reload itself remains strict."""
        torch = self.training.torch
        if torch.is_tensor(value):
            flat = value.detach().reshape(-1)
            if flat.numel():
                count = min(64, flat.numel())
                indices = torch.arange(count, device=flat.device, dtype=torch.int64)
                indices = indices * (flat.numel()-1) // max(count-1, 1)
                samples = flat[indices].float().cpu().tolist()
            else:
                samples = []
            return {"shape": list(value.shape), "dtype": str(value.dtype), "samples": samples}
        if isinstance(value, dict):
            return {str(k): self.tensor_summary(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
        if isinstance(value, (list, tuple)):
            return [self.tensor_summary(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"type": type(value).__name__}

    def digest(self, value):
        return hashlib.sha256(json.dumps(self.tensor_summary(value), sort_keys=True).encode()).hexdigest()

    def optimizer_payload(self, optimizer):
        """Include DP-sharded Adam moments and FP32 masters, not only metadata.

        DistributedOptimizer.state_dict() deliberately omits rank-local
        parameter state. Read each underlying optimizer as well; this does not
        gather shards or change the native checkpoint format.
        """
        children = getattr(optimizer, 'chained_optimizers', None)
        if children is not None:
            return [self.optimizer_payload(child) for child in children]
        inner = getattr(optimizer, 'optimizer', None)
        require(inner is not None, "Native optimizer wrapper has no inner optimizer")
        return {'wrapper': optimizer.state_dict(),
                'inner': inner.state_dict(),
                'master_parameters': [group['params'] for group in inner.param_groups]}

    def rng_digest(self):
        import random
        import numpy
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        torch = self.training.torch
        # RNG states are small; compare every byte, not sampled coordinates.
        def tensor_bytes(value):
            return hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
        np_state = numpy.random.get_state()
        np_payload = (np_state[0], np_state[1].tobytes().hex(), *np_state[2:])
        return {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                "numpy": hashlib.sha256(repr(np_payload).encode()).hexdigest(),
                "torch_cpu": tensor_bytes(torch.get_rng_state()),
                "torch_cuda": tensor_bytes(torch.cuda.get_rng_state()),
                "model_parallel": {key: tensor_bytes(value) for key, value in
                                   get_cuda_rng_tracker().get_states().items()}}

    def state_summary(self):
        a = self.training.get_args()
        rng_before = self.rng_digest()
        versions_before = {name: parameter._version for name, parameter in self.explicit.named_parameters()}
        routers = {name: value for name, value in self.explicit.named_buffers() if name.endswith('expert_bias')}
        result = dict(parameters=self.digest(dict(self.explicit.named_parameters())),
                    router_bias=self.digest(routers),
                    optimizer=self.digest(self.optimizer_payload(self.optimizer)),
                    scheduler=self.tensor_summary(self.scheduler.state_dict()),
                    consumed_train_samples=a.consumed_train_samples,
                    rng=rng_before)
        require(result['optimizer'] == self.digest(self.optimizer_payload(self.optimizer)),
                "Reading optimizer state was not observational")
        require(versions_before == {name: parameter._version for name, parameter in self.explicit.named_parameters()},
                "State diagnostics modified model parameters")
        require(self.rng_digest() == rng_before, "State diagnostics consumed RNG")
        return result

    def model_ready(self, model, optimizer, config, opt_param_scheduler=None):
        torch = self.training.torch
        require(len(model) == 1, "Native test expects PP1")
        network = model[0]
        self.explicit = network
        while hasattr(self.explicit, 'module'):
            self.explicit = self.explicit.module
        self.optimizer = optimizer
        self.scheduler = opt_param_scheduler
        require(self.scheduler is not None, "Native optimizer scheduler is missing")
        expected = self.manifest['recipe']['model']['branches']
        require([len(branch.blocks) for branch in self.explicit.decoder.branches] == expected,
                "Explicit branch graph differs from the recipe")
        self.initial_state = self.state_summary()
        if self.manifest['resume_step']:
            proof = Path(self.manifest['output_dir']) / f"native-resume-proof-{self.manifest['resume_step']}.json"
            saved = json.loads(proof.read_text())['state']
            require(self.initial_state == saved,
                    "Native resume changed sampled model/optimizer state, router bias, scheduler, cursor, or RNG")
            print('MBT_NATIVE_RESUME_STATE PASS', flush=True)
        modes = [(module, module.training) for module in network.modules()]
        try:
            with preserve_rng(torch), torch.no_grad():
                network.eval()
                ids = torch.randint(0, 1000, (1, 16), device="cuda")
                positions = torch.arange(16, device="cuda").unsqueeze(0)
                changed = ids.clone()
                changed[:, 8:] = (changed[:, 8:] + 311) % 1000
                logits = network(ids, positions, None)
                other = network(changed, positions, None)
                require(tuple(logits.shape[:2]) == (1, 16), "Unexpected native logits layout")
                # Future-token changes can alter grouped expert GEMM sizes and
                # BF16 rounding, so equality is checked at BF16 precision.
                torch.testing.assert_close(logits[:, :8], other[:, :8], rtol=1e-2, atol=2e-2)
                require(not torch.equal(logits[:, 8:], other[:, 8:]), "Output ignored changed suffix tokens")
        finally:
            for module, mode in modes:
                module.training = mode
        print("MBT_NATIVE_CAUSALITY " + json.dumps(dict(status="PASS",
              model=self.manifest["recipe"]["model"])), flush=True)

    def before_checkpoint(self, iteration, non_persistent=False):
        self.state_at_save = self.state_summary()
        require(self.state_at_save['parameters'] != self.initial_state['parameters'],
                "Native training did not update sampled model parameters")
        require(self.state_at_save['optimizer'] != self.initial_state['optimizer'],
                "Native training did not update optimizer state")
        require(self.state_at_save['consumed_train_samples'] == iteration * self.training.get_args().global_batch_size,
                "Native train-data cursor is inconsistent")
        for module in self.explicit.modules():
            margins = getattr(module, 'qb_margin_chunks', None)
            require(margins is None or len(margins) == 0, "QB update-boundary state was not drained")

    def after_checkpoint(self, iteration, non_persistent=False):
        super().after_checkpoint(iteration, non_persistent)
        if self.training.torch.distributed.get_rank() == 0:
            path = Path(self.manifest['output_dir']) / f"native-resume-proof-{iteration}.json"
            path.write_text(json.dumps(dict(iteration=iteration, state=self.state_at_save,
                parameter_optimizer_check='all tensor shapes/dtypes and bounded coordinate samples',
                rng_check='all state bytes', full_checkpoint_load='native strict torch_dist'), indent=2)+'\n')


if __name__ == "__main__":
    raise SystemExit(entry.main(lifecycle_factory=NativeProbeLifecycle))
