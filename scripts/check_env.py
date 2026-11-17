#!/usr/bin/env python3
"""Print dependency versions, without importing CUDA unless --cuda is given."""
import argparse
import importlib.metadata
import importlib.util
import json
import platform

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--cuda', action='store_true', help='Inspect the GPU in your allocated compute session')
a = p.parse_args()
result = {'python': platform.python_version(), 'packages': {}}
for name in ('torch','transformer-engine','flash-attn-4','numpy','pybind11','tiktoken','wandb'):
    try:
        result['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result['packages'][name] = None
result['fused_gradient_extension'] = importlib.util.find_spec('fused_weight_gradient_mlp_cuda') is not None
if a.cuda:
    import torch
    result['cuda'] = torch.version.cuda
    result['devices'] = [{'name': torch.cuda.get_device_name(i),
                          'capability': torch.cuda.get_device_capability(i)}
                         for i in range(torch.cuda.device_count())]
print(json.dumps(result, indent=2))
