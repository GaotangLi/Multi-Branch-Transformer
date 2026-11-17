# Installation

The supported entry point is a Linux/Python 3.12 environment with NVIDIA GPUs.
The model recipe uses BF16 and Transformer Engine. CPU configuration checks
do not require PyTorch or CUDA.

## 1. Create an environment on large storage

```bash
git clone https://github.com/GaotangLi/Multi-Branch-Transformer.git
cd Multi-Branch-Transformer
export MBT_STORAGE_ROOT=/scratch/$USER/mbt
mkdir -p "$MBT_STORAGE_ROOT"/{envs,tmp,cache}
export TMPDIR="$MBT_STORAGE_ROOT/tmp"
export PIP_CACHE_DIR="$MBT_STORAGE_ROOT/cache/pip"
export HF_HOME="$MBT_STORAGE_ROOT/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$MBT_STORAGE_ROOT/cache/torch_extensions"
python3.12 -m venv "$MBT_STORAGE_ROOT/envs/mbt"
source "$MBT_STORAGE_ROOT/envs/mbt/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[test]'
```

Replace `/scratch/$USER/mbt` with an existing writable large-disk path on your
machine. Keep checkpoints, compiler caches, dataset indexes, W&B files, and
temporary build output there. Do not put them in this Git checkout.

## 2. Install GPU dependencies

Install a CUDA-enabled PyTorch using the [official selector](https://pytorch.org/get-started/locally/).
Then install the training Python dependencies:

```bash
python -m pip install -e '.[train,wandb]'
```

The research environment used Transformer Engine at commit
`42b840051647eef89761a16dfdff87e82bb253ab`. To build that version against your
PyTorch installation, with compatible CUDA/cuDNN headers and libraries available:

```bash
NVTE_FRAMEWORK=pytorch MAX_JOBS=4 python -m pip install --no-build-isolation \
  'git+https://github.com/NVIDIA/TransformerEngine.git@42b840051647eef89761a16dfdff87e82bb253ab'
python -m pip install 'flash-attn-4==4.0.0b11'
```

Follow [Transformer Engine's installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/installation.html)
for CUDA/cuDNN prerequisites. FlashAttention-4 supports Hopper and Blackwell;
its [upstream instructions](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/README.md)
describe the CUDA-13 extra. The recorded environment used PyTorch 2.12.0;
mixing an extension built for a different PyTorch/CUDA ABI is unsupported.
Install/build on a compute machine, with build outputs on large storage.

The native training recipe also uses Apex's fused gradient-accumulation
extension, `fused_weight_gradient_mlp_cuda`. Use an environment that already
provides it (for example, an appropriately matched NVIDIA PyTorch container),
or build [NVIDIA Apex](https://github.com/NVIDIA/apex) with CUDA extensions:

```bash
mkdir -p "$MBT_STORAGE_ROOT/build"
git clone https://github.com/NVIDIA/apex "$MBT_STORAGE_ROOT/build/apex"
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 MAX_JOBS=4 python -m pip install \
  --no-build-isolation "$MBT_STORAGE_ROOT/build/apex"
python -c 'import fused_weight_gradient_mlp_cuda'
```

A Python-only Apex installation does not provide this extension. The CUDA
toolkit used to build it must match the PyTorch environment.

Do not globally set `CUTE_DSL_ARCH` to another GPU's architecture. Any local
compatibility override must match the allocated GPU, and belongs in your
site environment rather than a model configuration.

## 3. Validate before training

Run the CPU tests first, then the single-GPU smoke described in the
[training guide](training.md). The latter exercises the actual native
attention and sparse MoE. An eight-GPU capacity check must use the intended
EP8 layout; single-GPU memory use does not predict EP8 capacity.

Full training needs a prepared indexed dataset and a locally available
compatible tokenizer. See [data preparation](data.md). The example Slurm script
is deliberately free of institutional accounts, reservations, node names,
and absolute paths; supply those at submission for your own cluster.
