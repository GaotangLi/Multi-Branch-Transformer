"""Convert a local native release bundle to an experimental HF model folder.

Input is a model-only release bundle, not a Megatron distributed training
checkpoint. Nothing is downloaded or uploaded; source files remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

try:
    from .weight_mapping import configuration_from_architecture, convert_values
except ImportError:  # Direct invocation: python utils/checkpoint_conversion/convert_to_hf.py
    from weight_mapping import configuration_from_architecture, convert_values


SOURCE_FILES = ("config.json", "manifest.json", "model.safetensors")
MODEL_CODE = ("configuration_mbt.py", "modeling_mbt.py", "modeling_mbt_core.py")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "LICENSE.tokenizer", "NOTICE.tokenizer")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def tokenizer_inputs(directory, config):
    """Check the explicitly supplied local tokenizer's vocabulary and roles.

    This does not establish that arbitrary BPE merges match training. The caller
    must provide the exact compatible tokenizer, including its license files.
    """
    if directory is None:
        return []
    directory = Path(directory).resolve(strict=True)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        if not (directory / filename).is_file():
            raise ValueError("Missing local fast-tokenizer file: " + filename)
    metadata = json.loads((directory / "tokenizer_config.json").read_text())
    if metadata.get("auto_map"):
        raise ValueError("Supply the compatible fast tokenizer, without remote tokenizer code")
    if metadata.get("chat_template"):
        raise ValueError("These pretrained base checkpoints do not use a chat template")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    if tokenizer.get_vocab_size(with_added_tokens=True) != config["vocab_size"]:
        raise ValueError("Tokenizer vocabulary size differs from the model")
    for name, expected in (("<|begin_of_text|>", config["bos_token_id"]),
                           ("<|end_of_text|>", config["eos_token_id"]),
                           ("<|fim_pad|>", config["pad_token_id"])):
        if tokenizer.token_to_id(name) != expected:
            raise ValueError("Tokenizer special-token mapping differs: " + name)
    # Check the serialized wrapper roles as well as the backend vocabulary.
    for role, spelling in (("bos_token", "<|begin_of_text|>"),
                           ("eos_token", "<|end_of_text|>"),
                           ("pad_token", "<|fim_pad|>")):
        actual = metadata.get(role)
        if isinstance(actual, dict):
            actual = actual.get("content")
        if actual != spelling:
            raise ValueError("Tokenizer wrapper role differs: " + role)
    if metadata.get("add_bos_token") or metadata.get("add_eos_token"):
        raise ValueError("The training tokenizer does not automatically add BOS/EOS")
    return [directory / filename for filename in TOKENIZER_FILES if (directory / filename).is_file()]


def convert(bundle, output, tokenizer_dir=None):
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import GenerationConfig
    try:
        from .hf_reference.configuration_mbt import MBTConfig
    except ImportError:
        from hf_reference.configuration_mbt import MBTConfig

    bundle, output = Path(bundle).resolve(strict=True), Path(output).resolve()
    if not bundle.is_dir() or output.exists():
        raise ValueError("Input must be a bundle directory; output must not exist")
    if output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("Input and output directories must be separate")
    for filename in SOURCE_FILES:
        if not (bundle / filename).is_file():
            raise ValueError("Missing native release file: " + filename)
    native_config = json.loads((bundle / "config.json").read_text())
    manifest = json.loads((bundle / "manifest.json").read_text())
    hashes = {}
    for filename, field in (("config.json", "config_sha256"), ("model.safetensors", "weights_sha256")):
        expected = manifest.get(field)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Missing native manifest checksum: " + field)
        hashes[filename] = sha256(bundle / filename)
        if hashes[filename] != expected:
            raise ValueError("Native bundle checksum mismatch: " + filename)
    config_data = configuration_from_architecture(native_config["architecture"])
    tokenizer_files = tokenizer_inputs(tokenizer_dir, config_data)
    values = load_file(str(bundle / "model.safetensors"), device="cpu")
    converted, report = convert_values(values, config_data, native_config["bundle_format"])
    output.mkdir(parents=True, exist_ok=False)
    save_file(converted, str(output / "model.safetensors"), metadata={"format": "pt"})
    reloaded = load_file(str(output / "model.safetensors"), device="cpu")
    if set(reloaded) != set(converted):
        raise ValueError("Serialized target tensor set changed")
    for name, tensor in converted.items():
        if reloaded[name].dtype != tensor.dtype or not torch.equal(reloaded[name], tensor):
            raise ValueError("Serialization changed a tensor: " + name)

    config = MBTConfig(**config_data)
    config.architectures = ["MBTForCausalLM"]
    config.auto_map = {"AutoConfig": "configuration_mbt.MBTConfig",
                       "AutoModelForCausalLM": "modeling_mbt.MBTForCausalLM"}
    config.dtype = "bfloat16"
    config.save_pretrained(output)
    GenerationConfig(bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id, use_cache=True, do_sample=False).save_pretrained(output)
    source = Path(__file__).parent / "hf_reference"
    code_hashes = {}
    for filename in MODEL_CODE:
        shutil.copyfile(source / filename, output / filename)
        code_hashes[filename] = sha256(output / filename)
    for path in tokenizer_files:
        shutil.copyfile(path, output / path.name)
    # Hashes and relative names only: omit machine paths, training identities,
    # data locations, credentials, and private native provenance.
    report.update(
        status="EXACT_TENSOR_MAPPING_VERIFIED", runtime_validation_performed=False,
        source_format=native_config["bundle_format"], source_files=hashes,
        target_weights_sha256=sha256(output / "model.safetensors"),
        model_code_sha256=code_hashes, all_tensors_exact_after_reload=True,
        tokenizer_files=[path.name for path in tokenizer_files],
        scope="Tensor mapping and serialization only; native BF16 output equivalence is not claimed.",
    )
    write_json(output / "conversion.json", report)
    (output / "README.md").write_text(
        "# Experimental Transformers conversion\n\n"
        "Converted from the native Multi-Branch Transformer release. "
        "The native bundle remains the authoritative checkpoint representation.\n\n"
        "Load with Transformers 5.15, PyTorch 2.12, `trust_remote_code=True`, "
        "and `dtype=torch.float32` for reference inference. BF16 kernel changes "
        "can alter logits and greedy predictions; this is not a bitwise native replica "
        "or the inference engine used for the paper's speed measurements.\n\n"
        + ("A local tokenizer was supplied explicitly; retain its applicable license terms.\n"
           if tokenizer_files else
           "No tokenizer is included. Supply the exact training-compatible tokenizer; "
           "do not substitute a default Llama chat tokenizer.\n")
    )
    print(json.dumps({"status": report["status"], "output": str(output),
                      "tensors": report["target_tensor_count"], "uploaded": False}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="bundle", required=True, help="Local native release bundle")
    parser.add_argument("--output", required=True, help="Fresh, separate output directory")
    parser.add_argument("--tokenizer-dir", help="Optional exact compatible HF fast tokenizer, supplied locally")
    args = parser.parse_args()
    convert(args.bundle, args.output, args.tokenizer_dir)


if __name__ == "__main__":
    main()
