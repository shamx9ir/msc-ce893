"""Turn a LoRA adapter into an Ollama model, quantised like the base Ollama model (Q4_K_M), so
the LoRA arm differs from the harness-only arm only by the adapter.

Needs a llama.cpp checkout for convert_hf_to_gguf.py. Quantising to Q4_K_M is done by llama-quantize
when --quantize is given, otherwise by Ollama itself (`ollama create -q q4_K_M`, same ggml quantiser):
    git clone https://github.com/ggml-org/llama.cpp ..\\llama.cpp
    pip install -r ..\\llama.cpp\\requirements.txt      # plus a llama.cpp release build for llama-quantize

    python scripts/export_ollama.py --base Qwen/Qwen2.5-Coder-7B-Instruct --adapter lora/qwen7b/final \\
        --llama-cpp ..\\llama.cpp --quantize ..\\llama.cpp\\build\\bin\\llama-quantize.exe \\
        --ollama-base qwen2.5-coder:7b --name qwen7b-cube-lora

Merging runs on the CPU one safetensors shard at a time (a few GB of RAM, also for 14B).
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> str:
    print(">", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout


def merge_by_shard(base: str, adapter: Path, out: Path) -> None:
    """W' = W + (alpha / r) * B @ A, one safetensors shard at a time.

    Same result as PeftModel.merge_and_unload(), but only one shard (~4 GB) is in RAM at a
    time, so a 14B model merges on a 32 GB machine."""
    import json
    import shutil
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    src = Path(snapshot_download(base, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(adapter / "adapter_model.safetensors"))
    pairs = {}  # base weight name -> (A, B)
    for k, v in lora.items():
        if ".lora_A." in k:
            name = k.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")
            pairs[name] = (v, lora[k.replace(".lora_A.", ".lora_B.")])
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    for shard in sorted(src.glob("*.safetensors")):
        tensors = {}
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                w = f.get_tensor(name)
                if name in pairs:
                    a, b = pairs[name]
                    w = (w.float() + scale * (b.float() @ a.float())).to(w.dtype)
                    done.add(name)
                tensors[name] = w
            meta = f.metadata()
        save_file(tensors, str(out / shard.name), metadata=meta or {"format": "pt"})
        print(f"merged {shard.name}", flush=True)
        del tensors
    missing = set(pairs) - done
    if missing:
        raise SystemExit(f"{len(missing)} LoRA weights had no base weight, e.g. {sorted(missing)[:3]}")
    for f in src.iterdir():
        if f.suffix != ".safetensors" and f.is_file():
            shutil.copy(f, out / f.name)
    print(f"merged {len(done)} weights (scale {scale}) into {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--quantize", type=Path, default=None,
                    help="path to llama-quantize; without it Ollama quantises the f16 GGUF (ollama create -q)")
    ap.add_argument("--ollama-base", required=True, help="the harness-only Ollama model, e.g. qwen2.5-coder:7b")
    ap.add_argument("--name", required=True, help="name of the new Ollama model")
    ap.add_argument("--work", type=Path, default=Path("lora/export"))
    ap.add_argument("--merged-dir", type=Path, default=None,
                    help="where the merged bf16 model goes (default <work>/merged); can be on another drive")
    args = ap.parse_args()

    work = args.work / args.name
    merged = args.merged_dir or work / "merged"
    if not (merged / "config.json").exists():
        merge_by_shard(args.base, args.adapter, merged)
    f16 = work / "model-f16.gguf"
    q4 = work / "model-Q4_K_M.gguf"
    if not f16.exists():
        run([sys.executable, args.llama_cpp / "convert_hf_to_gguf.py", merged, "--outtype", "f16", "--outfile", f16])
    if args.quantize and not q4.exists():
        run([args.quantize, f16, q4, "Q4_K_M"])
    source = q4 if args.quantize else f16

    # Same chat template, stop tokens and parameters as the base Ollama model; only the weights change.
    base_modelfile = run(["ollama", "show", "--modelfile", args.ollama_base])
    lines = base_modelfile.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("FROM "))  # first FROM is the model line
    lines[i] = f"FROM {source.resolve()}"
    modelfile = work / "Modelfile"
    modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")  # no \r in TEMPLATE on Windows
    create = ["ollama", "create", args.name, "-f", modelfile]
    print(run(create if args.quantize else create + ["-q", "q4_K_M"]))
    tpl = lambda m: run(["ollama", "show", "--template", m])
    if tpl(args.name) != tpl(args.ollama_base):
        sys.exit(f"chat template of {args.name} differs from {args.ollama_base}")
    print(f"done: python -m harness.evaluate ... --model ollama_chat/{args.name}")


if __name__ == "__main__":
    main()
