"""Turn a LoRA adapter into an Ollama model, quantised like the base Ollama model (Q4_K_M), so
the LoRA arm differs from the harness-only arm only by the adapter.

Needs a llama.cpp checkout (for convert_hf_to_gguf.py and llama-quantize):
    git clone https://github.com/ggml-org/llama.cpp ..\\llama.cpp
    pip install -r ..\\llama.cpp\\requirements.txt      # plus a llama.cpp release build for llama-quantize

    python scripts/export_ollama.py --base Qwen/Qwen2.5-Coder-7B-Instruct --adapter lora/qwen7b/final \\
        --llama-cpp ..\\llama.cpp --quantize ..\\llama.cpp\\build\\bin\\llama-quantize.exe \\
        --ollama-base qwen2.5-coder:7b --name qwen7b-cube-lora

Merging runs on the CPU in bf16: about 16 GB of RAM for 7B and 30 GB for 14B.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> str:
    print(">", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--quantize", type=Path, required=True, help="path to the llama-quantize binary")
    ap.add_argument("--ollama-base", required=True, help="the harness-only Ollama model, e.g. qwen2.5-coder:7b")
    ap.add_argument("--name", required=True, help="name of the new Ollama model")
    ap.add_argument("--work", type=Path, default=Path("lora/export"))
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    work = args.work / args.name
    merged = work / "merged"
    if not (merged / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cpu")
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        model.save_pretrained(merged, safe_serialization=True)
        AutoTokenizer.from_pretrained(args.base).save_pretrained(merged)
        del model
    f16 = work / "model-f16.gguf"
    q4 = work / "model-Q4_K_M.gguf"
    if not f16.exists():
        run([sys.executable, args.llama_cpp / "convert_hf_to_gguf.py", merged, "--outtype", "f16", "--outfile", f16])
    if not q4.exists():
        run([args.quantize, f16, q4, "Q4_K_M"])

    # Same chat template, stop tokens and parameters as the base Ollama model; only the weights change.
    base_modelfile = run(["ollama", "show", "--modelfile", args.ollama_base])
    lines = base_modelfile.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("FROM "))  # first FROM is the model line
    lines[i] = f"FROM {q4.resolve()}"
    modelfile = work / "Modelfile"
    modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(run(["ollama", "create", args.name, "-f", modelfile]))
    print(f"done: python -m harness.evaluate ... --model ollama_chat/{args.name}")


if __name__ == "__main__":
    main()
