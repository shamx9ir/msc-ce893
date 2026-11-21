"""Write an Ollama Modelfile whose chat template lists the tool schemas exactly as in training.

Ollama renders the `tools` it receives through its own Go structs, which drop fields such as
"title" and "additionalProperties" and change the instruction line. The search_values v3 adapter
fits the training format so tightly that, on Ollama's rendering, it writes run_query calls without
the "query" wrapper. This template writes harness/tool_schema.json literally (json.dumps of each
entry, one per line, as in the Qwen chat template used for training) whenever tools are present.
The <tool_call> tags are kept, so Ollama still returns native tool calls. One line keeps Ollama's
wording ("within <tools></tools>:" where training says "... XML tags:"); the reported -nt runs used
exactly this template, for the sv3 and the baseline model alike.

Usage (a GGUF that export_ollama.py already made, or an existing Ollama model):
    python scripts/make_literal_tools_modelfile.py --from E:/lora-export/qwen7b-cube-sv3/model-Q4_K_M.gguf --out Modelfile
    ollama create qwen7b-cube-sv3-nt -f Modelfile
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = r'''{{- if .Suffix }}<|fim_prefix|>{{ .Prompt }}<|fim_suffix|>{{ .Suffix }}<|fim_middle|>
{{- else if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools>:
<tools>
@TOOLS@
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="GGUF path or Ollama model name for the FROM line")
    ap.add_argument("--out", type=Path, default=Path("Modelfile"))
    args = ap.parse_args()
    tools = json.loads((ROOT / "harness" / "tool_schema.json").read_text(encoding="utf-8"))
    tpl = TEMPLATE.replace("@TOOLS@", "\n".join(json.dumps(t, ensure_ascii=False) for t in tools))
    text = (f"FROM {args.src}\nTEMPLATE \"\"\"{tpl}\"\"\"\n"
            'SYSTEM "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."\n')
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
