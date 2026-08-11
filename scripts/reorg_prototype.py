#!/usr/bin/env python3
"""One-shot reorg of prototype/ into meaningful subfolders."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "prototype"

LAYOUT = {
    "pillars": [
        "pretrain_lm.py",
        "benchmark_gpu.py",
        "retrofit_lora.py",
        "prompt_injection_bench.py",
    ],
    "security": [
        "leakage_attack.py",
        "multi_tier_experiment.py",
        "nl_redteam_suite.py",
        "multi_probe_bench.py",
    ],
    "retrofit": [
        "open_llm_retrofit.py",
        "hf_nsa_retrofit.py",
        "llama_security_showcase.py",
        "real_llama_showcase.py",
        "retrofit_evolution_bench.py",
        "native_vs_retrofit_exp.py",
    ],
    "experiments": [
        "toy_experiment.py",
        "state_transformer.py",
        "dynamic_nsa_tradeoff.py",
        "ablation_study.py",
        "ablation_gating_study.py",
        "coupling_pareto_sweep.py",
    ],
    "demos": [
        "web_demo.py",
        "visualize_attention.py",
        "eval_showcase_prompts.py",
    ],
    "reporting": [
        "generate_benchmark_report.py",
    ],
    "results": [],
}

SHIM = '''"""Compatibility shim — script moved to prototype/{sub}/{name}.py"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "{sub}" / "{name}.py"
    runpy.run_path(str(target), run_name="__main__")
'''


def main() -> None:
    name_to_sub = {}
    for sub, files in LAYOUT.items():
        d = PROTO / sub
        d.mkdir(parents=True, exist_ok=True)
        if sub != "results":
            init = d / "__init__.py"
            if not init.exists():
                init.write_text(
                    f'"""prototype.{sub} — NSA research scripts."""\n',
                    encoding="utf-8",
                )
        for f in files:
            name_to_sub[f[:-3]] = sub
            src = PROTO / f
            dst = d / f
            if src.exists() and src.resolve() != dst.resolve():
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))
                print(f"moved {f} -> {sub}/")
            elif dst.exists():
                print(f"already {sub}/{f}")
            else:
                print(f"missing {f}")

    results_dir = PROTO / "results"
    results_dir.mkdir(exist_ok=True)
    for p in list(PROTO.glob("results_*.json")) + list(PROTO.glob("*.html")):
        if p.parent == results_dir:
            continue
        dest = results_dir / p.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(p), str(dest))
        print(f"moved result {p.name}")

    def fix_sys_path(text: str) -> str:
        text = re.sub(
            r'os\.path\.join\(\s*os\.path\.dirname\(__file__\)\s*,\s*["\']\.\.["\']\s*\)',
            'os.path.join(os.path.dirname(__file__), "..", "..")',
            text,
        )
        text = re.sub(
            r'Path\(__file__\)\.resolve\(\)\.parents\[1\]',
            'Path(__file__).resolve().parents[2]',
            text,
        )
        return text

    def rewrite_text(text: str) -> str:
        def repl_from(m: re.Match) -> str:
            name = m.group(1)
            tail = m.group(2)
            if name in name_to_sub:
                return f"from prototype.{name_to_sub[name]}.{name}{tail}"
            return m.group(0)

        text = re.sub(
            r"from prototype\.([A-Za-z_][A-Za-z0-9_]*)(\s+import\s+)",
            repl_from,
            text,
        )

        def repl_import(m: re.Match) -> str:
            name = m.group(1)
            if name in name_to_sub:
                return f"import prototype.{name_to_sub[name]}.{name}"
            return m.group(0)

        text = re.sub(
            r"import prototype\.([A-Za-z_][A-Za-z0-9_]*)\b",
            repl_import,
            text,
        )

        for name, sub in name_to_sub.items():
            text = text.replace(f"prototype/{name}.py", f"prototype/{sub}/{name}.py")
            text = text.replace(
                f'parents[1] / "prototype" / "{name}.py"',
                f'parents[1] / "prototype" / "{sub}" / "{name}.py"',
            )
            text = text.replace(
                f'/ "prototype" / "{name}.py"',
                f'/ "prototype" / "{sub}" / "{name}.py"',
            )

        # default out paths
        text = re.sub(
            r'(["\'])prototype/results_([A-Za-z0-9_\.]+)\1',
            r'\1prototype/results/results_\2\1',
            text,
        )
        text = text.replace(
            "prototype/results/results/results_",
            "prototype/results/results_",
        )
        return text

    # Patch moved scripts (depth 2 under prototype/<sub>/)
    for sub in LAYOUT:
        if sub == "results":
            continue
        for py in (PROTO / sub).glob("*.py"):
            if py.name == "__init__.py":
                continue
            text = py.read_text(encoding="utf-8")
            orig = text
            text = fix_sys_path(text)
            text = rewrite_text(text)
            if text != orig:
                py.write_text(text, encoding="utf-8")
                print(f"patched {sub}/{py.name}")

    # Tests
    tests = ROOT / "tests"
    if tests.exists():
        for t in tests.rglob("*.py"):
            text = t.read_text(encoding="utf-8")
            orig = text
            text = rewrite_text(text)
            if text != orig:
                t.write_text(text, encoding="utf-8")
                print(f"patched test {t.name}")

    # Makefile
    mk = ROOT / "Makefile"
    if mk.exists():
        text = mk.read_text(encoding="utf-8")
        orig = text
        for name, sub in name_to_sub.items():
            text = text.replace(f"prototype/{name}.py", f"prototype/{sub}/{name}.py")
        if text != orig:
            mk.write_text(text, encoding="utf-8")
            print("patched Makefile")

    # Compatibility shims at old flat paths
    for name, sub in name_to_sub.items():
        shim_path = PROTO / f"{name}.py"
        # only write shim if real file is not still at top level
        real = PROTO / sub / f"{name}.py"
        if real.exists():
            shim_path.write_text(SHIM.format(sub=sub, name=name), encoding="utf-8")
            print(f"shim {name}.py")

    (PROTO / "__init__.py").write_text(
        '"""NSA prototype experiments and demos."""\n', encoding="utf-8"
    )

    readme = PROTO / "README.md"
    readme.write_text(
        """# NSA Prototype Folder

Research scripts, benchmarks, and demos. Prefer the subfolder paths; top-level
`*.py` files are thin compatibility shims.

## Layout

| Subfolder | Contents |
|-----------|----------|
| `pillars/` | Pillar 1–4 validation benches (pretrain, GPU attn, LoRA, prompt injection) |
| `security/` | Leakage, multi-tier lattice, NL red-team, multi-probe |
| `retrofit/` | Open-LLM / HF retrofit, Llama showcase, evolution & native-vs-retrofit |
| `experiments/` | Toy LM, dynamic trade-off, ablations, coupling sweeps |
| `demos/` | Web UI, attention visualizer, showcase eval harness |
| `reporting/` | Benchmark report generator |
| `results/` | JSON/HTML outputs (git-ignore-friendly) |

## Examples

```bash
python prototype/experiments/toy_experiment.py
python prototype/security/nl_redteam_suite.py
python prototype/retrofit/hf_nsa_retrofit.py --model sshleifer/tiny-gpt2
make pillar-4
make tradeoff
```

`requirements.txt` stays at `prototype/requirements.txt`.
""",
        encoding="utf-8",
    )
    print("wrote prototype/README.md")

    print("\nFinal layout:")
    for sub in sorted(LAYOUT):
        files = sorted(
            p.name
            for p in (PROTO / sub).glob("*")
            if p.is_file() and p.name != "__init__.py"
        )
        print(f"  {sub}/: {files}")


if __name__ == "__main__":
    main()
