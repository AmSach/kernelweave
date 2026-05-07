from __future__ import annotations

import argparse
import json
from pathlib import Path

from .providers import ModelCatalog, ModelPreset, run_preset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernelweave-models", description="Run prompts against pluggable model presets")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list", help="list model presets")
    list_p.add_argument("--models-dir", type=Path, default=None)

    run_p = sub.add_parser("run", help="run a prompt against a preset")
    run_p.add_argument("preset_id")
    run_p.add_argument("prompt")
    run_p.add_argument("--models-dir", type=Path, default=None)
    run_p.add_argument("--system-prompt", default="")
    run_p.add_argument("--temperature", type=float, default=None)
    run_p.add_argument("--max-tokens", type=int, default=None)

    show_p = sub.add_parser("show", help="show a preset as JSON")
    show_p.add_argument("preset_id")
    show_p.add_argument("--models-dir", type=Path, default=None)

    return parser


def _load_catalog(models_dir: Path | None) -> ModelCatalog:
    if models_dir is None:
        return ModelCatalog.load_default()
    if models_dir.exists():
        return ModelCatalog.from_paths([models_dir])
    return ModelCatalog()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    catalog = _load_catalog(args.models_dir)

    if args.cmd == "list":
        print(json.dumps([preset.to_dict() for preset in catalog.list()], indent=2, sort_keys=True))
        return

    if args.cmd == "show":
        preset = catalog.get(args.preset_id)
        print(json.dumps(preset.to_dict(), indent=2, sort_keys=True))
        return

    if args.cmd == "run":
        preset = catalog.get(args.preset_id)
        response = run_preset(
            args.prompt,
            preset,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(json.dumps(response.to_dict(), indent=2, sort_keys=True))
        return


if __name__ == "__main__":
    main()
