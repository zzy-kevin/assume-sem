"""Generate resumable, grammar-constrained morphology-theme judgments with local Gemma."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from build_llm_synthesis_viewer_data import build_bundle


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "cross_magnification_synthesis_llm"
DEFAULT_MODEL = Path(r"D:\UOFT\antigravity_projects\chexpert_vlm_rag\models\gemma-4-12b-it-Q4_K_M.gguf")
OUTPUT_DIRECTORY = "responses_gemma_local_json"
CORRELATION_LEVELS = (
    "strong_negative",
    "moderate_negative",
    "weak_negative",
    "neutral",
    "weak_positive",
    "moderate_positive",
    "strong_positive",
    "mixed",
)
UNCERTAINTY_LEVELS = ("low", "medium", "high")

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "synthesis_correlation": {"type": "string", "enum": list(CORRELATION_LEVELS)},
                    "oer_correlation": {"type": "string", "enum": list(CORRELATION_LEVELS)},
                    "uncertainty": {"type": "string", "enum": list(UNCERTAINTY_LEVELS)},
                },
                "required": ["theme", "synthesis_correlation", "oer_correlation", "uncertainty"],
                "additionalProperties": False,
            },
        },
        "uncertain_themes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["themes", "uncertain_themes"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You identify concise visible SEM morphology themes and judge their directional relationships with one synthesis parameter and OER overpotential.

Use only the supplied evidence. Describe associations, not causes or mechanisms. Do not infer composition from morphology. Archetype letters are arbitrary evidence labels with no numerical meaning.

Semantically combine similar morphology across the supplied archetypes, including across magnifications when appropriate. Do not force distinct morphologies into one theme. Return only JSON matching the required schema, with no prose before or after it."""

TASK_PROMPT = """STRUCTURED OUTPUT TASK

Group the supplied archetypes into concise morphology themes. Each theme must be a short morphology phrase, not a sentence. Do not include archetype letters, magnification bands, explanations, or numerical values in a theme name.

For each coherent theme, judge:
- synthesis_correlation: positive means associated with higher values of the synthesis parameter; negative means associated with lower values.
- oer_correlation: positive means associated with higher overpotential; negative means associated with lower overpotential.
- uncertainty: the uncertainty or spread of the judgment after considering correlation magnitudes, confidence intervals, contradictory directions, semantic coherence, and amount of support.

Use strong, moderate, or weak positive/negative labels when direction is coherent enough to rate. Use neutral when evidence coherently indicates little directional relationship. Use mixed when meaningful evidence exists in both directions. Neutral is a valid relationship and is not the same as uncertainty.

Place morphology that cannot be grouped or rated coherently enough on both axes in uncertain_themes as short phrases only. Do not write a conclusion or any additional fields."""


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def structured_messages(prompt_record: dict) -> tuple[str, str, str]:
    evidence = prompt_record["messages"][1]["content"].split("\n\nReturn these sections:", 1)[0].rstrip()
    user_prompt = f"{evidence}\n\n{TASK_PROMPT}"
    schema_text = json.dumps(JSON_SCHEMA, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256((SYSTEM_PROMPT + "\n\n" + user_prompt + "\n\n" + schema_text).encode("utf-8")).hexdigest()
    return SYSTEM_PROMPT, user_prompt, digest


def validate_structured(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"themes", "uncertain_themes"}:
        raise ValueError("Top-level JSON keys must be exactly themes and uncertain_themes")
    themes = value["themes"]
    uncertain = value["uncertain_themes"]
    if not isinstance(themes, list) or not isinstance(uncertain, list):
        raise ValueError("themes and uncertain_themes must be arrays")
    for index, theme in enumerate(themes):
        expected = {"theme", "synthesis_correlation", "oer_correlation", "uncertainty"}
        if not isinstance(theme, dict) or set(theme) != expected:
            raise ValueError(f"Theme {index} has invalid keys")
        if not isinstance(theme["theme"], str) or not theme["theme"].strip():
            raise ValueError(f"Theme {index} has no name")
        if theme["synthesis_correlation"] not in CORRELATION_LEVELS:
            raise ValueError(f"Theme {index} has invalid synthesis_correlation")
        if theme["oer_correlation"] not in CORRELATION_LEVELS:
            raise ValueError(f"Theme {index} has invalid oer_correlation")
        if theme["uncertainty"] not in UNCERTAINTY_LEVELS:
            raise ValueError(f"Theme {index} has invalid uncertainty")
    if any(not isinstance(theme, str) or not theme.strip() for theme in uncertain):
        raise ValueError("Every uncertain theme must be a non-empty string")
    return value


def existing_is_current(path: Path, source_hash: str, structured_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        record = load_json(path)
        validate_structured(record.get("structured_response"))
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return record.get("prompt_sha256") == source_hash and record.get("structured_prompt_sha256") == structured_hash


def load_model(model_path: Path, n_ctx: int):
    from llama_cpp import Llama

    return Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=-1, main_gpu=0, split_mode=0, n_batch=512, n_ubatch=512, seed=42, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--parameters", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    manifest = load_json(args.input / "manifest.json")
    parameters = manifest["parameters"]
    if args.parameters:
        requested = set(args.parameters)
        parameters = [row for row in parameters if row["parameter"] in requested]
        missing = requested - {row["parameter"] for row in parameters}
        if missing:
            raise KeyError(f"Unknown parameters: {sorted(missing)}")
    response_dir = args.input / OUTPUT_DIRECTORY
    response_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for row in parameters:
        prompt_record = load_json(args.input / "prompts" / f"{row['stem']}.json")
        system, user, structured_hash = structured_messages(prompt_record)
        path = response_dir / f"{row['stem']}.json"
        if args.force or not existing_is_current(path, row["prompt_sha256"], structured_hash):
            pending.append((row, system, user, structured_hash, path))
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        build_bundle(args.input)
        print("Gemma local-JSON responses are already complete.")
        return

    from llama_cpp import LlamaGrammar

    grammar = LlamaGrammar.from_json_schema(json.dumps(JSON_SCHEMA, ensure_ascii=False), verbose=False)
    llm = None
    try:
        print(f"Loading {args.model.name} once for {len(pending)} structured prompts...", flush=True)
        llm = load_model(args.model, args.n_ctx)
        for position, (row, system, user, structured_hash, path) in enumerate(pending, 1):
            llm.reset()
            started = time.perf_counter()
            completion = llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=args.max_tokens,
                temperature=0.1,
                grammar=grammar,
            )
            raw = completion["choices"][0]["message"]["content"]
            if not raw:
                raise RuntimeError(f"Gemma returned an empty response for {row['parameter']}")
            structured = validate_structured(json.loads(raw))
            elapsed = time.perf_counter() - started
            record = {
                "schema": 1,
                "parameter": row["parameter"],
                "model": args.model.name,
                "prompt_sha256": row["prompt_sha256"],
                "structured_prompt_sha256": structured_hash,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "elapsed_seconds": round(elapsed, 2),
                "raw_response": raw,
                "structured_response": structured,
            }
            atomic_json(path, record)
            build_bundle(args.input)
            print(f"[{position}/{len(pending)}] {row['parameter']}: {len(structured['themes'])} themes, {len(structured['uncertain_themes'])} uncertain, {elapsed:.1f}s", flush=True)
    finally:
        if llm is not None:
            llm.reset()
            llm.close()
        del llm
        gc.collect()


if __name__ == "__main__":
    main()
