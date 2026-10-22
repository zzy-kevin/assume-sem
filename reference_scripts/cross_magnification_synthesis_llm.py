"""Build and optionally run concise cross-magnification morphology prompts.

Selection follows the viewer's selected-parameter filter. Each band contributes
up to K strongest positive and negative synthesis associations. Prompts expose
arbitrary letter aliases; dictionary-local concept IDs remain in private maps.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
VIEWER = HERE / "viewer"
DEFAULT_OUTPUT = HERE / "cross_magnification_synthesis_llm"
DEFAULT_MODEL = Path(r"D:\UOFT\antigravity_projects\chexpert_vlm_rag\models\gemma-4-12b-it-Q4_K_M.gguf")
TOP_K = 5

BANDS = (
    ("le400", "≤400×", VIEWER / "bands" / "le400"),
    ("near2000", "2,000×", VIEWER / "bands" / "near2000"),
    ("near10000", "10,000×", VIEWER / "bands" / "near10000"),
    ("near40000", "40,000×", VIEWER / "bands" / "near40000"),
)

SYSTEM_PROMPT = """You summarize associations between synthesis conditions, visible SEM morphology, and OER overpotential.

Use only the supplied evidence. Describe associations, not causes or mechanisms. Do not infer composition from morphology. Treat the archetype letters as arbitrary labels with no numerical meaning.

A morphology theme may appear in more than one magnification band. If it does, note the recurrence as stronger evidence and name the supporting bands. A theme may still be reported when it appears in only one band. Do not call a theme recurrent when its support comes from only one band, and do not merge distinct morphologies merely to create recurrence. Keep higher-value and lower-value morphology separate. State whether each theme is associated with lower overpotential, higher overpotential, or mixed OER directions.

Be concise."""

DESCRIPTION_FIELDS = (
    "Archetype concept",
    "Targeted region",
    "Surrounding morphology",
    "Target-surrounding relationship",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def parse_description(response: str) -> dict[str, str]:
    pattern = re.compile(r"^(Activation evidence|Archetype concept|Targeted region|Surrounding morphology|Target-surrounding relationship):\s*(.*)$")
    fields: dict[str, str] = {}
    current = None
    for line in response.splitlines():
        match = pattern.match(line.strip())
        if match:
            current = match.group(1)
            fields[current] = match.group(2).strip()
        elif current and line.strip():
            fields[current] = f"{fields[current]} {line.strip()}".strip()
    return fields


def alias_for(index: int) -> str:
    value = index + 1
    label = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(65 + remainder) + label
    return label


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "parameter"


def passes_filter(item: dict, overall_rho: float, split_values: list[float | None]) -> bool:
    values = np.asarray([value for value in split_values if finite(value)], dtype=float)
    direction = np.sign(overall_rho)
    border = item.get("bor")
    return (
        abs(float(item["rho"])) > 0
        and (not finite(border) or float(border) < 1)
        and float(item["fire"]) > 1e-3
        and len(values) == 5
        and direction != 0
        and np.all(np.sign(values) == direction)
        and float(np.min(np.abs(values))) > 0.01
        and float(np.ptp(values)) < 0.2
    )


def parameter_context(values: np.ndarray, group: str) -> str:
    unique = int(np.unique(values).size)
    zero_fraction = float(np.mean(values == 0))
    notes = []
    if group == "precursor" or zero_fraction >= 0.5:
        notes.append("This parameter is zero-inflated and primarily distinguishes absence from presence, although the activation-weighted summaries below are continuous. Do not describe the result as a dose response.")
    elif unique <= 5:
        notes.append(f"This parameter has {unique} distinct observed values. Treat it as a categorical or ordinal contrast, not a continuous dose effect.")
    else:
        notes.append("This parameter is continuously valued in this dataset.")
    return " ".join(notes)


def oer_direction(rho: float) -> str:
    if rho < 0:
        return "lower overpotential"
    if rho > 0:
        return "higher overpotential"
    return "no directional overpotential association"


def direct_oer_text(rho: float) -> str:
    if abs(rho) < 0.1:
        direction = "little directional association"
    elif rho < 0:
        direction = "higher values are associated with lower overpotential"
    else:
        direction = "higher values are associated with higher overpotential"
    return f"Spearman ρ = {rho:+.3f}; {direction}."


def band_data(folder: Path, parameter_key: str) -> dict:
    index = load_json(folder / "index.json")
    params = load_json(folder / "params.json")["params"]
    splits = load_json(folder / "five_split_stability.json")
    weighted = load_json(folder / "weighted_synthesis_specimen.json")
    descriptions = load_json(folder / "vlm_descriptions.json")
    items = index["items"]
    concept_ids = [int(item["c"]) for item in items]
    if splits["concept_ids"] != concept_ids or weighted["concept_ids"] != concept_ids:
        raise RuntimeError(f"Unaligned viewer inputs in {folder}")
    parameter = next(row for row in params if row["key"] == parameter_key)
    split = next(row for row in splits["params"] if row["key"] == parameter_key)
    summary = next(row for row in weighted["params"] if row["key"] == parameter_key)
    rows = []
    for i, item in enumerate(items):
        if not passes_filter(item, parameter["rho"][i], split["rho"][i]):
            continue
        mean, low, high = summary["mean"][i], summary["low"][i], summary["high"][i]
        if not all(finite(value) for value in (mean, low, high)):
            continue
        concept = concept_ids[i]
        fields = parse_description(descriptions["descriptions"][str(concept)]["response"])
        if any(not fields.get(field) for field in DESCRIPTION_FIELDS):
            raise RuntimeError(f"Incomplete VLM description for {folder.name} concept {concept}")
        rows.append({
            "concept": concept,
            "synthesis_rho": float(parameter["rho"][i]),
            "oer_rho": float(item["rho"]),
            "weighted_mean": float(mean),
            "ci_low": float(low),
            "ci_high": float(high),
            "fields": {field: fields[field] for field in DESCRIPTION_FIELDS},
        })
    x = np.asarray([row["oer_rho"] for row in rows], dtype=float)
    y = np.asarray([row["weighted_mean"] for row in rows], dtype=float)
    if len(rows) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        profile_r = float(pearsonr(x, y).statistic)
        profile_r2 = profile_r * profile_r
    else:
        profile_r = profile_r2 = None
    baseline = float(summary["specimen_mean"])
    delta = y - baseline
    return {
        "rows": rows,
        "baseline": baseline,
        "observed_min": float(summary["min"]),
        "observed_max": float(summary["max"]),
        "profile_r": profile_r,
        "profile_r2": profile_r2,
        "quadrants": {
            "same_direction": int(np.sum(delta * x > 0)),
            "opposite_direction": int(np.sum(delta * x < 0)),
            "on_axis": int(np.sum(delta * x == 0)),
        },
    }


def select_rows(rows: list[dict], top_k: int) -> tuple[list[dict], list[dict]]:
    per_side = min(top_k, len(rows) // 2)
    ordered = sorted(rows, key=lambda row: (row["weighted_mean"], row["concept"]))
    lower = ordered[:per_side]
    higher = list(reversed(ordered[-per_side:])) if per_side else []
    return higher, lower


def compact_entry(row: dict) -> str:
    fields = row["fields"]
    return (
        f"Archetype {row['alias']}\n"
        f"Activation-weighted value: {row['weighted_mean']:.4g}, 95% interval [{row['ci_low']:.4g}, {row['ci_high']:.4g}]\n"
        f"OER association: {oer_direction(row['oer_rho'])}, ρ = {row['oer_rho']:+.3f}\n"
        f"Concept: {fields['Archetype concept']}\n"
        f"Targeted region: {fields['Targeted region']}\n"
        f"Surrounding morphology: {fields['Surrounding morphology']}\n"
        f"Relationship: {fields['Target-surrounding relationship']}"
    )


def format_group(title: str, selections: dict[str, list[dict]]) -> str:
    blocks = [title]
    for _, band_label, _ in BANDS:
        rows = selections[band_label]
        blocks.append(f"MAGNIFICATION {band_label}")
        blocks.append("\n\n".join(compact_entry(row) for row in rows) if rows else "No eligible archetypes.")
    return "\n\n".join(blocks)


def profile_line(label: str, data: dict) -> str:
    if data["profile_r"] is None:
        return f"{label}: unavailable; {len(data['rows'])} eligible archetypes."
    return f"{label}: r = {data['profile_r']:+.3f}, R² = {data['profile_r2']:.3f}, based on {len(data['rows'])} eligible archetypes."


def build_prompt(label: str, context: str, direct_rho: float, bands: dict[str, dict], higher: dict[str, list[dict]], lower: dict[str, list[dict]]) -> str:
    profiles = "\n".join(profile_line(band_label, bands[band_label]) for _, band_label, _ in BANDS)
    return f"""SYNTHESIS PARAMETER
{label}

PARAMETER CONTEXT
{context}

OVERALL RESULTS
Specimen-level association with OER: {direct_oer_text(direct_rho)}
For each magnification below, r and R² describe the linear relationship between activation-weighted synthesis value and archetype–OER ρ. These are descriptive archetype-profile statistics, not prediction scores.
{profiles}

The activation-weighted values below are continuous summaries. Their intervals show uncertainty and were not used to select archetypes.

{format_group('HIGHER-VALUE ASSOCIATED ARCHETYPES', higher)}

{format_group('LOWER-VALUE ASSOCIATED ARCHETYPES', lower)}

Return these sections:

## Overall relationship
Summarize the overall morphology–OER alignment in two or three sentences.

## Higher-value morphology
Summarize the supplied morphology themes and their OER directions. Note themes recurring across magnification bands as stronger evidence.

## Lower-value morphology
Use the same approach.

## Mixed evidence
Briefly state contradictions, inconsistent OER directions, scale-specific observations, and missing evidence.

## Conclusion
Give one cautious sentence describing what morphology is associated with this synthesis parameter and whether that morphology has a consistent relationship with overpotential."""


def assign_aliases(higher: dict[str, list[dict]], lower: dict[str, list[dict]]) -> list[dict]:
    aliases = []
    index = 0
    for direction, groups in (("higher", higher), ("lower", lower)):
        for band_key, band_label, _ in BANDS:
            for row in groups[band_label]:
                row["alias"] = alias_for(index)
                aliases.append({
                    "alias": row["alias"],
                    "direction": direction,
                    "band": band_key,
                    "band_label": band_label,
                    "concept": row["concept"],
                    "synthesis_rho": row["synthesis_rho"],
                    "oer_rho": row["oer_rho"],
                    "weighted_mean": row["weighted_mean"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                })
                index += 1
    return aliases


def plot_parameter(path: Path, label: str, bands: dict[str, dict], aliases: list[dict]) -> None:
    alias_lookup = {(row["band_label"], row["concept"]): row["alias"] for row in aliases}
    all_rows = [row for data in bands.values() for row in data["rows"]]
    xmax = max((abs(row["oer_rho"]) for row in all_rows), default=0.25) * 1.12
    ymin = min((data["observed_min"] for data in bands.values()), default=0)
    ymax = max((data["observed_max"] for data in bands.values()), default=1)
    ypad = max((ymax - ymin) * 0.05, 1e-8)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharex=True, sharey=True)
    for ax, (_, band_label, _) in zip(axes.flat, BANDS):
        data = bands[band_label]
        for row in data["rows"]:
            alias = alias_lookup.get((band_label, row["concept"]))
            selected = next((entry for entry in aliases if entry["band_label"] == band_label and entry["concept"] == row["concept"]), None)
            colour = "#eb6834" if selected and selected["direction"] == "higher" else "#2a78d6" if selected else "#969da2"
            alpha = 0.95 if alias else 0.42
            ax.errorbar(row["oer_rho"], row["weighted_mean"], yerr=[[row["weighted_mean"] - row["ci_low"]], [row["ci_high"] - row["weighted_mean"]]], fmt="o", ms=4.5 if alias else 3.5, color=colour, ecolor=colour, alpha=alpha, elinewidth=0.7, capsize=1.5)
            if alias:
                ax.annotate(alias, (row["oer_rho"], row["weighted_mean"]), xytext=(4, 3), textcoords="offset points", fontsize=7, color=colour, weight="bold")
        ax.axvline(0, color="#444", lw=0.8, ls="--")
        ax.axhline(data["baseline"], color="#444", lw=0.8, ls=":")
        metric = "unavailable" if data["profile_r"] is None else f"r={data['profile_r']:+.2f}, R²={data['profile_r2']:.2f}, n={len(data['rows'])}"
        ax.set_title(f"{band_label}  |  {metric}")
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.grid(alpha=0.12)
    for ax in axes[:, 0]:
        ax.set_ylabel("Activation-weighted synthesis value")
    for ax in axes[-1, :]:
        ax.set_xlabel("Archetype–OER Spearman ρ")
    fig.suptitle(f"{label}: cross-magnification morphology profile\nOrange/blue = selected higher/lower-value archetypes; dotted line = specimen mean", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def eligible_parameters() -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    raw = load_json(VIEWER / "specimens.json")
    metadata = load_json(VIEWER / "params.json")["params"]
    values = {row["key"]: np.asarray(row["values"], dtype=float) for row in raw["params"]}
    oer = values["lsv_overpotential_mV"]
    selected = []
    for row in metadata:
        if row["group"] == "target":
            continue
        nonzero = int(np.sum(values[row["key"]] != 0))
        if row["group"] in {"process", "descriptor"} or (row["group"] == "precursor" and nonzero > 30):
            selected.append({**row, "nonzero_specimens": nonzero})
    return selected, values, oer


def validate_response(text: str, valid_aliases: set[str]) -> list[str]:
    mentioned = set(re.findall(r"\bArchetype\s+([A-Z]{1,2})\b", text))
    return sorted(mentioned - valid_aliases)


def write_index(output: Path, manifest: list[dict]) -> None:
    metrics_path = output / "profile_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "label", "group", "nonzero_specimens", "direct_oer_rho", "band", "eligible_archetypes", "profile_r", "profile_r2", "same_direction", "opposite_direction", "on_axis"])
        for row in manifest:
            for band, metrics in row["bands"].items():
                writer.writerow([row["parameter"], row["label"], row["group"], row["nonzero_specimens"], row["direct_oer_rho"], band, metrics["eligible"], metrics["profile_r"], metrics["profile_r2"], metrics["quadrants"]["same_direction"], metrics["quadrants"]["opposite_direction"], metrics["quadrants"]["on_axis"]])
    lines = [
        "# Cross-magnification synthesis–morphology prompts",
        "",
        f"This directory contains {len(manifest)} byte-stable prompts: every processing condition and descriptor, plus bath precursors with strictly more than 30 nonzero specimens. Selection uses the viewer's parameter-specific reliability filter, then takes up to five non-overlapping highest and lowest activation-weighted synthesis values per magnification band. Confidence intervals are displayed but never used for selection. Families are not collapsed.",
        "",
        "Each prompt uses arbitrary letters instead of dictionary-local IDs. The private alias map preserves traceability. Profile r and R² are calculated across every eligible archetype before top-k selection. Local responses use Gemma-4 12B Q4_K_M at temperature 0.1. Alias validation checks that every explicit `Archetype X` citation was supplied; it does not judge semantic quality.",
        "",
        "The easiest larger-model input is the `.txt` prompt. `prompts.jsonl` contains the same system/user messages for API use.",
        "",
        "Machine-readable summaries: [profile metrics](profile_metrics.csv) · [response QA](response_quality.csv)",
        "",
        "| Parameter | Group | Nonzero | Direct OER ρ | Prompt | Plot | Alias map | Local response |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for row in manifest:
        response = output / "responses" / f"{row['stem']}.md"
        response_link = f"[Gemma](responses/{row['stem']}.md)" if response.exists() else "pending"
        label = str(row["label"]).replace("|", "\\|")
        lines.append(f"| {label} | {row['group']} | {row['nonzero_specimens']} | {row['direct_oer_rho']:+.3f} | [text](prompts/{row['stem']}.txt) · [JSON](prompts/{row['stem']}.json) | [profile](plots/{row['stem']}.png) | [private map](alias_maps/{row['stem']}.json) | {response_link} |")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_responses(output: Path, manifest: list[dict]) -> list[dict]:
    headings = ("## Overall relationship", "## Higher-value morphology", "## Lower-value morphology", "## Mixed evidence", "## Conclusion")
    band_tokens = ("≤400×", "2,000×", "10,000×", "40,000×")
    recurrence_terms = ("recurrent", "across multiple scales", "across multiple magnification", "across different scales")
    rows = []
    for item in manifest:
        path = output / "responses" / f"{item['stem']}.json"
        if not path.exists():
            rows.append({"parameter": item["parameter"], "response_present": False, "hash_match": False, "invalid_aliases": [], "missing_headings": list(headings), "ends_cleanly": False, "recurrence_lines_needing_review": []})
            continue
        record = load_json(path)
        text = record.get("response", "")
        recurrence = []
        for line in text.splitlines():
            lowered = line.lower()
            if any(term in lowered for term in recurrence_terms) and sum(token in line for token in band_tokens) < 2:
                recurrence.append(line.strip())
        rows.append({
            "parameter": item["parameter"],
            "response_present": bool(text),
            "hash_match": record.get("prompt_sha256") == item["prompt_sha256"],
            "invalid_aliases": record.get("invalid_aliases", []),
            "missing_headings": [heading for heading in headings if heading not in text],
            "ends_cleanly": bool(re.search(r"[.!?)]$", text.strip())),
            "response_characters": len(text),
            "recurrence_lines_needing_review": recurrence,
        })
    (output / "response_quality.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "response_quality.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "response_present", "hash_match", "invalid_alias_count", "missing_heading_count", "ends_cleanly", "response_characters", "recurrence_lines_needing_review"])
        for row in rows:
            writer.writerow([row["parameter"], row["response_present"], row["hash_match"], len(row["invalid_aliases"]), len(row["missing_headings"]), row["ends_cleanly"], row.get("response_characters", 0), len(row["recurrence_lines_needing_review"])])
    return rows


def load_model(model_path: Path, n_ctx: int):
    from llama_cpp import Llama

    return Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=-1, main_gpu=0, split_mode=0, n_batch=512, n_ubatch=512, seed=42, verbose=False)


def run_local(llm, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    llm.reset()
    response = llm.create_chat_completion(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], max_tokens=max_tokens, temperature=0.1)
    text = response["choices"][0]["message"]["content"]
    if not text:
        raise RuntimeError("Gemma returned an empty response")
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--parameters", nargs="*")
    parser.add_argument("--run-local", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=1100)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if args.run_local and not args.model.is_file():
        raise FileNotFoundError(args.model)
    for name in ("prompts", "alias_maps", "plots", "responses"):
        (args.output / name).mkdir(parents=True, exist_ok=True)

    parameters, values, oer = eligible_parameters()
    if args.parameters:
        requested = set(args.parameters)
        parameters = [row for row in parameters if row["key"] in requested]
        missing = requested - {row["key"] for row in parameters}
        if missing:
            raise KeyError(f"Unavailable or ineligible parameters: {sorted(missing)}")

    manifest = []
    jsonl = []
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for number, parameter in enumerate(parameters, 1):
        key = parameter["key"]
        label = parameter.get("label", key)
        bands = {}
        higher: dict[str, list[dict]] = {}
        lower: dict[str, list[dict]] = {}
        for _, band_label, folder in BANDS:
            data = band_data(folder, key)
            bands[band_label] = data
            higher[band_label], lower[band_label] = select_rows(data["rows"], args.top_k)
        aliases = assign_aliases(higher, lower)
        direct_rho = float(spearmanr(values[key], oer).statistic)
        context = parameter_context(values[key], parameter["group"])
        user_prompt = build_prompt(label, context, direct_rho, bands, higher, lower)
        prompt_hash = hashlib.sha256((SYSTEM_PROMPT + "\n\n" + user_prompt).encode("utf-8")).hexdigest()
        stem = f"{number:02d}_{safe_name(key)}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        prompt_record = {
            "schema": 1,
            "parameter": key,
            "label": label,
            "group": parameter["group"],
            "nonzero_specimens": parameter["nonzero_specimens"],
            "prompt_sha256": prompt_hash,
            "messages": messages,
        }
        (args.output / "prompts" / f"{stem}.json").write_text(json.dumps(prompt_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.output / "prompts" / f"{stem}.txt").write_text(f"SYSTEM\n{SYSTEM_PROMPT}\n\nUSER\n{user_prompt}\n", encoding="utf-8")
        alias_record = {"parameter": key, "prompt_sha256": prompt_hash, "aliases": aliases}
        (args.output / "alias_maps" / f"{stem}.json").write_text(json.dumps(alias_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        plot_parameter(args.output / "plots" / f"{stem}.png", label, bands, aliases)
        manifest.append({
            "number": number,
            "stem": stem,
            "parameter": key,
            "label": label,
            "group": parameter["group"],
            "nonzero_specimens": parameter["nonzero_specimens"],
            "direct_oer_rho": direct_rho,
            "prompt_sha256": prompt_hash,
            "selected_archetypes": len(aliases),
            "bands": {band_label: {key2: data[key2] for key2 in ("profile_r", "profile_r2", "quadrants")} | {"eligible": len(data["rows"])} for band_label, data in bands.items()},
        })
        jsonl.append(json.dumps({"parameter": key, "prompt_sha256": prompt_hash, "messages": messages}, ensure_ascii=False))

    (args.output / "prompts.jsonl").write_text("\n".join(jsonl) + "\n", encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps({"schema": 1, "generated_at": generated_at, "top_k": args.top_k, "precursor_rule": "strictly more than 30 nonzero specimens", "system_prompt": SYSTEM_PROMPT, "parameters": manifest}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_responses(args.output, manifest)
    write_index(args.output, manifest)
    print(f"Generated {len(manifest)} prompts, alias maps, and plots in {args.output}", flush=True)

    if not args.run_local:
        return
    llm = None
    try:
        print(f"Loading {args.model.name} once for {len(manifest)} prompts...", flush=True)
        llm = load_model(args.model, args.n_ctx)
        for completed, row in enumerate(manifest, 1):
            response_json = args.output / "responses" / f"{row['stem']}.json"
            response_md = args.output / "responses" / f"{row['stem']}.md"
            if response_json.exists() and not args.force:
                previous = load_json(response_json)
                if previous.get("prompt_sha256") == row["prompt_sha256"] and previous.get("response"):
                    print(f"[{completed}/{len(manifest)}] {row['parameter']}: existing response", flush=True)
                    continue
            prompt = load_json(args.output / "prompts" / f"{row['stem']}.json")
            started = time.perf_counter()
            response = run_local(llm, prompt["messages"][0]["content"], prompt["messages"][1]["content"], args.max_tokens)
            elapsed = time.perf_counter() - started
            aliases = load_json(args.output / "alias_maps" / f"{row['stem']}.json")["aliases"]
            invalid = validate_response(response, {entry["alias"] for entry in aliases})
            record = {
                "schema": 1,
                "parameter": row["parameter"],
                "model": args.model.name,
                "prompt_sha256": row["prompt_sha256"],
                "elapsed_seconds": round(elapsed, 2),
                "invalid_aliases": invalid,
                "response": response,
            }
            response_json.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            response_md.write_text(response + "\n", encoding="utf-8")
            print(f"[{completed}/{len(manifest)}] {row['parameter']}: {elapsed:.1f}s, invalid aliases={invalid}", flush=True)
        audit_responses(args.output, manifest)
        write_index(args.output, manifest)
    finally:
        if llm is not None:
            llm.reset()
            llm.close()
        del llm
        gc.collect()


if __name__ == "__main__":
    main()
