"""Blind Gemma-4 descriptions of SEM archetypes at the OER-correlation extremes.

The selection and overlay rendering intentionally mirror archetypal/viewer/index.html:
the shared dictionary, the preferred synthesis-union filter, and the viewer's
Viridis overlay at opacity 0.75. Gemma receives no correlation or process metadata.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
VIEWER = HERE / "viewer"
DEFAULT_OUT = HERE / "vlm_overpotential_gemma4_12b_context"
DEFAULT_MODEL = Path(r"D:\UOFT\antigravity_projects\chexpert_vlm_rag\models\gemma-4-12b-it-Q4_K_M.gguf")
DEFAULT_MMPROJ = Path(r"D:\UOFT\antigravity_projects\chexpert_vlm_rag\models\mmproj-BF16.gguf")

VIRIDIS = np.asarray(
    [
        [68, 1, 84],
        [71, 44, 122],
        [59, 81, 139],
        [44, 113, 142],
        [33, 144, 141],
        [39, 173, 129],
        [92, 200, 99],
        [170, 220, 50],
        [253, 231, 37],
    ],
    dtype=np.float32,
)

SYSTEM_PROMPT = "You are an expert oxygen evolution reaction (OER) electrocatalyst experimentalist and scanning electron microscopist. Translate learned activation patterns into precise, reproducible SEM morphology concepts. Use your OER-catalyst domain knowledge to recognize relevant surface structures and interfaces, but report only morphology supported by the images; never infer composition, synthesis, mechanism, or catalytic performance from appearance."

USER_PROMPT = """The ten images below are the strongest SEM examples of one visual concept learned by a Relaxed Archetypal Sparse Autoencoder from contextual vision-model patch representations. Each image contains a semi-transparent activation overlay: brighter yellow or yellow-green indicates where evidence for the concept is strongest, while dark purple or transparent regions provide weaker evidence.

The activation overlay is coarse concept localization, not a pixel segmentation mask and not an exact object boundary. The learned concept can represent a structure, texture, boundary, valley, interface, or a relationship between the activated feature and its local surroundings. Therefore inspect both the strongly activated region and the immediately adjacent morphology needed to interpret what that region represents. For example, if activation follows dark crack valleys, identify the crack network as the archetype focus and separately describe the plates or domains that the cracks surround and divide. Do not automatically name the most visually dominant object as the archetype; decide what the activation is actually centered on.

Across all ten images, infer the recurring concept from evidence repeated in multiple examples rather than from a striking feature seen once. Use observable, geometry-based SEM language: particle or feature shape, approximate relative scale, aspect ratio, edge character, surface texture, packing, porosity, connectivity, aggregation, layering, branching, faceting, cracking, and spatial organization. Ignore borders, annotations, scale bars, charging or detector artefacts, and the artificial overlay colour itself. Do not infer chemical composition, material identity, crystal phase, synthesis conditions, fabrication history, electrochemical mechanism, or OER performance. Do not describe the ten images separately; synthesize the shared concept and its perceptual field.

Return exactly these six fields in plain text:
Archetype concept: a concise noun phrase naming the activated feature or target-context relationship
Targeted region: two or three sentences stating explicitly what the high-activation region is centered on and describing its visible geometry and texture
Surrounding morphology: two or three sentences describing the immediately adjacent structures, including their shape, approximate relative scale, texture, packing, and spatial organization
Target-surrounding relationship: one or two sentences explaining how the targeted feature is positioned relative to, separates, covers, bridges, borders, emerges from, or otherwise relates to the surroundings
Evidence across images: one sentence stating how consistently this interpretation appears and the meaningful variants
Uncertainty: one sentence naming genuine ambiguity about the concept focus or heterogeneity without guessing"""


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_passes(folds: list[float | None], max_spread: float = 0.2) -> bool:
    finite = [float(value) for value in folds if value is not None and math.isfinite(value)]
    return len(finite) == 5 and max(finite) - min(finite) <= max_spread + 1e-12


def select_extremes(count: int = 5) -> tuple[list[dict], dict]:
    meta = load_json(VIEWER / "index.json")
    params = load_json(VIEWER / "params.json")
    split = load_json(VIEWER / "five_split_stability.json")
    items = meta["items"]
    if meta["tag"] != "ttaavg_c4096" or params["tag"] != meta["tag"] or split["tag"] != meta["tag"]:
        raise RuntimeError("The shared viewer exports are not mutually aligned.")
    if split["concept_ids"] != [item["c"] for item in items]:
        raise RuntimeError("Five-split concept order does not match index.json.")
    parameter_by_key = {parameter["key"]: parameter for parameter in params["params"]}
    split_by_key = {parameter["key"]: parameter["rho"] for parameter in split["params"]}
    oer = next(parameter for parameter in params["params"] if parameter["group"] == "target")
    synthesis_keys = [parameter["key"] for parameter in params["params"] if parameter["group"] != "target"]
    eligible = []
    for index, item in enumerate(items):
        fire_pass = float(item["fire"]) >= 1e-3
        border_pass = item.get("bor") is None or float(item["bor"]) <= 1.0
        witnesses = [key for key in synthesis_keys if split_passes(split_by_key[key][index])]
        if fire_pass and border_pass and witnesses:
            eligible.append(
                {
                    "concept": int(item["c"]),
                    "rho": float(oer["rho"][index]),
                    "fire_rate": float(item["fire"]),
                    "witness_parameters": witnesses,
                }
            )
    if len(eligible) < 2 * count:
        raise RuntimeError(f"Only {len(eligible)} archetypes pass the preferred filter.")
    high = sorted(eligible, key=lambda row: (-row["rho"], row["concept"]))[:count]
    low = sorted(eligible, key=lambda row: (row["rho"], row["concept"]))[:count]
    for rank, row in enumerate(high, 1):
        row.update(group="high_positive", rank=rank)
    for rank, row in enumerate(low, 1):
        row.update(group="low_negative", rank=rank)
    selection = high + low
    provenance = {
        "dictionary": "shared",
        "tag": meta["tag"],
        "target": oer["key"],
        "target_label": oer["label"],
        "eligible_count": len(eligible),
        "selection_rule": f"top {count} and bottom {count} signed OER Spearman rho among preferred-filter eligible archetypes",
        "preferred_filter": {
            "minimum_fire_rate": 1e-3,
            "maximum_border_share": 1.0,
            "minimum_correlation_magnitude": 0.0,
            "minimum_split_magnitude": 0.0,
            "maximum_five_split_signed_rho_spread": 0.2,
            "collapse_families": False,
            "stability_scope": "union across non-target synthesis parameters",
        },
    }
    return selection, provenance


def viridis_rgb(values: np.ndarray) -> np.ndarray:
    positions = np.clip(values, 0.0, 1.0) * (len(VIRIDIS) - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(VIRIDIS) - 1)
    fraction = (positions - lower)[..., None]
    return VIRIDIS[lower] + (VIRIDIS[upper] - VIRIDIS[lower]) * fraction


def render_overlay(base_path: Path, activation: np.ndarray, output_path: Path, opacity: float = 0.75) -> None:
    with Image.open(base_path) as source:
        base = source.convert("RGB").resize((960, 592), Image.Resampling.BILINEAR)
    values = activation.astype(np.float32) / 255.0
    colours = np.clip(viridis_rgb(values), 0, 255).astype(np.uint8)
    alpha = np.clip(np.power(values, 0.7) * 255.0 * opacity, 0, 255).astype(np.uint8)
    heatmap = Image.fromarray(np.dstack([colours, alpha]), mode="RGBA").resize(base.size, Image.Resampling.BILINEAR)
    rendered = Image.alpha_composite(base.convert("RGBA"), heatmap).convert("RGB")
    rendered.save(output_path, format="JPEG", quality=95, subsampling=0)


def make_contact_sheet(image_paths: list[Path], names: list[str], output_path: Path) -> None:
    thumb_size = (480, 296)
    label_height = 30
    sheet = Image.new("RGB", (thumb_size[0] * 2, (thumb_size[1] + label_height) * 5), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for index, (path, name) in enumerate(zip(image_paths, names)):
        with Image.open(path) as source:
            thumbnail = source.convert("RGB").resize(thumb_size, Image.Resampling.LANCZOS)
        x = (index % 2) * thumb_size[0]
        y = (index // 2) * (thumb_size[1] + label_height)
        sheet.paste(thumbnail, (x, y))
        draw.text((x + 8, y + thumb_size[1] + 6), f"Image {index + 1}: {name}", fill="black", font=font)
    sheet.save(output_path, format="JPEG", quality=94, subsampling=0)


def prepare_inputs(selection: list[dict], out_dir: Path) -> list[dict]:
    input_root = out_dir / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    prepared = []
    for row in selection:
        concept = row["concept"]
        concept_dir = input_root / f"archetype_{concept}"
        concept_dir.mkdir(exist_ok=True)
        mapping = load_json(VIEWER / "maps" / f"{concept}.json")
        if len(mapping["img"]) != 10:
            raise RuntimeError(f"Archetype {concept} has {len(mapping['img'])} references, expected 10.")
        with Image.open(VIEWER / "maps" / f"{concept}.png") as sprite_source:
            sprite = np.asarray(sprite_source.convert("L"), dtype=np.uint8)
        expected_shape = (mapping["gh"] * 10, mapping["gw"])
        if sprite.shape != expected_shape:
            raise RuntimeError(f"Archetype {concept} sprite is {sprite.shape}, expected {expected_shape}.")
        image_paths = []
        for index, image_id in enumerate(mapping["img"]):
            destination = concept_dir / f"{index + 1:02d}.jpg"
            activation = sprite[index * mapping["gh"] : (index + 1) * mapping["gh"], :]
            render_overlay(VIEWER / "img" / f"{image_id}.jpg", activation, destination)
            image_paths.append(destination)
        contact_sheet = concept_dir / "contact_sheet.jpg"
        make_contact_sheet(image_paths, mapping["name"], contact_sheet)
        prepared.append(
            {
                **row,
                "image_paths": [str(path.resolve()) for path in image_paths],
                "source_image_ids": mapping["img"],
                "source_names": mapping["name"],
                "activation_peaks": mapping["peak"],
                "contact_sheet": str(contact_sheet.resolve()),
            }
        )
        print(f"prepared archetype {concept:+d} rho={row['rho']:+.3f}", flush=True)
    return prepared


def image_to_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def build_messages(image_paths: list[Path]) -> list[dict]:
    content = [{"type": "text", "text": USER_PROMPT}]
    for index, image_path in enumerate(image_paths, 1):
        content.append({"type": "text", "text": f"Image {index} of 10:"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}})
    content.append({"type": "text", "text": "Now synthesize the learned morphology concept and its surrounding perceptual field using the required six-field format."})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def run_inference(records: list[dict], model_path: Path, mmproj_path: Path, out_dir: Path) -> None:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Gemma4ChatHandler

    print(f"loading Gemma-4 12B from {model_path}", flush=True)
    started = time.time()
    handler = Gemma4ChatHandler(clip_model_path=str(mmproj_path), verbose=False)
    llm = Llama(
        model_path=str(model_path),
        chat_handler=handler,
        n_ctx=4096,
        n_gpu_layers=-1,
        main_gpu=0,
        split_mode=0,
        n_batch=512,
        n_ubatch=512,
        seed=42,
        verbose=False,
    )
    print(f"model loaded in {time.time() - started:.1f}s", flush=True)
    results_path = out_dir / "results.json"
    completed = {}
    if results_path.exists():
        completed = {int(row["concept"]): row for row in load_json(results_path)}
    for position, record in enumerate(records, 1):
        concept = record["concept"]
        if concept in completed and completed[concept].get("vlm_response"):
            print(f"[{position}/10] archetype {concept}: existing response retained", flush=True)
            continue
        llm.reset()
        messages = build_messages([Path(path) for path in record["image_paths"]])
        print(f"[{position}/10] archetype {concept} rho={record['rho']:+.3f}: inference", flush=True)
        inference_started = time.time()
        try:
            response = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.1)
            raw = response["choices"][0]["message"]["content"]
            error = None
        except Exception as exc:
            raw = ""
            error = f"{type(exc).__name__}: {exc}"
        completed[concept] = {
            **record,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT,
            "model": "gemma-4-12b-it-Q4_K_M.gguf",
            "temperature": 0.1,
            "seed": 42,
            "context_cleared_before_call": True,
            "elapsed_seconds": round(time.time() - inference_started, 2),
            "vlm_response": raw,
            "error": error,
        }
        ordered = [completed[row["concept"]] for row in records if row["concept"] in completed]
        with results_path.open("w", encoding="utf-8") as handle:
            json.dump(ordered, handle, indent=2, ensure_ascii=False)
        print(f"[{position}/10] archetype {concept}: {completed[concept]['elapsed_seconds']:.1f}s", flush=True)


def markdown_report(records: list[dict], provenance: dict, out_dir: Path) -> None:
    lines = [
        "# Blind Gemma-4 12B concept-and-context descriptions of OER-associated archetypes",
        "",
        "## Protocol",
        "",
        f"The shared 4,096-archetype viewer dictionary was filtered with the viewer's preferred synthesis-union settings. This retained {provenance['eligible_count']} archetypes. The five largest positive signed Spearman correlations with measured overpotential and the five most negative correlations were selected. Negative ρ denotes association with lower overpotential (better OER), while positive ρ denotes association with higher overpotential (worse OER). Selection used correlation only; correlation and performance direction were withheld from Gemma.",
        "",
        "Each archetype was described from its ten strongest reference micrographs. The exact viewer rendering was reproduced at 960 × 592 pixels with the same piecewise Viridis colour map, activation-dependent alpha, and overlay opacity 0.75. The prompt treats activation as coarse evidence for an abstract RA-SAE concept rather than a pixel segmentation mask, and forces the model to distinguish the activation-centered feature from its immediate morphological surroundings and their relationship. This follows the concept-dictionary interpretation in Fel et al., [Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models](https://proceedings.mlr.press/v267/fel25a.html). Gemma-4 12B Q4_K_M and its matched vision projector were loaded once. Before every archetype, the llama context was explicitly reset and a new two-message request containing only that archetype's ten images was constructed.",
        "",
        "## Exact prompt",
        "",
        "System prompt:",
        "",
        "<pre>" + html.escape(SYSTEM_PROMPT) + "</pre>",
        "",
        "User prompt:",
        "",
        "<pre>" + html.escape(USER_PROMPT) + "</pre>",
        "",
        "The user message then contained Image 1 through Image 10 as separate image inputs, followed by: <code>Now synthesize the learned morphology concept and its surrounding perceptual field using the required six-field format.</code>",
        "",
        "## Selected archetypes",
        "",
        "| Group | Rank | Archetype | OER Spearman ρ | Firing rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        group = "High positive" if record["group"] == "high_positive" else "Low / negative"
        lines.append(f"| {group} | {record['rank']} | {record['concept']} | {record['rho']:+.3f} | {record['fire_rate']:.3g} |")
    lines.extend(["", "## Images and blind VLM outputs", ""])
    for record in records:
        group = "High positive correlation with overpotential" if record["group"] == "high_positive" else "Low / negative correlation with overpotential"
        concept = record["concept"]
        contact_relative = Path(record["contact_sheet"]).relative_to(out_dir).as_posix()
        lines.extend(
            [
                f"### {group}: archetype {concept} (ρ = {record['rho']:+.3f})",
                "",
                "The contact sheet below is assembled from the ten exact overlaid JPEG inputs passed separately to Gemma, in the same order. The full-resolution input files are linked underneath.",
                "",
                f"![Ten overlaid SEM inputs for archetype {concept}]({contact_relative})",
                "",
                "Exact VLM inputs: " + " · ".join(f"[Image {index}]({Path(path).relative_to(out_dir).as_posix()})" for index, path in enumerate(record["image_paths"], 1)),
                "",
                "Gemma-4 12B response:",
                "",
                "<pre>" + html.escape(record.get("vlm_response") or record.get("error") or "No response") + "</pre>",
                "",
                f"Inference time: {record.get('elapsed_seconds', 0):.1f} seconds. Context reset before call: yes.",
                "",
            ]
        )
    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mmproj", type=Path, default=DEFAULT_MMPROJ)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selection, provenance = select_extremes()
    records = prepare_inputs(selection, args.out)
    with (args.out / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump({"provenance": provenance, "records": records}, handle, indent=2, ensure_ascii=False)
    print("selection:", flush=True)
    for row in records:
        print(f"  {row['group']:13s} rank={row['rank']} archetype={row['concept']:4d} rho={row['rho']:+.3f}", flush=True)
    if args.prepare_only:
        return
    if not args.model.is_file() or not args.mmproj.is_file():
        raise FileNotFoundError("Gemma model or matched vision projector is missing.")
    run_inference(records, args.model, args.mmproj, args.out)
    results = load_json(args.out / "results.json")
    markdown_report(results, provenance, args.out)
    print(f"wrote {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
