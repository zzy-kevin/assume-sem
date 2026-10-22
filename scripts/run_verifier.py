#!/usr/bin/env python3
"""CLI Runner for the Literature Verifier pass.

Loads a concept dictionary and the retrieved literature evidence per concept
(data/verifier/evidence_part*.json, produced with the Valency Bond paper search MCP),
asks the Gemini VerifierAgent to KEEP or PRUNE each concept, and writes:
    data/verifier/verifier_report.json            all decisions with rationale and references
    data/verifier/concept_dictionary_verified.json  kept concepts with a 'references' field

The input concept dictionary is never modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import sys

# Ensure project root is on Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assume.agents.verifier import VerifierAgent
from assume.agents.vertex_client import ResilientVertexClient

logger = logging.getLogger("assume.verifier_runner")

VERIFIER_DIR = PROJECT_ROOT / "data" / "verifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ASSUME literature verifier over a concept dictionary.")
    parser.add_argument("--concepts", type=Path, default=VERIFIER_DIR / "concepts_snapshot.json",
                        help="Concept dictionary JSON to verify (default: data/verifier/concepts_snapshot.json).")
    parser.add_argument("--evidence-glob", type=str, default="evidence_part*.json",
                        help="Glob for evidence files inside data/verifier (default: evidence_part*.json).")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent verifier calls (default: 4).")
    parser.add_argument("--model", type=str, default="gemini-3.8-flash", help="Vertex model id (default: gemini-3.8-flash).")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args()

    concepts = json.loads(args.concepts.read_text(encoding="utf-8"))
    evidence = {}
    for p in sorted(VERIFIER_DIR.glob(args.evidence_glob)):
        evidence.update(json.loads(p.read_text(encoding="utf-8")))
    missing = [c["concept_id"] for c in concepts if c["concept_id"] not in evidence]
    logger.info("Loaded %d concepts, evidence for %d (%d missing).", len(concepts), len(evidence), len(missing))
    if missing:
        logger.warning("No evidence for: %s", missing)

    client = ResilientVertexClient(model=args.model, max_retries=25, base_delay=3.0, max_delay=60.0, timeout=120.0)
    verifier = VerifierAgent(client=client)

    def run_one(c):
        papers = evidence.get(c["concept_id"], {}).get("papers", [])
        try:
            return verifier.verify(c, papers).to_dict()
        except Exception as exc:  # keep the concept if verification fails, and record why
            logger.error("Verifier failed for %s: %s", c["concept_id"], exc)
            return {"concept_id": c["concept_id"], "action": "KEEP", "category": "error",
                    "rationale": f"verification failed: {exc}", "references": []}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        decisions = list(pool.map(run_one, concepts))

    by_id = {d["concept_id"]: d for d in decisions}
    report = []
    verified = []
    for c in concepts:
        d = by_id[c["concept_id"]]
        report.append({"name": c["name"], **d})
        if d["action"] == "KEEP":
            verified.append({**c, "references": d["references"]})

    (VERIFIER_DIR / "verifier_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (VERIFIER_DIR / "concept_dictionary_verified.json").write_text(
        json.dumps(verified, indent=2, ensure_ascii=False), encoding="utf-8")

    pruned = [r for r in report if r["action"] == "PRUNE"]
    logger.info("Kept %d / %d concepts, pruned %d.", len(verified), len(concepts), len(pruned))
    for r in pruned:
        logger.info("  PRUNE [%s] %s: %s", r["category"], r["name"], r["rationale"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
