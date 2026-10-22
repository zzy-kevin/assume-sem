import sys
sys.path.insert(0, ".")
import json
import logging
from pathlib import Path
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from assume.agents.proposer import ProposerAgent, ConceptRubric
from assume.data.ingestion import preprocess_micrograph

df = pd.read_parquet("eval_manifest.parquet")
# Load 6 images from different classes
sample_paths = df.groupby("class_label")["full_path"].first().iloc[:6].tolist()
images = [preprocess_micrograph(Path(p)) for p in sample_paths]

with open("concept_dictionary.json", "r", encoding="utf-8") as f:
    existing_concepts = json.load(f)

print(f"Loaded {len(images)} images and {len(existing_concepts)} existing concepts.")

proposer = ProposerAgent()
print("Calling ProposerAgent.propose()...")
new_rubrics = proposer.propose(images=images, existing_concepts=existing_concepts, n_proposals=3)
print(f"Successfully proposed {len(new_rubrics)} concepts:")
for r in new_rubrics:
    print(f"  - {r.name}: {r.definition}")
