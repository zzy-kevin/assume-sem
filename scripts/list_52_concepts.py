import json

with open("concept_dictionary.json", "r", encoding="utf-8") as f:
    concepts = json.load(f)

print(f"Total concepts successfully discovered and checkpointed: {len(concepts)}\n")
for i, c in enumerate(concepts, start=1):
    tag = "[NEW]" if i > 30 else "[ORIGINAL]"
    print(f"{i:2d}. {tag} {c['name']}")
