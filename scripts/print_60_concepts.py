import json

with open("concept_dictionary.json", "r", encoding="utf-8") as f:
    concepts = json.load(f)

print(f"Total concepts: {len(concepts)}\n")
for i, c in enumerate(concepts, start=1):
    tag = "[ORIGINAL]" if i <= 30 else "[NEW]"
    print(f"| {i:2d} | {tag} | **{c['name']}** | {c.get('definition', '')} |")
