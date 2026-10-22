import json

with open("data/concept_dictionary.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print(f"Total concepts currently: {len(d)}")
print("\nNewly Discovered Concepts (31 to 45):")
for i, c in enumerate(d[30:], start=31):
    print(f"{i:2d}. **{c['name']}**: {c.get('definition', '')}")
