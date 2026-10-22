import json

with open("concept_dictionary.json", "r", encoding="utf-8") as f:
    concepts = json.load(f)

print(f"Loaded {len(concepts)} concepts:\n")
for i, c in enumerate(concepts, start=1):
    cid = c.get("concept_id", f"C{i:02d}")
    name = c.get("name", "Unnamed")
    defn = c.get("definition", c.get("description", ""))
    s1 = c.get("scale_1", "")
    s5 = c.get("scale_5", "")
    s10 = c.get("scale_10", "")
    print(f"| {i} | `{cid}` | **{name}** | {defn} |")
