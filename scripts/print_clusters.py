import json

with open("visualizations/kmeans_k10_clusters_summary.json", "r", encoding="utf-8") as f:
    d = json.load(f)

for category, clusters in d.items():
    print(f"\n==================== {category} (K=10) ====================")
    for c in clusters:
        print(f"  Cluster {c['cluster_rank']:2d} (N={c['size']:2d}): Top Concept = {c['signature_concept']:<40} (score = {c['signature_score']:.1f})")
