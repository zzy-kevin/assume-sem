import json

with open("visualizations/kmeans_k5_summary.json", "r", encoding="utf-8") as f:
    d = json.load(f)

for cls, clusters in d.items():
    print(f"\n==================== {cls} (K=5) ====================")
    for c in clusters:
        print(f"  Cluster {c['cluster_id']} (N={c['size']:2d}):")
        print(f"    Top Positive: {c['top_positive_concept']} (score = {c['top_positive_score']:.1f}, delta = {c['top_positive_delta']:+.2f})")
        print(f"    Top Negative: {c['top_negative_concept']} (score = {c['top_negative_score']:.1f}, delta = {c['top_negative_delta']:+.2f})")
        top_acts = ", ".join([f"{x['concept'][:25]}={x['score']:.1f}" for x in c['top_active_concepts']])
        print(f"    Top Active Overall: {top_acts}")
