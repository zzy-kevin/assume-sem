import json

with open("visualizations/latent_correlations_summary.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("=== GLOBAL PCA (Variance: PC1={:.1f}%, PC2={:.1f}%) ===".format(
    d["global_pca_variance_ratio"][0] * 100, d["global_pca_variance_ratio"][1] * 100
))
print("\nGlobal PC1 Top Correlations:")
for x in d["global_top_pc1"][:5]:
    print(f"  - {x['concept_name']}: r = {x['pearson_r']:+.3f} (p={x['pearson_pval']:.2e})")

print("\nGlobal PC2 Top Correlations:")
for x in d["global_top_pc2"][:5]:
    print(f"  - {x['concept_name']}: r = {x['pearson_r']:+.3f} (p={x['pearson_pval']:.2e})")

print("\nGlobal UMAP1 Top Correlations:")
for x in d["global_top_umap1"][:5]:
    print(f"  - {x['concept_name']}: r = {x['pearson_r']:+.3f} (p={x['pearson_pval']:.2e})")

print("\nGlobal UMAP2 Top Correlations:")
for x in d["global_top_umap2"][:5]:
    print(f"  - {x['concept_name']}: r = {x['pearson_r']:+.3f} (p={x['pearson_pval']:.2e})")

print("\n=== PER-CLASS TOP DISCRIMINATIVE CONCEPTS ===")
for cls, info in d["per_class"].items():
    var1 = info["pca_variance_ratio"][0] * 100
    var2 = info["pca_variance_ratio"][1] * 100
    top_p1 = info["top_pc1_concepts"][0]
    top_p2 = info["top_pc2_concepts"][0]
    top_u1 = info["top_umap1_concepts"][0]
    top_u2 = info["top_umap2_concepts"][0]
    print(f"\n[{cls}] (PCA var: PC1={var1:.1f}%, PC2={var2:.1f}%):")
    print(f"  PC1   : {top_p1['concept_name']} (r = {top_p1['pearson_r']:+.3f})")
    print(f"  PC2   : {top_p2['concept_name']} (r = {top_p2['pearson_r']:+.3f})")
    print(f"  UMAP1 : {top_u1['concept_name']} (r = {top_u1['pearson_r']:+.3f})")
    print(f"  UMAP2 : {top_u2['concept_name']} (r = {top_u2['pearson_r']:+.3f})")
