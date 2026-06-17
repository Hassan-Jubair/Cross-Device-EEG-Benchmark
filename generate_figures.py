# generate_figures.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math
import zipfile

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

domains = ['Domain 0', 'Domain 1', 'Domain 2', 'Domain 3', 'Domain 4']
models = ['SimpleCNN', 'EEGNet', 'ShallowConvNet', 'DANN']

results_dict = {
    'SimpleCNN': [36.07, 53.11, 58.26, 57.93, 39.86],
    'EEGNet': [26.64, 55.37, 53.35, 54.37, 45.45],
    'ShallowConvNet': [32.67, 55.59, 58.70, 55.57, 38.57],
    'DANN': [54.58, 54.48, 58.15, 48.85, 56.97]
}

data = []
for model in models:
    for i, domain in enumerate(domains):
        data.append({'Model': model, 'Domain': domain, 'Accuracy (%)': results_dict[model][i]})
df = pd.DataFrame(data)

sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")
palette = sns.color_palette("muted")

os.makedirs("manuscript_figures", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='Accuracy (%)', data=df, palette=palette, capsize=.1, errorbar=('ci', 95), edgecolor='black')
plt.axhline(y=50, color='r', linestyle='--', label='Chance Level (50%)')
plt.title('Overall Mean Accuracy (with 95% CI)', pad=15, fontweight='bold')
plt.ylim(20, 65)
plt.ylabel('Mean Accuracy (%)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("manuscript_figures/Fig1_Overall_Mean_CI.png", dpi=300, bbox_inches='tight')
plt.savefig("manuscript_figures/Fig1_Overall_Mean_CI.pdf", bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x='Domain', y='Accuracy (%)', hue='Model', data=df, palette=palette, edgecolor='black')
plt.axhline(y=50, color='r', linestyle='--', label='Chance Level (50%)')
plt.title('Cross-Dataset LODO Performance per Domain', pad=15, fontweight='bold')
plt.ylim(20, 65)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("manuscript_figures/Fig2_PerDomain_Bar.png", dpi=300, bbox_inches='tight')
plt.savefig("manuscript_figures/Fig2_PerDomain_Bar.pdf", bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Model', y='Accuracy (%)', data=df, palette=palette, width=0.5, linewidth=2)
sns.stripplot(x='Model', y='Accuracy (%)', data=df, color='black', alpha=0.6, size=7, jitter=True)
plt.axhline(y=50, color='r', linestyle='--', label='Chance Level (50%)')
plt.title('Distribution of Cross-Dataset Accuracy', pad=15, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim(20, 65)
plt.tight_layout()
plt.savefig("manuscript_figures/Fig3_Stability_Boxplot.png", dpi=300, bbox_inches='tight')
plt.savefig("manuscript_figures/Fig3_Stability_Boxplot.pdf", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = [n / float(len(domains)) * 2 * math.pi for n in range(len(domains))]
angles += angles[:1] 

for i, model in enumerate(models):
    values = results_dict[model].copy()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=palette[i])
    ax.fill(angles, values, alpha=0.1, color=palette[i])

plt.xticks(angles[:-1], domains)
ax.set_ylim(20, 65)
plt.title('Performance Profile Across Domains', size=15, fontweight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("manuscript_figures/Fig4_Radar_Chart.png", dpi=300, bbox_inches='tight')
plt.savefig("manuscript_figures/Fig4_Radar_Chart.pdf", bbox_inches='tight')
plt.close()

zip_filename = "Manuscript_Figures_HQ.zip"
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files_list in os.walk("manuscript_figures"):
        for file in files_list:
            zipf.write(os.path.join(root, file), file)

if IN_COLAB:
    files.download(zip_filename)