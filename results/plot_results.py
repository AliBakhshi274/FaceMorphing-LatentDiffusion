import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./final_comparison_metrics.csv')


df['Weight_Num'] = df['Weight'].map({'w_0': 0.0, 'w_5': 0.5, 'w_10': 1.0})


sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))



# Plot 1 -----------------------------------------------------------------------
# Plot the cosine similarity against the negative prompt weight for P-
sns.lineplot(data=df, x='Weight_Num', y='Gen_Sim_P-', hue='Architecture', 
             marker='o', ax=axes[0], errorbar='sd', linewidth=2.5, markersize=8)

# line
axes[0].axhline(y=0.10, color='red', linestyle='--', label='Threshold (0.10)')
# Set titles and labels for the first plot
axes[0].set_title('Suppression of Known Identity (P-)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Negative Prompt Weight (w)', fontsize=12)
axes[0].set_ylabel('Cosine Similarity with P-', fontsize=12)
axes[0].legend()


# Plot 2 ----------------------------------------------------------------------------------------
# Plot the cosine similarity against the negative prompt weight for the Target identity
sns.lineplot(data=df, x='Weight_Num', y='Gen_Sim_Target', hue='Architecture', 
             marker='s', ax=axes[1], errorbar='sd', linewidth=2.5, markersize=8)
axes[1].set_title('Revelation of Hidden Target', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Negative Prompt Weight (w)', fontsize=12)
axes[1].set_ylabel('Cosine Similarity with Target', fontsize=12)

# Adjust layout
plt.tight_layout()

plt.savefig('comparison_plot.png', dpi=300)
print("[+] Plot successfully saved as 'comparison_plot.png'")
