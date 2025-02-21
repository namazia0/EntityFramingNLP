import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Performance data
prompting_strategies = [
    'Zero-shot',
    'One-shot',
    'Few-shot (3 ex)',
    'Few-shot (5 ex)'
]

exact_match_ratio = [0.0000, 0.0110, 0.0000, 0.0440]
micro_f1_score = [0.1610, 0.1434, 0.1838, 0.2327]
main_role_accuracy = [0.7033, 0.7692, 0.7912, 0.8022]

# Set up the plot style
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Create bar plot
x = np.arange(len(prompting_strategies))
width = 0.25

plt.bar(x - width, exact_match_ratio, width, label='Exact Match Ratio', color='blue', alpha=0.7)
plt.bar(x, micro_f1_score, width, label='Micro F1-score', color='green', alpha=0.7)
plt.bar(x + width, main_role_accuracy, width, label='Main Role Accuracy', color='red', alpha=0.7)

plt.xlabel('Prompting Strategies')
plt.ylabel('Performance Metrics')
plt.title('Performance Comparison of Prompting Strategies')
plt.xticks(x, prompting_strategies, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# Create line plot for trend visualization
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

plt.plot(prompting_strategies, exact_match_ratio, marker='o', label='Exact Match Ratio')
plt.plot(prompting_strategies, micro_f1_score, marker='o', label='Micro F1-score')
plt.plot(prompting_strategies, main_role_accuracy, marker='o', label='Main Role Accuracy')

plt.xlabel('Prompting Strategies')
plt.ylabel('Performance Metrics')
plt.title('Performance Trends Across Prompting Strategies')
plt.legend()

plt.tight_layout()
plt.show()



prompting_strategies = [
    'Llama3.2 3b',
    'Llama3.2 1b',
    'Llama3.1 8b'
]

exact_match_ratio = [0.0319, 0.044, 0.0494]
micro_f1_score = [0.1725, 0.1027, 0.2181]
macro_f1_score = [0.1128, 0.0376, 0.1405]
main_role_accuracy = [0.7264, 0.7264, 0.5583]

# Set up the plot style
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Create bar plot
x = np.arange(len(prompting_strategies))
width = 0.25

plt.bar(x - width, exact_match_ratio, width, label='Exact Match Ratio', color='#004e9f', alpha=0.7)
plt.bar(x, micro_f1_score, width, label='Micro F1-score', color='#fcba00', alpha=0.7)
plt.bar(x + width, main_role_accuracy, width, label='Main Role Accuracy', color='#909085', alpha=0.7)

plt.ylabel('Metrics', fontsize=12)
plt.title('Performance Comparison of Prompting Strategies')
plt.xticks(x, prompting_strategies, rotation=45, fontsize=14)
plt.legend()

plt.tight_layout()
plt.show()
