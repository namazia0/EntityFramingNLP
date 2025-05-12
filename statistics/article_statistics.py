from scripts.dataset import load_data
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    df_train = load_data("../dataset/train_4_december/EN/subtask-1-annotations.txt")
    print(df_train.keys())
    protagonist = 0
    antagonist = 0
    innocent = 0
    for role in df_train["main_role"]:
        if role == "Protagonist":
            protagonist += 1
        elif role == "Antagonist":
            antagonist += 1
        else:
            innocent += 1

    # Data for the plot
    categories = ["Protagonist", "Antagonist", "Innocent"]
    values = [protagonist, antagonist, innocent]
    colors = ["#A3FFB3", "#FFB3B3", "#A3C8FF"]
    plt.style.use("fivethirtyeight")

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = plt.bar(categories, values, color=colors, edgecolor="black", linewidth=1.2)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.title("Distribution of Main Roles", fontsize=16, fontweight="bold")
    plt.ylabel("Number of Entities", fontsize=14)
    plt.xlabel("Roles", fontsize=14)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height}", ha="center", va="bottom", fontsize=14)

    # Dataset statistics
    stats = {
        'Avg Sentence Length': 25.87,
        'Avg Tokens per Document': 499.25,
        'Avg Unique Words per Document': 271.89
    }

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 6))

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Customize spines to only keep the x and y axes
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)

    bars = ax.bar(stats.keys(), stats.values(), color=['#004e9f', '#fcba00', '#909085'])
    ax.set_title('Linguistic Statistics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', fontsize=15)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height}',
            ha='center',
            va='bottom',
            fontsize=14
        )

    plt.tight_layout()
    plt.show()