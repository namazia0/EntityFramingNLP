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
    colors = ["#A3FFB3", "#FFB3B3", "#A3C8FF"]#["#1f77b4", "#ff7f0e", "#2ca02c"]  # Custom colors (blue, orange, green)

    # Set a modern style
    plt.style.use("fivethirtyeight")

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = plt.bar(categories, values, color=colors, edgecolor="black", linewidth=1.2)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Add gridlines
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add title and labels
    plt.title("Distribution of Main Roles", fontsize=16, fontweight="bold")
    plt.ylabel("Number of Entities", fontsize=14)
    plt.xlabel("Roles", fontsize=14)

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height}", ha="center", va="bottom", fontsize=14)

    # Show the plot
    plt.tight_layout()
    plt.show()




    # Dataset statistics
    stats = {
        'Avg Sentence Length': 25.87,
        'Avg Tokens per Document': 499.25,
        'Avg Unique Words per Document': 271.89
    }

    # Set style without grid
    sns.set_style("white")  # Base white style
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set background colors to white
    fig.patch.set_facecolor('white')  # White figure background
    ax.set_facecolor('white')  # White axes background

    # Disable the grid
    # ax.grid(False)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Customize spines to only keep the x and y axes
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)  # Thicker y-axis line if needed
    ax.spines['bottom'].set_linewidth(1)  # Thicker x-axis line if needed
    ax.spines['top'].set_linewidth(1)  # Thicker y-axis line if needed
    ax.spines['right'].set_linewidth(1)  # Thicker x-axis line if needed

    # Create bar plot
    bars = ax.bar(stats.keys(), stats.values(), color=['#004e9f', '#fcba00', '#909085'])

    # Customize the plot
    ax.set_title('Linguistic Statistics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', fontsize=15)

    # Add value labels at the top of each bar
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

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()



    categories = ["Protagonist", "Antagonist", "Innocent"]
    values = [130, 476, 79]
    colors = ["#A3FFB3", "#FFB3B3", "#A3C8FF"]  # Custom colors

    # Set a modern style
    plt.style.use("fivethirtyeight")

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color=colors)  # No edgecolor or linewidth

    # Set the background to white
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Remove the grid inside the diagram but keep axes
    ax.grid(False)  # Disable all gridlines
    #ax.yaxis.grid(True, linestyle="--", alpha=0.7)  # Enable only y-axis gridlines

    # Customize spines to only show x and y axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    # Add title and labels
    ax.set_title("Distribution of Main Roles", fontsize=16, fontweight="bold")
    ax.set_ylabel("Number of Entities", fontsize=14)
    ax.set_xlabel("Main Roles", fontsize=14)

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height + 5,  # Adjust space above the bar
            f"{height}", 
            ha="center", 
            va="bottom", 
            fontsize=14
        )

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


    # Dataset statistics
    stats = {
        'Protagonist': 130,
        'Antagonist': 476,
        'Innocent': 79
    }

    # Set style without grid
    sns.set_style("white")  # Base white style
    fig, ax = plt.subplots(figsize=(8, 5))

    # Set background colors to white
    fig.patch.set_facecolor('white')  # White figure background
    ax.set_facecolor('white')  # White axes background

    # Disable the grid
    ax.grid(False)

    # Customize spines to only keep the x and y axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)  # Thicker y-axis line if needed
    ax.spines['bottom'].set_linewidth(1.2)  # Thicker x-axis line if needed

    # Create bar plot
    bars = ax.bar(stats.keys(), stats.values(), color=['#A3FFB3', '#FFB3B3', '#A3C8FF'])

    # Customize the plot
    ax.set_title('Distribution of Main Roles', fontsize=16, fontweight='bold')
    ax.set_ylabel("Number of Entities", fontsize=14)
    ax.set_xlabel("Main Roles", fontsize=14)

    # Add value labels at the top of each bar
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

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


    # Data
    prompting_types = ['Zero-shot', 'One-shot', 'Few-shot (3)', 'Few-shot (5)']
    main_role_accuracy = [0.7033, 0.7692, 0.7912, 0.8022]

    # Create line plot
    plt.figure(figsize=(8, 5))
    plt.plot(prompting_types, main_role_accuracy, marker='o', linestyle='-', color='b', label='Main Role Accuracy')

    # Add labels and title
    plt.title('Main Role Accuracy Across Different Prompting Types', fontsize=16, fontweight='bold')
    plt.xlabel('Prompting Type', fontsize=14)
    plt.ylabel('Main Role Accuracy', fontsize=14)

    # Add value labels at each point
    for i, value in enumerate(main_role_accuracy):
        plt.text(prompting_types[i], value, f'{value:.4f}', ha='center', va='bottom', fontsize=12)

    # Show grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.show()


    # Data
    stats = {
        'Protagonist': 382,
        'Antagonist': 607,
        'Innocent': 378
    }

    # Set style without grid
    sns.set_style("white")  # Base white style
    fig, ax = plt.subplots(figsize=(8, 8))  # Change the figure size for a circular plot

    # Set background colors to white
    fig.patch.set_facecolor('white')  # White figure background
    ax.set_facecolor('white')  # White axes background

    # Disable the grid (not necessary for pie chart)
    ax.grid(False)

    # Customize spines to only keep the x and y axes (not necessary for pie chart)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)  # Thicker y-axis line if needed
    ax.spines['bottom'].set_linewidth(1.2)  # Thicker x-axis line if needed

    # Function to display both value and percentage in each wedge (without decimal points)
    def func(pct, allvals):
        absolute = round(pct / 100.*sum(allvals))
        return f"{absolute} ({pct:.1f}%)"

    # Create pie chart
    colors = ['#004e9f', '#fcba00', '#909085']
    wedges, texts, autotexts = ax.pie(stats.values(), labels=stats.keys(), colors=colors,
                                    autopct=lambda pct: func(pct, stats.values()), startangle=90,
                                    )

    # Customize the plot title
    ax.set_title('Distribution of Main Roles', fontsize=16, fontweight='bold')

    # Add value labels inside the chart
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Adjust layout to ensure everything fits within the plot
    plt.tight_layout()
    plt.show()

# import matplotlib.pyplot as plt
# # Data
# prompting_methods = [
#     "Few-shot prompting (5-shot learning)",
#     "Few-shot prompting (3-shot learning)",
#     "One-shot prompting",
#     "Zero-shot prompting"
# ]
# micro_f1_scores = [0.2294, 0.2477, 0.2663, 0.3246]

# # Create the bar plot
# plt.figure(figsize=(10, 6))


# bars = plt.bar(prompting_methods, micro_f1_scores, color=['#A3FFB3', '#FFB3B3', '#A3C8FF', '#FFA07A'])

# # Add value labels above the bars
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.4f}', ha='center', va='bottom', fontsize=12)

# # Customize plot
# plt.title("Micro F1 Scores for Different Prompting Methods", fontsize=16, fontweight='bold')
# plt.ylabel("Micro F1 Score", fontsize=14)
# plt.xlabel("Prompting Method", fontsize=14)
# plt.xticks(rotation=25, ha='right', fontsize=12)  # Rotate x-axis labels for better readability
# plt.ylim(0, 0.35)  # Adjust y-axis limit to provide space above the bars

# # Add grid for better readability
# #plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Sample data (replace with your actual lists)
# micro_f1_list = [0.82, 0.85, 0.88, 0.79, 0.84]
# macro_f1_list = [0.75, 0.78, 0.80, 0.77, 0.79]
# accuracy = [0.90, 0.88, 0.91, 0.89, 0.90]
# exact_match_ratio = [0.65, 0.68, 0.66, 0.64, 0.67]

# # Combine data into a dictionary
# data = {
#     "Micro F1": micro_f1_list,
#     "Macro F1": macro_f1_list,
#     "Accuracy": accuracy,
#     "Exact Match Ratio": exact_match_ratio
# }

# # Create a boxplot
# plt.figure(figsize=(8, 6))
# sns.boxplot(data=data)

# # Customize plot
# plt.title("Performance Metrics Boxplot")
# plt.ylabel("Scores")
# plt.xticks(rotation=20)  # Rotate labels for better visibility

# # Show plot
# plt.show()
