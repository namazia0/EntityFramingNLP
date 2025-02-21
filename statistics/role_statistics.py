import matplotlib.pyplot as plt
import os

# Role categories and colors
role_colors = {
    "Protagonist": "#004e9f",
    "Antagonist": "#fcba00",
    "Innocent": "#909085"
}

# Define sub-roles for each main role
protagonist_roles = {"Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"}
antagonist_roles = {"Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"}
innocent_roles = {"Forgotten", "Exploited", "Victim", "Scapegoat"}

# Initialize dictionaries to store counts
main_role_counts = {"Protagonist": 0, "Antagonist": 0, "Innocent": 0}
sub_role_counts = {}

# Load data from file
with open("combination_roles_pt.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split("\t")
        if len(parts) >= 6:  # Ensure enough columns exist
            main_role = parts[4]
            sub_roles = parts[5:]  # 6th and 7th columns (sub-roles)
            
            # Count main roles
            if main_role in main_role_counts:
                main_role_counts[main_role] += 1

            # Count sub-roles
            for sub_role in sub_roles:
                sub_role_counts[sub_role] = sub_role_counts.get(sub_role, 0) + 1

# Print results
print("Main Role Counts:")
for role, count in main_role_counts.items():
    print(f"{role}: {count}")

print("\nSub-Role Counts:")
for sub_role, count in sorted(sub_role_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{sub_role}: {count}")

# Sort sub-roles for better visualization
sorted_sub_roles = sorted(sub_role_counts.items(), key=lambda x: x[1], reverse=True)
labels, counts = zip(*sorted_sub_roles)

# Assign colors based on main role category
colors = []
for sub_role in labels:
    if sub_role in protagonist_roles:
        colors.append(role_colors["Protagonist"])
    elif sub_role in antagonist_roles:
        colors.append(role_colors["Antagonist"])
    elif sub_role in innocent_roles:
        colors.append(role_colors["Innocent"])

# Plot the boxplot
plt.figure(figsize=(12, 6))
plt.bar(labels, counts, color=colors)
plt.tick_params(axis='y', labelsize=13)
plt.xticks(rotation=45, ha="right", fontsize=13)
plt.xlabel("Subroles", fontsize=15)
plt.ylabel("Count", fontsize=15)
# plt.title("Distribution of Sub-Roles in Dataset (EN)", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Create a legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in role_colors.values()]
plt.legend(handles, role_colors.keys(), title="Main Role", loc="upper right", fontsize=13, title_fontsize=14)

plt.show()

directory_path = "../dataset/dev_4_december/PT/subtask-1-documents"
num_items = len(os.listdir(directory_path))

print(f"Number of items in '{directory_path}': {num_items}")


directory_path = "../dataset/dev_4_december/EN/subtask-1-documents"
num_items = len(os.listdir(directory_path))

print(f"Number of items in '{directory_path}': {num_items}")


directory_path = "../dataset/train_4_december/PT/raw-documents"
num_items = len(os.listdir(directory_path))

print(f"Number of items in '{directory_path}': {num_items}")

directory_path = "../dataset/train_4_december/EN/raw-documents"
num_items = len(os.listdir(directory_path))

print(f"Number of items in '{directory_path}': {num_items}")



from collections import Counter

# File path
file_path = "../dataset/train_4_december/EN/subtask-1-annotations.txt"

# Read file and count named entities
named_entities = []

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        elements = line.strip().split("\t")  # Split by tab
        if len(elements) > 1:  # Ensure there is a named entity
            named_entities.append(elements[1])  # Second column is the named entity

# Count occurrences of each named entity
entity_counts = Counter(named_entities)

# Sort entities by count (ascending order) and get the top 5 most frequent ones
top_10_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
print(top_10_entities)

# Print the results
for entity, count in top_10_entities:
    print(f"{entity}: {count}")

# Extract names and counts for plotting
entities, counts = zip(*top_10_entities)

# Create bar chart
plt.figure(figsize=(10, 5))
plt.bar(entities, counts, color="#004e9f", edgecolor="black")
plt.grid(axis="y", linestyle="--", alpha=0.7)
# Add labels and title
plt.xlabel("Named Entities", fontsize=12)
plt.ylabel("Frequency", fontsize=15)
plt.title("Top 5 Most Frequent Named Entities", fontsize=14)
plt.xticks(rotation=45, fontsize=12)  # Rotate labels for better readability

# Show the plot
plt.show()

