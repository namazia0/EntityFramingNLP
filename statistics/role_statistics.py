import matplotlib.pyplot as plt
import os

role_colors = {
    "Protagonist": "#004e9f",
    "Antagonist": "#fcba00",
    "Innocent": "#909085"
}

# Define sub-roles for each main role
protagonist_roles = {"Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"}
antagonist_roles = {"Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"}
innocent_roles = {"Forgotten", "Exploited", "Victim", "Scapegoat"}

main_role_counts = {"Protagonist": 0, "Antagonist": 0, "Innocent": 0}
sub_role_counts = {}

# Load data from file
with open("combination_roles_pt.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split("\t")
        if len(parts) >= 6:
            main_role = parts[4]
            sub_roles = parts[5:]  # 6th and 7th columns (sub-roles)
            
            # Count main roles
            if main_role in main_role_counts:
                main_role_counts[main_role] += 1

            # Count sub-roles
            for sub_role in sub_roles:
                sub_role_counts[sub_role] = sub_role_counts.get(sub_role, 0) + 1

print("Main Role Counts:")
for role, count in main_role_counts.items():
    print(f"{role}: {count}")

print("\nSub-Role Counts:")
for sub_role, count in sorted(sub_role_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{sub_role}: {count}")

sorted_sub_roles = sorted(sub_role_counts.items(), key=lambda x: x[1], reverse=True)
labels, counts = zip(*sorted_sub_roles)

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