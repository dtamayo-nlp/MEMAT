import pandas as pd
import matplotlib.pyplot as plt
from math import pi

data = {
    'Efficacy \n(ENG)': [98.4, 88.8, 96.4, 99.4, 98.3, 93.4],
    'Generalization\n (ENG)': [92.6, 84.0, 92.5, 95.7, 94.8, 83.7],
    'Specificity\n (ENG)': [71.8, 71.8, 68.9, 68.2, 65.4, 67.4],
    'Specificity\n (CAT)': [71.2, 70.5, 69.5, 67.9, 65.1, 67.8],
    'Generalization\n (CAT)': [83.7, 88.3, 87.6, 92.8, 91.3, 84.8],
    'Efficacy\n (CAT)': [88.9, 97.4, 93.8, 98.6, 97.1, 92.2],
}

data_2 = {
    'Efficacy\n (ENG)': [78.8, 45.2, 69.0, 69.4, 69.5, 55.2],
    'Generalization\n (ENG)': [52.0, 33.4, 48.4, 52.8, 51.4, 37.0],
    'Specificity\n (ENG)': [9.6, 8.8, 9.2, 6.9, 7.4, 10.0],
    'Specificity\n (CAT)': [12.3, 12.7, 9.0, 9.3, 9.3, 7.4],
    'Generalization\n (CAT)': [46.1, 55.7, 45.8, 56.4, 57.2, 27.9],
    'Efficacy\n (CAT)': [57.7, 82.6, 65.6, 76.8, 75.5, 42.1],
}

data_3 = {
    'Efficacy\n (ENG)': [61.5, 31.9,54.4, 53.7, 54.5, 36.7],
    'Generalization\n (ENG)': [38.5, 22.2,37.6, 39.7, 39.2, 23.5],
    'Specificity\n (ENG)': [6.8, 6.6, 6.8, 4.5, 4.6, 5.3],
    'Specificity\n (CAT)': [7.7, 7.3, 6.5, 5.2, 4.5, 4.8],
    'Generalization\n (CAT)': [31.8, 40.4,33.6, 43.8, 43.9, 18.0],
    'Efficacy\n (CAT)': [40.9, 67.0,53.1, 60.3, 59.7, 28.3],
}

selected_rows1 = [2, 3, 4, 5]  # English TRAINING
selected_rows2 = [1, 3, 5]  # English TRAINING

df1 = pd.DataFrame(data).iloc[selected_rows1]


df2 = pd.DataFrame(data_2).iloc[selected_rows1]


df3 = pd.DataFrame(data_3).iloc[selected_rows1]

categories = list(df1.columns)
N = len(categories)


font = 17
# What will be the angle of each axis in the plot
angles = [n / float(N) * 2 * pi + pi/6 for n in range(N)]
angles += angles[:1]

# Create subplots with shared y-axis
fig, ((ax1, ax2,ax3)) = plt.subplots(1, 3, figsize=(19, 6), subplot_kw=dict(polar=True), sharey=True)

# Set common attributes for both subplots
for ax in [ax1, ax2, ax3]:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticklabels([])  # Hide y-axis labels
    # Set x-axis labels only once
    ax.set_xticks(angles[:-1])
    # ax1.set_xticklabels(categories)
    ax.set_xticklabels(categories, rotation=45, fontsize=font, va='center', position=(0, -0.2))
    ax.set_yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10,fontsize=font)
    ax.set_ylim(0, 100)

ax1.set_title('Success Metrics', y=1.3,fontsize=font)
ax2.set_title('Accuracy Metrics', y=1.3,fontsize=font)
ax3.set_title('Magnitude Metrics', y=1.3,fontsize=font)

label_text = ['MEMIT        \n   Bilingual Training']
for i, ax in enumerate([ax1]):
    ax.yaxis.set_label_coords(-0.55, 0.5)  # Adjust the position of the label
    ax.set_ylabel(label_text[i], rotation=0, ha='right',fontsize=font)


labels = ["500 samples (diff subj)\n       (CAT)+(ENG)", " 500 samples\n (CAT)+(ENG)", "1,000 samples\n (CAT)+(ENG)", "1,000 samples\n  (CAT+ENG)"]
linestyles = ["solid","solid","solid","solid","solid","solid"]

# Plot for the first subplot
for i, row_index in enumerate(selected_rows1):
    values = df1.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax1.plot(angles, values, linewidth=2, linestyle=linestyles[i], label=f"{labels[i]}")

# Plot for the second subplot
for i, row_index in enumerate(selected_rows1):
    values = df2.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax2.plot(angles, values, linewidth=2, linestyle=linestyles[i], label=f"{labels[i]}")

# Plot for the third subplot
for i, row_index in enumerate(selected_rows1):
    values = df3.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax3.plot(angles, values, linewidth=2, linestyle=linestyles[i], label=f"{labels[i]}")


plt.subplots_adjust(hspace=0, wspace=1.5)

# Add legends
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.13), borderaxespad=0., ncols=6, fontsize=font)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("./radar_charts/figures/CrossLingual_2.pdf")

