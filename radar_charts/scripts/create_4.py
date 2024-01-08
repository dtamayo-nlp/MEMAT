import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# # Your DataFrame

data = {
    'Efficacy\n (ENG)': [98.4, 88.8, 93.4, 99.0],
    'Generalization\n (ENG)': [92.6, 84.0, 83.7, 96.2],
    'Specificity\n (ENG)': [71.8, 71.8, 67.4, 67.2],
    'Specificity\n (CAT)': [71.2, 70.5, 67.8, 66.6],
    'Generalization\n (CAT)': [83.7, 88.3, 84.8, 93.2],
    'Efficacy\n (CAT)': [88.9, 97.4, 92.2, 97.6],
}

data_2 = {
    'Efficacy (ENG)': [78.8, 45.2, 69.5, 78.5],
    'Generalization\n (ENG)': [52.0, 33.4, 51.4, 63.2],
    'Specificity (ENG)': [9.6, 8.8, 7.4, 11.5],
    'Specificity (CAT)': [12.3, 12.7, 9.3, 13.6],
    'Generalization\n (CAT)': [46.1, 55.7, 57.2, 65.6],
    'Efficacy (CAT)': [57.7, 82.6, 75.5, 80.1],
}

data_3 = {
    'Efficacy (ENG)': [61.5, 31.9, 54.5, 68.0],
    'Generalization\n (ENG)': [38.5, 22.2, 39.2, 53.1],
    'Specificity (ENG)': [6.8, 6.6, 4.6, 7.1],
    'Specificity (CAT)': [7.7, 7.3, 4.5, 7.0],
    'Generalization\n (CAT)': [31.8, 40.4, 43.9, 55.3],
    'Efficacy (CAT)': [40.9, 67.0, 59.7, 70.0],
}



selected_rows1 = [0, 1, 2, 3]  # English TRAINING
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
fig, ((ax1, ax2,ax3)) = plt.subplots(1, 3, figsize=(19, 7), subplot_kw=dict(polar=True), sharey=True)

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

labels = ["MEMIT (ENG)", "MEMIT (CAT)", "MEMIT (CAT)+(ENG)", "MEMAT-16 (CAT+ENG)"]

linestyles = ["dashed","dashed","dashdot","solid"]
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


plt.subplots_adjust(hspace=0, wspace=0.5)

# Add legends
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.13), borderaxespad=0., ncols=4, fontsize=font)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("./radar_charts/figures/CrossLingual_4.pdf")
