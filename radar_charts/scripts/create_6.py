import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# # Your DataFrame

data = {
    'Efficacy\n (ENG)': [90.3, 99.0, 90.7, 98.7, 89.4, 98.6, 89.6, 98.5],
    'Generalization\n (ENG)': [86.2, 94.6, 87.0, 94.4, 85.0, 93.5, 85.3, 93.4],
    'Specificity\n (ENG)': [72.7, 72.8, 73.4, 72.4, 72.0, 71.9, 72.1, 71.9],
    'Specificity\n (CAT)': [71.8, 72.5, 72.4, 72.8, 71.0, 71.8, 71.0, 71.6],
    'Generalization\n (CAT)': [89.6, 85.3, 87.0, 85.1, 88.9, 83.8, 88.4, 84.2],
    'Efficacy\n (CAT)': [97.8, 89.1, 97.6, 89.0, 97.8, 88.2, 97.7, 88.7],
}

# New data_2
data_2 = {
    'Efficacy (ENG)': [51.6, 84.8, 57.6, 83.2, 46.7, 80.4, 48.2, 79.5],
    'Generalization\n (ENG)': [39.3, 62.2, 45.4, 60.7, 35.4, 53.5, 36.1, 53.2],
    'Specificity (ENG)': [11.5, 13.8, 14.4, 13.7, 9.8, 10.4, 10.1, 10.3],
    'Specificity (CAT)': [15.3, 16.2, 16.6, 17.6, 13.4, 13.0, 13.6, 13.4],
    'Generalization\n (CAT)': [60.8, 52.6, 63.4, 54.8, 57.9, 47.4, 57.6, 48.0],
    'Efficacy (CAT)': [84.3, 62.9, 84.6, 64.2, 83.6, 59.6, 83.3, 59.9],
}

# New data_3
data_3 = {
    'Efficacy (ENG)': [39.0, 72.0, 43.8, 67.6, 34.1, 64.4, 34.9, 63.5],
    'Generalization\n (ENG)': [27.8, 50.0, 32.2, 46.5, 23.6, 40.9, 24.0, 40.4],
    'Specificity (ENG)': [8.3, 9.5, 9.7, 8.8, 7.0, 7.3, 7.2, 7.3],
    'Specificity (CAT)': [9.1, 10.0, 9.7, 10.3, 7.7, 7.9, 7.7, 8.1],
    'Generalization\n (CAT)': [47.9, 40.2, 50.3, 39.8, 43.0, 33.8, 42.5, 34.0],
    'Efficacy (CAT)': [72.8, 49.1, 73.3, 49.3, 68.9, 43.0, 68.4, 43.3],
}





selected_rows1 = [i for i in range(8)]  # English TRAINING
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
fig, ((ax1, ax2,ax3)) = plt.subplots(1, 3, figsize=(19, 8), subplot_kw=dict(polar=True), sharey=True)

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

labels = ["MEMAT-16 (CC)","MEMAT-16 (EE)","MEMAT-16 (CE)","MEMAT-16 (EC)","MEMAT-R16 (CC)","MEMAT-R16 (EE)","MEMAT-R16 (CE)","MEMAT-R16 (EC)"]

colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:cyan","navajowhite","limegreen","lightcoral"]
linestyles = ["dashed" for i in range(4)]+["solid" for i in range(4)]

# Plot for the first subplot
for i, row_index in enumerate(selected_rows1):
    values = df1.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax1.plot(angles, values, linewidth=2, linestyle=linestyles[i], label=f"{labels[i]}",color = colors[i])

# Plot for the second subplot
for i, row_index in enumerate(selected_rows1):
    values = df2.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax2.plot(angles, values, linewidth=2, linestyle=linestyles[i], label=f"{labels[i]}",color = colors[i])

# Plot for the third subplot
for i, row_index in enumerate(selected_rows1):
    values = df3.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax3.plot(angles, values, linewidth=2, linestyle=linestyles[i], label=f"{labels[i]}",color = colors[i])


plt.subplots_adjust(hspace=0, wspace=0.5)

# Add legends
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.13), borderaxespad=0., ncols=4, fontsize=font)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("./radar_charts/figures/CrossLingual_6.pdf")


