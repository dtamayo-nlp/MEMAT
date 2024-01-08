import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# # Your DataFrame
data = {
    'Efficacy\n (ENG)': [98.4, 88.8, 90.3, 99.0, 90.7, 98.7],
    'Generalization\n (ENG)': [92.6, 84.0, 86.2, 94.6, 87.0, 94.4],
    'Specificity\n (ENG)': [71.8, 71.8, 72.7, 72.8, 73.4, 72.4],
    'Specificity\n (CAT)': [71.2, 70.5, 71.8, 72.5, 72.4, 72.8],
    'Generalization\n (CAT)': [83.7, 88.3, 89.6, 85.3, 89.8, 85.1],
    'Efficacy\n (CAT)': [88.9, 97.4, 97.8, 89.1, 97.6, 89.0],
}
data_2 = {
    'Efficacy\n (ENG)': [78.8, 45.2, 51.6, 84.8, 57.6, 83.2],
    'Generalization\n (ENG)': [52.0, 33.4, 39.3, 62.2, 45.4, 60.7],
    'Specificity\n (ENG)': [9.6, 8.8, 11.5, 13.8, 14.4, 13.7],
    'Specificity\n (CAT)': [12.3, 12.7, 15.3, 16.2, 16.6, 17.6],
    'Generalization\n (CAT)': [46.1, 55.7, 60.8, 52.6, 63.4, 54.8],
    'Efficacy\n (CAT)': [57.7, 82.6, 84.3, 62.9, 84.6, 64.2],
}

data_3 = {
    'Efficacy\n (ENG)': [61.5, 31.9, 39.0, 72.0, 43.8, 67.6],
    'Generalization\n (ENG)': [38.5, 22.2, 27.8, 50.0, 32.2, 46.5],
    'Specificity (ENG)': [6.8, 6.6, 8.3, 9.5, 9.7, 8.8],
    'Specificity (CAT)': [7.7, 7.3, 9.1, 10.0, 9.7, 10.3],
    'Generalization\n (CAT)': [31.8, 40.4, 47.9, 40.2, 50.3, 39.8],
    'Efficacy\n (CAT)': [40.9, 67.0, 72.8, 49.1, 73.3, 49.3],
}

order = ["Baseline ENG Training","Baseline CAT Training","MEMAT-16 (CC)","MEMAT-16 (EE)","MEMAT-16 (CE)","MEMAT-16 (EC)"]
font= 21.5

selected_rows1 = [0, 3, 5]  # English TRAINING
selected_rows2 = [1, 2, 4]  # English TRAINING

df1 = pd.DataFrame(data).iloc[selected_rows1]
df2 = pd.DataFrame(data).iloc[selected_rows2]


df3 = pd.DataFrame(data_2).iloc[selected_rows1]
df4 = pd.DataFrame(data_2).iloc[selected_rows2]


df5 = pd.DataFrame(data_3).iloc[selected_rows1]
df6 = pd.DataFrame(data_3).iloc[selected_rows2]

categories = list(df1.columns)
N = len(categories)

# What will be the angle of each axis in the plot
angles = [n / float(N) * 2 * pi + pi/6 for n in range(N)]
angles += angles[:1]

# Create subplots with shared y-axis
fig, ((ax1, ax2,ax5), (ax3, ax4,ax6)) = plt.subplots(2, 3, figsize=(28, 13), subplot_kw=dict(polar=True), sharey=True)

label_text = ['English MEMIT\nTraining     ', 'Catalan MEMIT\nTraining     ']
for i, ax in enumerate([ax1,ax3]):
    ax.yaxis.set_label_coords(-0.5, 0.5)  # Adjust the position of the label
    ax.set_ylabel(label_text[i], rotation=0, ha='right',fontsize=font)

# Set common attributes for both subplots
for ax in [ax1, ax2, ax3, ax4,ax5,ax6]:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticklabels([])  # Hide y-axis labels
    # Set x-axis labels only once
    ax.set_xticks(angles[:-1])
    # ax1.set_xticklabels(categories)
    ax.set_xticklabels(categories, rotation=45, fontsize=font, va='center', position=(0, -0.3))
    ax.set_yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10,fontsize=font)
    ax.set_ylim(0, 100)

ax1.set_title('Success Metrics', y=1.3, fontsize=font)
ax2.set_title('Accuracy Metrics', y=1.3, fontsize=font)
ax5.set_title('Magnitude Metrics', y=1.3, fontsize=font)


labels = [order[i] for i in selected_rows1]
linestyles = ["dashed","solid","solid"]
colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:olive"]
# Plot for the first subplot
sel_cols = [colors[i] for i in selected_rows1]
for i, row_index in enumerate(selected_rows1):
    values = df1.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax1.plot(angles, values, linewidth=2, linestyle=linestyles[i], label=f"{labels[i]}", color= sel_cols[i])

labels = [order[i] for i in selected_rows2]
sel_cols = [colors[i] for i in selected_rows2]

# Plot for the second subplot
for i, row_index in enumerate(selected_rows2):
    values = df2.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax3.plot(angles, values, linewidth=2, linestyle=linestyles[i], label=f"{labels[i]}", color= sel_cols[i])

sel_cols = [colors[i] for i in selected_rows1]

labels = [order[i] for i in selected_rows1]
# Plot for the third subplot
for i, row_index in enumerate(selected_rows1):
    values = df3.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax2.plot(angles, values, linewidth=2, linestyle=linestyles[i], label=f"{labels[i]}", color= sel_cols[i])

labels = [order[i] for i in selected_rows2]
sel_cols = [colors[i] for i in selected_rows2]

# Plot for the second subplot
for i, row_index in enumerate(selected_rows2):
    values = df4.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax4.plot(angles, values, linewidth=2, linestyle=linestyles[i], label=f"{labels[i]}", color= sel_cols[i])

labels = [order[i] for i in selected_rows1]
sel_cols = [colors[i] for i in selected_rows1]

    # Plot for the third subplot
for i, row_index in enumerate(selected_rows1):
    values = df5.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax5.plot(angles, values, linewidth=2, linestyle=linestyles[i], label=f"{labels[i]}", color= sel_cols[i])
labels = [order[i] for i in selected_rows2]
sel_cols = [colors[i] for i in selected_rows2]

# Plot for the second subplot
for i, row_index in enumerate(selected_rows2):
    values = df6.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax6.plot(angles, values, linewidth=2, linestyle=linestyles[i], label=f"{labels[i]}", color= sel_cols[i])

plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Add legends
# handles, labels = ax1.get_legend_handles_labels()

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax3.get_legend_handles_labels()

# Concatenate labels and handles
handles = handles1 + handles2
labels = labels1 + labels2

custom_order = [0, 3, 4, 1, 5, 2]  # Specify the desired order of legend entries
all_handles = []
all_labels = []

for i in custom_order:
    all_handles.append(handles[i])
    all_labels.append(labels[i])

fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 0.045), borderaxespad=0., ncols=6, fontsize=font)
fig.savefig("./radar_charts/figures/CrossLingual_3.pdf")