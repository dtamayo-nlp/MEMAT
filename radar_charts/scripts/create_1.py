import pandas as pd
import matplotlib.pyplot as plt
from math import pi

data = {
    'Efficacy\n (ENG)': [94.8, 56.2, 98.8, 88.8, 98.4, 88.8],
    'Generalization\n (ENG)': [87.5, 55.4, 93.2, 84.4, 92.6, 84.0],
    'Specificity\n (ENG)': [71.6, 73.5, 72.1, 72.0, 71.8, 71.8],
    'Specificity\n (CAT)': [73.4, 72.0, 72.6, 72.0, 71.2, 70.5],
    'Generalization\n (CAT)': [49.4, 77.1, 82.7, 88.8, 83.7, 88.3],
    'Efficacy\n (CAT)': [45.8, 91.8, 87.8, 98.0, 88.9, 97.4],
}

data_2 = {
    'Efficacy (ENG)': [66.8, 11.0, 76.2, 44.8, 78.8, 45.2],
    'Generalization\n (ENG)': [44.6, 10.1, 50.3, 33.7, 52.0, 33.4],
    'Specificity (ENG)': [10.4, 10.9, 8.9, 8.0, 9.6, 8.8],
    'Specificity (CAT)': [12.0, 10.3, 11.7, 11.3, 12.3, 12.7],
    'Generalization\n (CAT)': [13.1, 35.7, 41.8, 56.8, 46.1, 55.7],
    'Efficacy (CAT)': [15.0, 63.4, 54.6, 82.4, 57.7, 82.6],
}

data_3 = {
    'Efficacy (ENG)': [54.9, 6.1, 60.0, 32.0, 61.5, 31.9],
    'Generalization\n (ENG)': [34.2, 3.1, 38.2, 22.3, 38.5, 22.2],
    'Specificity (ENG)': [8.8, 9.9, 6.1, 5.7, 6.8, 6.6],
    'Specificity (CAT)': [9.3, 7.9, 7.7, 7.0, 7.7, 7.3],
    'Generalization\n (CAT)': [3.9, 25.2, 29.7, 42.2, 31.8, 40.4],
    'Efficacy (CAT)': [3.0, 51.8, 37.8, 67.6, 40.9, 67.0],
}

font=21.5

selected_rows1 = [0, 2, 4]  # English TRAINING
selected_rows2 = [1, 3, 5]  # English TRAINING

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
fig, ((ax1, ax2,ax5), (ax3, ax4,ax6)) = plt.subplots(2, 3, figsize=(26, 12), subplot_kw=dict(polar=True), sharey=True)

label_text = ['English MEMIT\nTraining     ', 'Catalan MEMIT\nTraining     ']
for i, ax in enumerate([ax1,ax3]):
    ax.yaxis.set_label_coords(-0.5, 0.5)  # Adjust the position of the label
    ax.set_ylabel(label_text[i], rotation=0, ha='right', fontsize=font)


# Set common attributes for both subplots
for ax in [ax1, ax2, ax3, ax4,ax5,ax6]:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticklabels([])  # Hide y-axis labels
    # Set x-axis labels only once
    ax.set_xticks(angles[:-1])
    # ax1.set_xticklabels(categories)
    ax.set_xticklabels(categories, rotation=45, fontsize=font, va='center', position=(0, -0.2))
    ax.set_yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10, fontsize=font)
    ax.set_ylim(0, 100)

ax1.set_title('Success Metrics', y=1.23,fontsize=font)
ax2.set_title('Accuracy Metrics', y=1.23,fontsize=font)
ax5.set_title('Magnitude Metrics', y=1.23,fontsize=font)

labels = ["500 samples (diff subj)", "500 samples", "1,000 samples"]
# Plot for the first subplot
for i, row_index in enumerate(selected_rows1):
    values = df1.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=f"{labels[i]}")

# Plot for the second subplot
for i, row_index in enumerate(selected_rows2):
    values = df2.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax3.plot(angles, values, linewidth=2, linestyle='solid', label=f"{labels[i]}")


# Plot for the third subplot
for i, row_index in enumerate(selected_rows1):
    values = df3.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax2.plot(angles, values, linewidth=2, linestyle='solid', label=f"{labels[i]}")

# Plot for the second subplot
for i, row_index in enumerate(selected_rows2):
    values = df4.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax4.plot(angles, values, linewidth=2, linestyle='solid', label=f"{labels[i]}")

    # Plot for the third subplot
for i, row_index in enumerate(selected_rows1):
    values = df5.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax5.plot(angles, values, linewidth=2, linestyle='solid', label=f"{labels[i]}")

# Plot for the second subplot
for i, row_index in enumerate(selected_rows2):
    values = df6.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax6.plot(angles, values, linewidth=2, linestyle='solid', label=f"{labels[i]}")

plt.subplots_adjust(hspace=0.5, wspace=0.2)

# Add legends
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), borderaxespad=0., ncols=6, fontsize=font)
fig.savefig("./radar_charts/figures/CrossLingual_1.pdf")
