import matplotlib.pyplot as pyplot
import numpy as np

def plot_quality_over_entropy(vad_series, anl2_series):
    figure, axis = pyplot.subplots()
    axis.set_ylabel("Sorting Quality") 
    axis.set_xlabel("Number of Philox Rounds")
    pyplot.plot(np.arange(len(vad_series)), vad_series, label="VAD", color="peachpuff")
    pyplot.plot(np.arange(len(anl2_series)), anl2_series, label="AND", color="lightblue")
    pyplot.legend()
    pyplot.show(block=True)

def plot_statistics():
    import json
    import os

    # Read the statistics from the JSON file
    statistics_file = "eval/statistics/permutation_effect_stats.json"
    if os.path.exists(statistics_file):
        with open(statistics_file, "r") as f:
            statistics = json.load(f)
    else:
        print(f"Warning: Statistics file {statistics_file} not found.")
        statistics = {}

    labels = list(statistics.keys())
    figure, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(20, 10))
    fontsize = 24  # Moved fontsize before usage

    # Set titles with padding
    axes[0].set_title(
        "Average Neighbor L2 Distance vs Permutation Type", fontsize=fontsize + 4, pad=20
    )
    axes[1].set_title("VAD vs Permutation Type", fontsize=fontsize + 4, pad=20)

    anl_data = [
        [statistics[label][i]["AND"] for i in range(len(statistics[label]))] for label in labels
    ]
    vad_data = [
        [statistics[label][i]["VAD"] for i in range(len(statistics[label]))] for label in labels
    ]
    axes[0].violinplot(anl_data, showmeans=True, showmedians=False)
    axes[1].violinplot(vad_data, showmeans=True, showmedians=False)

    labels[0] = "torch"
    for ax in axes:
        ax.set_xticks(np.arange(len(labels)) + 1, labels)
        ax.xaxis.grid(True)
        ax.set_xlabel("Permutation Type", fontsize=fontsize, labelpad=20)
        ax.set_ylabel("Value", fontsize=fontsize, labelpad=20)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    pyplot.tight_layout()
    pyplot.show(block=True)

plot_statistics()



