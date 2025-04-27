import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the benchmark performance results for all trials that used
# the given training data selection method
def load_acc_results(results_folder, selection_method, benchmarks):
    acc_results = {}

    for filename in os.listdir(results_folder):
        if filename.startswith(selection_method): # only load results from trials that used selection_method
            trial_file_path = os.path.join(results_folder, filename)
            with open(trial_file_path, "r") as trial_file:
                benchmark_results = json.load(trial_file)
                # collect the acurracy result from each benchmark
                accuracies = []
                for benchmark in benchmarks:
                    acc = benchmark_results[benchmark]["accuracy"]
                    accuracies.append(acc)
                acc_results[filename] = accuracies # store all benchmark accuracies for the current trial

    return acc_results


# Create a grouped bar chart showing the benchmark accuracies for each trial
def plot_results(acc_results, plot_title, benchmarks, benchmark_plot_styles, selection_method):
    trial_names = acc_results.keys()
    num_trials = len(trial_names)
    num_benchmarks = len(benchmarks)

    x = np.arange(num_trials)

    fig, axes = plt.subplots(figsize = (10,6))

    for i in range(num_benchmarks):
        benchmark = benchmarks[i]
        marker, linestyle, color = benchmark_plot_styles[benchmark]
        axes.plot(x, [acc_results[trial_name][i] for trial_name in trial_names], marker=marker, label=benchmark, linestyle=linestyle, color=color)

    axes.set_ylabel("Accuracy")
    axes.set_xlabel("Trials")
    axes.set_title(plot_title)
    axes.set_xticks(x)
    axes.set_xticklabels(trial_names) # show trial file names for each bar group
    axes.set_ylim(0, 1)
    axes.legend(title="Benchmark") # shows which benchmark each bar represents
    plt.tight_layout() # makes the plot fill the window
    
    save_path = os.path.join("result_graphs", selection_method)
    plt.savefig(save_path)
    plt.show()

def main():
    results_folder = "results"
    benchmarks = ["MATH500", "AMC23", "OlympiadBench", "Minerva"]
    # for each benchmark specify marker, linestyle, and color
    benchmark_plot_styles = {"MATH500": ["s", "-", "blue"], 
                             "AMC23": ["o", "-", "orange"],
                             "OlympiadBench": ["^", "--", "green"],
                             "Minerva": ["D", "--", "red"]} 
    selection_methods = ["rand", "len", "cluster"]
    plot_titles = {"rand": "Benchmark Performance with Random Training Data Selection",
                   "len": "Benchmark Performance with Training Data Selection by Length",
                   "cluster": "Benchmark Performance with Training Data Selection by Clustering"}

    for selection_method in selection_methods:
        acc_results = load_acc_results(results_folder, selection_method, benchmarks)
        plot_title = plot_titles[selection_method]
        plot_results(acc_results, plot_title, benchmarks, benchmark_plot_styles, selection_method)

if __name__ == "__main__":
    main()