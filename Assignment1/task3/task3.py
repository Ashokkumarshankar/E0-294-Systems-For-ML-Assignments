import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE

from time import sleep


def plot_graph(param_name, data, results_file_path):
    sns.set_style("ticks")
    sns.set(rc={'figure.figsize': (10, 10)})
    b = sns.barplot(data=data, x='Device Name',
                    y=param_name, hue='Swap Status')
    for container in b.containers:
        b.bar_label(container, padding=3)
    b.set_xlabel('')
    b.set_ylabel('')
    plt.yscale("symlog")
    plt.ylabel(param_name, fontweight='bold')
    plt.xlabel('Device Name', fontweight='bold')
    plt.tight_layout()
    b.figure.savefig(results_file_path)
    plt.close()


def process_data(path_to_csv_data, path_to_results_folder):
    df = pd.read_csv(path_to_csv_data)
    grouped = df.groupby(['Device Name', 'Swap Status'])
    new_df = grouped[['Time Taken (ms)']].mean()
    new_df = new_df.reset_index()
    for column in new_df.columns[2:]:
        plot_graph(column, new_df, os.path.join(
            path_to_results_folder, f'{column}.png'))


def main():
    results_folder = 'Results'
    results_file_name = 'Task3_Data.csv'
    result_file_path = os.path.join(results_folder, results_file_name)
    os.makedirs(results_folder, exist_ok=True)
    for _ in range(3):
        for swap_value in [0, 1]:
            process = Popen(["python3", 'vgg.py', str(
                swap_value)], stdout=PIPE, stderr=PIPE)
            out, err = process.communicate()
            sleep(3)
    process_data(result_file_path, results_folder)


if __name__ == '__main__':
    main()
