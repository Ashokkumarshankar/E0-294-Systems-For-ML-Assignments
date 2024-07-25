import os
import pandas as pd
import seaborn as sns
from time import sleep
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE


def plot_graph(param_name, data, results_file_path):
    sns.set(rc={'figure.figsize': (10, 10)})
    b = sns.barplot(data=data, x='Layer Type', y=param_name, hue='Status')
    for container in b.containers:
        b.bar_label(container, padding=3)
    b.set_xlabel('')
    b.set_ylabel('')
    plt.yscale("symlog")
    plt.ylabel(param_name, fontweight='bold')
    plt.xlabel('Layer Type', fontweight='bold')
    plt.tight_layout()
    b.figure.savefig(results_file_path)
    plt.close()


def process_data(path_to_csv_data, path_to_results_folder):
    df = pd.read_csv(path_to_csv_data)
    df['memory.used [MiB]'] = df['memory.used [MiB]'].apply(
        lambda row: int(row.split(' ')[0]))
    df['power.draw [W]'] = df[' power.draw [W]'].apply(
        lambda row: float(row.split(' ')[1]))
    df = df[['memory.used [MiB]', 'power.draw [W]', 'Layer Type', 'Status']]
    grouped = df.groupby(['Layer Type', 'Status'])
    new_df = grouped[['memory.used [MiB]', 'power.draw [W]']].mean()
    new_df = new_df.reset_index()
    for column in new_df.columns[2:]:
        plot_graph(column, new_df, os.path.join(
            path_to_results_folder, f'{column}.png'))


def main():
    results_folder = 'Results'
    results_file_name = 'Task2_Data.csv'
    result_file_path = os.path.join(results_folder, results_file_name)
    os.makedirs(results_folder, exist_ok=True)
    for _ in range(3):
        for file_name in ['relu.py', 'conv.py', 'maxpool.py', 'linear.py']:
            process = Popen(["python3", file_name], stdout=PIPE, stderr=PIPE)
            out, err = process.communicate()
            # Sleeping to make sure that it doesn't effect the results of power drawn.
            if file_name in ('maxpool.py', 'conv.py'):
                sleep(60)
            else:
                sleep(30)
    process_data(result_file_path, results_folder)


if __name__ == '__main__':
    main()
