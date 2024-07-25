import os
import io
import torch
import pandas as pd
import torch.nn as nn


def get_gpu_data(device_number):
    # utilization.gpu,utilization.memory -> Not working for layers which are compute intensive as missing chance to capture their data.
    result_data = os.popen(
        f"nvidia-smi --query-gpu=memory.used,power.draw -i {device_number} --format=csv").read()
    df = pd.read_csv(io.StringIO(result_data))
    return df


def main():
    warmup = False
    inpSize = (256, 3, 224, 224)
    result_file_name = './Results/Task2_Data.csv'
    device_number = 0
    device = torch.device(f'cuda:{device_number}')
    if os.path.exists(result_file_name):
        final_df = pd.read_csv(result_file_name)
    else:
        final_df = pd.DataFrame()
    if not warmup:
        torch.cuda.synchronize()
        temp_df = get_gpu_data(device_number)
        temp_df['Layer Type'] = 'Conv'
        temp_df['Status'] = 'Before'
        final_df = pd.concat([temp_df, final_df], ignore_index=True)
        model = nn.Conv2d(3, 128, kernel_size=3, padding=1).to(device)
        x = torch.randn(inpSize).to(device)
        y = model(x)
        torch.cuda.synchronize()
        temp_df = get_gpu_data(device_number)
        temp_df['Layer Type'] = 'Conv'
        temp_df['Status'] = 'After'
        final_df = pd.concat([temp_df, final_df], ignore_index=True)
        final_df.to_csv(result_file_name, index=False)
    else:
        model = nn.Conv2d(3, 128, kernel_size=3, padding=1).to(device)
        x = torch.randn(inpSize).to(device)
        y = model(x)
        torch.cuda.synchronize()
        temp_df = get_gpu_data(device_number)
        temp_df['Layer Type'] = 'Conv'
        temp_df['Status'] = 'Before'
        final_df = pd.concat([temp_df, final_df], ignore_index=True)
        y = model(x)
        torch.cuda.synchronize()
        temp_df = get_gpu_data(device_number)
        temp_df['Layer Type'] = 'Conv'
        temp_df['Status'] = 'After'
        final_df = pd.concat([temp_df, final_df], ignore_index=True)
        final_df.to_csv(result_file_name, index=False)


if __name__ == '__main__':
    main()
