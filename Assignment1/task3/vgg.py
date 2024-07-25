import os
import sys
import torch
import pandas as pd
import torch.nn as nn

from time import time


class VGG(nn.Module):
    def __init__(self, swap_status):
        super(VGG, self).__init__()
        # a vector descriping VGG16 dimensions
        vec = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ]

        self.listModule = []
        in_channels = 3
        for v in vec:
            if v == "M":
                if swap_status and isinstance(self.listModule[-1], nn.ReLU):
                    self.listModule.insert(-1,
                                           nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.listModule += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = int(v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                self.listModule += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        self.features = nn.Sequential(*self.listModule)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.cl = [
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        ]
        self.classifier = nn.Sequential(*self.cl)

    def forward(self, x):
        # x = self.features(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def main():
    # pass non-zero value for swapping relu and max-pool layers.
    num_of_warmup = 5
    num_of_repeat = 10
    results_folder = 'Results'
    results_file_name = 'Task3_Data.csv'
    column_names = ['Device Name', 'Swap Status', 'Time Taken (ms)']
    result_file_path = os.path.join(results_folder, results_file_name)
    if os.path.exists(result_file_path):
        final_df = pd.read_csv(result_file_path)
    else:
        final_df = pd.DataFrame(columns=column_names)
    swap = int(sys.argv[1])
    inpSize = (128, 3, 224, 224)
    device = torch.device("cuda:0")
    device_cpu = torch.device("cpu")
    model = VGG(swap).to(device)
    model_cpu = VGG(swap).to(device_cpu)

    x = torch.randn(inpSize).to(device)
    x_cpu = torch.randn(inpSize).to(device_cpu)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(num_of_warmup):
        y = model(x)
    torch.cuda.synchronize()
    start.record()
    for _ in range(num_of_repeat):
        y = model(x)
    end.record()
    torch.cuda.synchronize()
    time_taken = start.elapsed_time(end)/num_of_repeat
    print(f'Time taken for GPU is {time_taken} ms')
    new_row = {column_names[0]: 'GPU', column_names[1]: (
        'Swapped' if swap else 'Normal'), column_names[2]: time_taken}  
    final_df = final_df.append([new_row], ignore_index=True)
    start_cpu = time()
    y_cpu = model_cpu(x_cpu)
    end_cpu = time()
    time_taken_cpu = end_cpu - start_cpu
    print(f'Time taken for CPU is {time_taken_cpu} s')
    new_row = {column_names[0]: 'CPU', column_names[1]: (
        'Swapped' if swap else 'Normal'), column_names[2]: time_taken_cpu*1000}
    final_df = final_df.append([new_row], ignore_index=True)
    final_df.to_csv(result_file_path, index=False)


if __name__ == '__main__':
    main()
