#!/usr/bin/python3

import torch
import torch.nn as nn
from torchsummary import summary
import sys


def main(args):
    num_of_warmup = 5
    num_of_repeat = 10
    inpSize = (16, 64, 112, 112)
    device = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    num_of_conv = 84
    num_of_conv1 = 54  # int(sys.argv[1])
    num_of_fc = 62  # int(sys.argv[2])

    l = [
        nn.Conv2d(64, 64, kernel_size=3, padding=1) for i in range(num_of_conv)
    ]
    l1 = [
        nn.Conv2d(64, 64, kernel_size=3, padding=1) for i in range(num_of_conv1)
    ]

    l1 += [nn.MaxPool2d(kernel_size=4, stride=4),
           nn.Flatten(), nn.Linear(50176, 4096)]

    l1 += [
        nn.Linear(4096, 4096) for i in range(num_of_fc-1)
    ]

    model = nn.Sequential(*l).to(device)
    model1 = nn.Sequential(*l1).to(device1)
    # summary(model, (64, 112, 112), 16)
    # summary(model1, (64, 112, 112), 16)
    x = torch.randn(inpSize).to(device)

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

    print(f'Time taken by GPU0 {start.elapsed_time(end)/num_of_repeat}')

    y1 = y.to(device1)
    for _ in range(num_of_warmup):
        yt = model1(y1)

    torch.cuda.synchronize()
    start.record()
    for _ in range(num_of_repeat):
        yt = model1(y1)
    end.record()
    torch.cuda.synchronize()

    print(f'Time taken by GPU1 {start.elapsed_time(end)/num_of_repeat}')
    return 0


if __name__ == "__main__":
    main(None)
