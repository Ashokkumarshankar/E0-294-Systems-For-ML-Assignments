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
    num_of_conv = 62  # int(sys.argv[1])
    num_of_fc = 62  # int(sys.argv[2])

    l = [
        nn.Conv2d(64, 64, kernel_size=3, padding=1) for i in range(num_of_conv)
    ]

    l += [nn.MaxPool2d(kernel_size=4, stride=4),
          nn.Flatten(), nn.Linear(50176, 4096)]

    l += [
        nn.Linear(4096, 4096) for i in range(num_of_fc-1)
    ]

    model = nn.Sequential(*l).to(device)
    # summary(model, (64, 112, 112), 16)

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
    print(y.size())
    print(start.elapsed_time(end)/num_of_repeat)
    return 0


if __name__ == "__main__":
    main(None)
