import torch
import torch.profiler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class VGG(nn.Module):
    def __init__(self):
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


def main(batch_size, total_epochs):
    input_size = 224
    input_channels = 3
    output_classes = 1000
    total_input_count = 8192
    device_number = 0
    device = torch.device(f'cuda:{device_number}')
    model = VGG().to(device)
    dataset = [(torch.randn((input_channels, input_size, input_size)), torch.randint(
        0, output_classes, (1, )).item()) for _ in range(total_input_count)]
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.005,
                          weight_decay=0.005, momentum=0.9)
    train_data = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1, warmup=2, active=5, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            './log_single/vgg16'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for epoch in range(total_epochs):
            for source, targets in train_data:
                source = source.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                output = model(source)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                prof.step()
    end.record()
    torch.cuda.synchronize()
    print(
        f'Done with training in {start.elapsed_time(end)}ms on GPU{device_number}')


if __name__ == '__main__':
    total_epochs = 1
    batch_size = 64
    main(batch_size, total_epochs)
