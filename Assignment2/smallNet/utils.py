import torch
import torch.nn.functional as F
import tracemalloc
from time import time
import numpy as np
from torchvision import datasets, transforms


def load_data(batch_size, use_cuda):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    peak_data = []
    data_time = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            start = time()
            tracemalloc.start()
            output = model(data)
            data_time.append((time() - start))
            _, peak = tracemalloc.get_traced_memory()
            peak_data.append(peak)
            tracemalloc.reset_peak()
            tracemalloc.stop()
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(f'Took peak memory of {max(peak_data)} B')
    print(f'Took {sum(data_time)*1000} msecs')
