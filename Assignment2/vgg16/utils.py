
import os
import random
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm

PATH_TO_CONFIG_FILE = 'config.yaml'
PATH_TO_RESULT_FILES = './results/'


def set_seed(seed: int = 41, platform='torch') -> None:
    """Set Seed for reproducible results."""

    np.random.seed(seed)
    random.seed(seed)
    if platform == 'torch':
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
    else:
        raise Exception("Tensorflow seed.")
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = sum(np.any(correct[:k].cpu().numpy(), axis=0))
        res.append(correct_k)
    return res


def plot_loss_accuracy_data(data1, data2, label1, label2, xlabel, ylabel1, ylabel2, path_to_fig):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(range(len(data1)), data1, label=label1, color='orange')
    ax2.plot(range(len(data2)), data2, label=label2)

    ax1.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel1)
    ax2.set_ylabel(ylabel2)

    ax1.legend(loc='upper center')
    ax2.legend(loc='lower center')
    plt.title(f'{label1} vs {label2}')
    plt.savefig(path_to_fig)
    plt.close()


def plot_data(data1, data2, label1, label2, xlabel, ylabel, location, path_to_fig):
    plt.plot(data1, label=label1)
    plt.plot(data2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=location)
    plt.title(f'{label1} vs {label2}')
    plt.savefig(path_to_fig)
    plt.close()


def load_dataset_torch():
    from torchvision import datasets, transforms
    manual_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=manual_transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=manual_transform
    )

    return train_dataset, test_dataset


def validate_trained_torch_model(model, data_loader, device, num_of_batches_to_run, quantized=False):
    import torch
    from torchprofile import profile_macs
    result_data = {}
    peak_data = []
    data_time = []
    macs = []

    model.eval()
    model = model.to(device)
    # calculate the final train and test accuracy for the model trained.
    with torch.no_grad():
        for index, (images, _) in enumerate(data_loader):
            if index == num_of_batches_to_run:
                break
            images = images.to(device)
            start = time()
            tracemalloc.start()
            model(images)
            data_time.append((time() - start))
            _, peak = tracemalloc.get_traced_memory()
            peak_data.append(peak)
            tracemalloc.reset_peak()
            tracemalloc.stop()
            if not quantized:
                macs.append(profile_macs(model, images))
    if num_of_batches_to_run == -1:
        return

    result_data['peak_mem_usage'] = max(peak_data)
    if not quantized:
        result_data['macs'] = np.average(macs).item()
    result_data['inference_time'] = (
        sum(data_time) / num_of_batches_to_run) * 1000  # msecs

    return result_data


def get_train_test_accuracy(model, train_loader, test_loader, train_len, test_len, k_in_topk, device):

    import torch
    model.eval()
    train_accuracy, test_accuracy = 0, 0
    train_accuracy_topk, test_accuracy_topk = 0, 0

    # calculate the final train and test accuracy for the model trained.
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            acc_value = accuracy(output, labels, (1, k_in_topk))
            train_accuracy += acc_value[0].item()
            train_accuracy_topk += acc_value[1].item()

        train_accuracy /= train_len
        train_accuracy_topk /= train_len

        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            acc_value = accuracy(output, labels, (1, k_in_topk))
            test_accuracy += acc_value[0].item()
            test_accuracy_topk += acc_value[1].item()

        test_accuracy /= test_len
        test_accuracy_topk /= test_len

    return train_accuracy, test_accuracy, train_accuracy_topk, test_accuracy_topk


def quantize_model(configuration_data, path_to_model, dataloader, device):
    import torch
    from torch_vgg16 import VGG as vgg
    from torch_mobilenet import mobilenet_v2

    if configuration_data['model_name'] == 'VGG16':
        model = vgg(configuration_data['image_size'], configuration_data['image_channels'],
                    configuration_data['num_classes']).to(device)
    else:
        model = mobilenet_v2(
            configuration_data['num_classes'], configuration_data['image_size'], True).to(device)

    state_dict = torch.load(path_to_model)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    validate_trained_torch_model(
        model, dataloader, torch.device(device), -1)
    torch.quantization.convert(model, inplace=True)
    return model


def train_torch_model(configuration_data):
    import torch
    import torch.nn as nn
    from torch_vgg16 import VGG as vgg
    from torch_mobilenet import mobilenet_v2
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    workers = 2
    batch_size = configuration_data['batch_size']
    lr = configuration_data['learning_rate']
    epochs = configuration_data['epochs']
    # momentum = configuration_data['momentum']

    train_dataset, _ = load_dataset_torch()
    length_train_dataset = len(train_dataset)

    train_length = int(configuration_data['train_size'] * length_train_dataset)
    validation_length = len(train_dataset) - train_length

    print(f"{train_length=} {validation_length=}")
    if validation_length != 0:
        train_dataset, validation_dataset = torch.utils.data.random_split(
            train_dataset, [train_length, validation_length]
        )
    else:
        # Incase we train on complete data set aside few samples from the traindata itself for validation-set testing.
        validation_length = 1024
        _, validation_dataset = torch.utils.data.random_split(
            train_dataset, [train_length - validation_length, validation_length])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    # loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    if configuration_data['model_name'] == 'VGG16':
        model = vgg(configuration_data['image_size'], configuration_data['image_channels'],
                    configuration_data['num_classes']).to(device)
    else:
        model = mobilenet_v2(
            configuration_data['num_classes'], configuration_data['image_size'], True).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,  # momentum=momentum
    )

    train_loss_data, validation_loss_data = [], []
    train_accuracy_data, validation_accuracy_data = [], []

    for epoch in range(epochs):
        curr_train_loss, curr_train_accuracy = 0, 0
        # train loss and accuracy.
        model.train()
        train_batch_iterator = tqdm(
            train_loader, desc=f"Training Epoch {epoch:03d}"
        )
        iter_count = 0
        for images, labels in train_batch_iterator:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            curr_train_loss += loss.item()
            actual_labels = labels.cpu().detach().numpy()
            predicted_labels = torch.argmax(torch.softmax(
                output, dim=1), dim=1).cpu().detach().numpy()
            curr_train_accuracy += np.sum(actual_labels == predicted_labels)
            iter_count += 1

        curr_train_loss /= (iter_count + 1)
        curr_train_accuracy /= train_length

        train_loss_data.append(curr_train_loss)
        train_accuracy_data.append(curr_train_accuracy)

        curr_validation_loss, curr_validation_accuracy = 0, 0
        # validation loss and accuracy

        model.eval()
        with torch.no_grad():
            batch_iter = tqdm(validation_loader,
                              desc=f"Validation Epoch {epoch:03d}")
            iter_count = 0
            for images, labels in batch_iter:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                curr_validation_loss += loss.item()
                actual_labels = labels.cpu().detach().numpy()
                pred_labels = torch.argmax(torch.softmax(
                    output, dim=1), dim=1).cpu().detach().numpy()
                curr_validation_accuracy += np.sum(
                    actual_labels == pred_labels)
                iter_count += 1

        curr_validation_loss /= (iter_count + 1)
        curr_validation_accuracy /= validation_length
        validation_loss_data.append(curr_validation_loss)
        validation_accuracy_data.append(curr_validation_accuracy)

        print(
            f"{epoch=}, Train Loss: {curr_train_loss:.4f}, Validation Loss: {curr_validation_loss:.4f}, Train Accuracy: {
                curr_train_accuracy * 100:.2f}%, Validation Accuracy: {curr_validation_accuracy * 100:.2f}%"
        )

    return model, train_loss_data, validation_loss_data, train_accuracy_data, validation_accuracy_data
