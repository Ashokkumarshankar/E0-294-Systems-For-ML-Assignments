import os
from smallNet import *

from utils import test, load_data


def main():
    device = torch.device("cpu")
    batch_size = 64
    result_folder_name = 'Results'
    model_name = "mnist_cnn_cpu.pt"

    # load float32 model
    model = SmallNet().to(device)
    model.load_state_dict(torch.load(os.path.join(result_folder_name, model_name)))
    _, test_loader = load_data(batch_size, False)

    print('Before Quantization')
    # Code for memory usage
    test(model, device, test_loader)
    print('Size of the model(MB):', os.path.getsize(os.path.join(result_folder_name, model_name))/1e6)

if __name__ == "__main__":
    main()
