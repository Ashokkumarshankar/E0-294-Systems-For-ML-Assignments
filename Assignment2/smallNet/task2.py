import os
from smallNet import *
from utils import test, load_data


def main():
    results_folder_name = 'Results'
    model_name = "mnist_cnn_cpu.pt"
    quant_model_name = 'mnist_cnn_int8.pt'
    batch_size = 64
    os.makedirs(results_folder_name, exist_ok=True)
    device = torch.device("cpu")
    train_loader, test_loader = load_data(batch_size, False)

    # load float32 model
    model_fp32 = SmallNet().to(device)
    model_fp32.load_state_dict(torch.load(
        os.path.join(results_folder_name, model_name)))

    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_fp32_fused = torch.quantization.fuse_modules(
        model_fp32, [['conv1', 'relu1'], ['conv2', 'relu2']])
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
    model_fp32_prepared.eval()
    with torch.no_grad():
        for source, target in train_loader:
            model_fp32_prepared(source)

    model_int8 = torch.quantization.convert(model_fp32_prepared)
    with torch.no_grad():
        for source, target in train_loader:
            model_int8(source)

    test(model_int8, device, test_loader)
    torch.save(model_int8.state_dict(), os.path.join(results_folder_name, quant_model_name))

if __name__ == '__main__':
    main()
