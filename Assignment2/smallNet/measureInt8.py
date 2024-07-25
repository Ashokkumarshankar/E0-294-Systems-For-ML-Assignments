from smallNet import *
from utils import load_data, test
import os

def main():
    device = torch.device("cpu")
    batch_size = 64
    result_folder_name = 'Results'
    quant_model_name = "mnist_cnn_int8.pt"
    model = SmallNet().to(device)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_fp32_fused = torch.quantization.fuse_modules(model, [['conv1', 'relu1'], ['conv2', 'relu2']])
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

    model_int8 = torch.quantization.convert(model_fp32_prepared)

    # loading your quantization model
    model_int8.load_state_dict(torch.load(os.path.join(result_folder_name, quant_model_name)))
    # do your measurements
    _, test_loader = load_data(batch_size, False)
    # do your measurements
    print('After Quantization')
    # Code for memory usage
    test(model_int8, device, test_loader)
    print('Size of the model(MB):', os.path.getsize(os.path.join(result_folder_name, quant_model_name))/1e6) 
    return 0


if __name__ == "__main__":
    main()