import os
import yaml
from utils import train_torch_model, set_seed, PATH_TO_CONFIG_FILE, PATH_TO_RESULT_FILES, plot_loss_accuracy_data, plot_data, validate_trained_torch_model, quantize_model, load_dataset_torch, get_train_test_accuracy


def torch_model(configurations):

    from torch import device, save
    from torch.utils.data import DataLoader

    os.makedirs(PATH_TO_RESULT_FILES, exist_ok=True)

    non_quant_result_data, quant_result_data = {}, {}
    batches_data = [32, 64, 128, 256]

    # Training on VGG Model GPU.
    model_name = configurations['model_name']
    train_batch_size = configurations['batch_size']
    cpu_device = device('cpu')

    model_file_name = f'{model_name}_{
        configurations["platform"]}_{train_batch_size}.pt'
    path_to_quantized_model = os.path.join(
        configurations['path_to_trained_model'], f'quant_{model_file_name}')
    path_to_nonquantized_model = os.path.join(
        configurations['path_to_trained_model'], f'nonquant_{model_file_name}')

    model, train_loss_data, valid_loss_data, train_acc_data, valid_acc_data = train_torch_model(
        configurations)

    plot_loss_accuracy_data(train_loss_data, train_acc_data, 'Training Loss',
                            'Training Accuracy', 'Epoch number', 'Loss', 'Accuracy', os.path.join(PATH_TO_RESULT_FILES, f'{model_name}_train_acc_loss.png'))
    plot_data(train_loss_data, valid_loss_data, "Training Loss",
              "Validation Loss", 'Epoch number', 'Loss', 'upper right', os.path.join(PATH_TO_RESULT_FILES, f'{model_name}_loss_plot.png'))
    plot_data(train_acc_data, valid_acc_data, "Training Accuracy",
              "Validation Accuracy", 'Epoch number', 'Accuracy', 'upper left', os.path.join(PATH_TO_RESULT_FILES, f'{model_name}_acc_plot.png'))

    # Save
    save(model.state_dict(), path_to_nonquantized_model)

    # Get Data loader for finding accu and inference timing.

    train_dataset, test_dataset = load_dataset_torch()
    complete_train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    complete_test_loader = DataLoader(
        test_dataset, batch_size=train_batch_size, shuffle=False
    )
    length_train_dataset, length_test_dataset = len(
        train_dataset), len(test_dataset)

    # '''
    # For the batch_sizes provided to calculate before and after post training quantization :-
    # 1. train and test accuracy both top1 and k.
    # 2. # of parameters.
    # 3. flops
    # 4. inference time
    # 5. memory utilized
    # '''

    # finding final accu.
    train_acc, test_acc, train_acc_topk, test_acc_topk = get_train_test_accuracy(
        model, complete_train_loader, complete_test_loader, length_train_dataset, length_test_dataset, configurations['topk'], device('cuda'))

    model_size = os.path.getsize(path_to_nonquantized_model)/1e6
    non_quant_result_data['model_size'] = model_size
    non_quant_result_data['train_accuracy'] = train_acc
    non_quant_result_data['test_accuracy'] = test_acc
    non_quant_result_data['traink_accuracy'] = train_acc_topk
    non_quant_result_data['testk_accuracy'] = test_acc_topk
    non_quant_result_data['total_param_count'] = sum(
        p.numel() for p in model.parameters())

    model = model.to(cpu_device)
    print(model)
    result_data = []
    # Inference on CPU for diff batch-sizes
    for batch_size in batches_data:
        complete_test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        batch_result = validate_trained_torch_model(
            model, complete_test_loader, cpu_device, configurations['num_batches'])
        batch_result['batch_size'] = batch_size
        result_data.append(batch_result)
    non_quant_result_data['plot_data'] = result_data

    # quantize the model.
    complete_train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    complete_test_loader = DataLoader(
        test_dataset, batch_size=train_batch_size, shuffle=False
    )
    length_train_dataset, length_test_dataset = len(
        train_dataset), len(test_dataset)

    quantized_model = quantize_model(configurations,
                                     path_to_nonquantized_model, complete_train_loader, cpu_device)
    print(quantized_model)

    save(quantized_model.state_dict(), path_to_quantized_model)
    train_acc, test_acc, train_acc_topk, test_acc_topk = get_train_test_accuracy(
        quantized_model, complete_train_loader, complete_test_loader, length_train_dataset, length_test_dataset, configurations['topk'], cpu_device)

    model_size = os.path.getsize(path_to_quantized_model)/1e6
    quant_result_data['model_size'] = model_size
    quant_result_data['train_accuracy'] = train_acc
    quant_result_data['test_accuracy'] = test_acc
    quant_result_data['traink_accuracy'] = train_acc_topk
    quant_result_data['testk_accuracy'] = test_acc_topk
    quant_result_data['total_param_count'] = sum(
        p.numel() for p in quantized_model.parameters())

    result_data = []
    for batch_size in batches_data:
        complete_test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        batch_result = validate_trained_torch_model(
            quantized_model, complete_test_loader, cpu_device, configurations['num_batches'], True)
        batch_result['batch_size'] = batch_size
        result_data.append(batch_result)
    quant_result_data['plot_data'] = result_data

    print(f'Results for Non-Quantized {model_name} = {non_quant_result_data}')
    print(f'Results for Quantized {model_name} = {quant_result_data}')


def main():
    # os.makedirs(PATH_TO_RESULT_FILES, exist_ok=True)
    with open(PATH_TO_CONFIG_FILE) as f:
        configurations = yaml.load(f, Loader=yaml.FullLoader)

    to_run = configurations['platform']

    set_seed(configurations.get('seed', 41), to_run)
    if to_run == 'torch':
        # Run torch related funcs.
        torch_model(configurations)
    return 0


if __name__ == '__main__':
    main()
