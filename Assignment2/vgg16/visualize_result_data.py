import os
import numpy as np
import matplotlib.pyplot as plt

quant_vgg_result_data = {'model_size': 134.361374, 'train_accuracy': 0.93148, 'test_accuracy': 0.7264, 'traink_accuracy': 0.98288, 'testk_accuracy': 0.9109, 'total_param_count': 0, 'plot_data': [
    {'peak_mem_usage': 1097, 'inference_time': 43.40565204620361, 'batch_size': 32},
    {'peak_mem_usage': 2492, 'inference_time': 84.13886070251465, 'batch_size': 64},
    {'peak_mem_usage': 2492, 'inference_time': 146.48892879486084, 'batch_size': 128},
    {'peak_mem_usage': 2492, 'inference_time': 217.17472076416016, 'batch_size': 256},
]}

non_quant_vgg_result_data = {'model_size': 537.218294, 'train_accuracy': 0.93722, 'test_accuracy': 0.7333, 'traink_accuracy': 0.98362, 'testk_accuracy': 0.9157, 'total_param_count': 134301514, 'plot_data': [
    {'peak_mem_usage': 1051, 'macs': 13849608192.0,
        'inference_time': 72.2225570678711, 'batch_size': 32},
    {'peak_mem_usage': 1171, 'macs': 27699216384.0,
        'inference_time': 127.29480266571045, 'batch_size': 64},
    {'peak_mem_usage': 2204, 'macs': 55398432768.0,
        'inference_time': 245.8530330657959, 'batch_size': 128},
    {'peak_mem_usage': 2492, 'macs': 108200064000.0,
        'inference_time': 385.2933359146118, 'batch_size': 256},
]}


quant_time_data, non_quant_time_data, non_quant_mac_data = {}, {}, {}
path_to_result_folder = './results/'

for plots_data in non_quant_vgg_result_data['plot_data']:
    batch_size_value = str(plots_data['batch_size'])
    non_quant_time_data[batch_size_value] = plots_data['inference_time']
    non_quant_mac_data[batch_size_value] = plots_data['macs']

for plots_data in quant_vgg_result_data['plot_data']:
    batch_size_value = str(plots_data['batch_size'])
    quant_time_data[batch_size_value] = plots_data['inference_time']


non_quant_acc_data = {'Top1 Train': round(non_quant_vgg_result_data['train_accuracy'] * 100, 2),
                      'Top3 Train': round(non_quant_vgg_result_data['traink_accuracy'] * 100, 2),
                      'Top1 Test': round(non_quant_vgg_result_data['test_accuracy'] * 100, 2),
                      'Top3 Test': round(non_quant_vgg_result_data['testk_accuracy'] * 100, 2)
                      }

quant_acc_data = {'Top1 Train': round(quant_vgg_result_data['train_accuracy'] * 100, 2),
                  'Top3 Train': round(quant_vgg_result_data['traink_accuracy'] * 100, 2),
                  'Top1 Test': round(quant_vgg_result_data['test_accuracy'] * 100, 2),
                  'Top3 Test': round(quant_vgg_result_data['testk_accuracy'] * 100, 2)
                  }

x_values = np.arange(len(non_quant_acc_data.keys()))

non_quant_bar_data = plt.bar(
    x_values - 0.2, non_quant_acc_data.values(), 0.35, label='Non Quanitzed Model')
quant_bar_data = plt.bar(
    x_values + 0.2, quant_acc_data.values(), 0.35, label='Quantized Model')

plt.xticks(x_values, quant_acc_data.keys())
plt.legend(loc='lower right')
for rect in non_quant_bar_data + quant_bar_data:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0,
             height, height, ha='center', va='bottom')

plt.xlabel('Type')
plt.ylabel('Accuracy')
plt.title('Effect of Quantization on Accuracy')
plt.savefig(os.path.join(path_to_result_folder, 'accuracy_data.png'), bbox_inches='tight')
plt.close()


model_size_data = {'Quantized Model': quant_vgg_result_data['model_size'],
                   'Non-Quantized Model': non_quant_vgg_result_data['model_size']}

bar_data = plt.bar(model_size_data.keys(), model_size_data.values())
for rect in bar_data:
    height = rect.get_height()
    text_value = f'{height} MB'
    plt.text(rect.get_x() + rect.get_width() / 2.0,
             height, text_value, ha='center', va='bottom')


plt.xlabel('Model Type')
plt.ylabel('Model Size')
plt.title('Effect of Quantization on Model Size')
plt.savefig(os.path.join(path_to_result_folder, 'model_size.png'), bbox_inches='tight')
plt.close()


plt.plot(quant_time_data.keys(), quant_time_data.values(), label='Quantized Model')
plt.plot(non_quant_time_data.keys(), non_quant_time_data.values(), label='Non-Quantized Model')
plt.legend(loc='lower right')
plt.xlabel('Batch Size')
plt.ylabel('Inference Time (in milli-seconds)')
plt.title('Effect of batch-size on inference time')
plt.savefig(os.path.join(path_to_result_folder, 'inference_time.png'), bbox_inches='tight')
plt.close()


bar_data = plt.bar(non_quant_mac_data.keys(), non_quant_mac_data.values())
for rect in bar_data:
    height = rect.get_height()
    value = round(height.item() / 10 ** 9, 2)
    text_value = f'{value} G'
    plt.text(rect.get_x() + rect.get_width() / 2.0,
             height, text_value, ha='center', va='bottom')

plt.title('Effect of batch-size on MAC operations (Non-Quantized Model)')
plt.xlabel('Batch Size')
plt.ylabel('Multiple and Accumlate Operation Count')
plt.yscale('log')
plt.savefig(os.path.join(path_to_result_folder, 'macs.png'), bbox_inches='tight')
plt.close()


