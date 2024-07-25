import torch
import torch.nn as nn
import math

# Reference :- https://lekhuyen.medium.com/an-overview-of-vgg16-and-nin-models-96e4bf398484


class VGG(nn.Module):
    def __init__(self, image_shape, num_of_channels=3, num_of_classes=10):
        super(VGG, self).__init__()

        pooling_kernel_size, pooling_stride = 2, 2
        final_pool_size = 7
        conv_kern_size, conv_padding = 3, [1]
        dropout_ratio = 0.5
        output_size = 4096

        # a list descriping VGG16 dimensions
        layer_data = [
            64,  # 3 x 3, same padding
            64,  # 3 x 3, same padding
            "M",  # 2 x 2, stride 2
            128,  # 3 x 3, same padding
            128,  # 3 x 3, same padding
            "M",  # 2 x 2, stride 2
            256,  # 3 x 3, same padding
            256,  # 3 x 3, same padding
            256,  # 3 x 3, same padding
            "M",  # 2 x 2, stride 2
            512,  # 3 x 3, same padding
            512,  # 3 x 3, same padding
            512,  # 3 x 3, same padding
            "M",  # 2 x 2, stride 2
            512,  # 3 x 3, same padding
            512,  # 3 x 3, same padding
            512,  # 3 x 3, same padding
        ]

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.listModule = []
        self.in_channels = num_of_channels

        for layer in layer_data:
            if layer == "M":
                self.listModule += [
                    nn.MaxPool2d(
                        kernel_size=pooling_kernel_size, stride=pooling_stride)
                ]
            else:
                assert (isinstance(layer, int))
                kernel_count = layer
                self.listModule += [
                    nn.Conv2d(self.in_channels, kernel_count,
                              kernel_size=conv_kern_size, padding=1),
                    # nn.BatchNorm2d(kernel_count), Ideally should be there but since not present in actual paper so not including it.
                    nn.ReLU(inplace=True)
                ]
                self.in_channels = kernel_count

        self.features = nn.Sequential(*self.listModule)
        # making sure the shape is final_pool_size x final_pool_size x filters.
        self.avgpool = nn.AdaptiveAvgPool2d((final_pool_size))
        self.linear_layers = [
            nn.Linear(self.in_channels * final_pool_size *
                      final_pool_size, output_size),
            nn.ReLU(True),
            nn.Dropout(dropout_ratio),
            nn.Linear(output_size, output_size),
            nn.ReLU(True),
            nn.Dropout(dropout_ratio),
            nn.Linear(output_size, num_of_classes),
        ]
        self.classifier = nn.Sequential(*self.linear_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x
