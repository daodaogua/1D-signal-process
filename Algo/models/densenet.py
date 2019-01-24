
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

class _DenseLayer(nn.Module):
    def __init__(self, input_features_num, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm1d(input_features_num)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv1d(input_features_num, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _Transition(nn.Sequential):
    def __init__(self, input_features_num, output_features_num):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(input_features_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(input_features_num, output_features_num,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=3, stride=2))

class _DenseBlock(nn.Module):
    def __init__(self, layers_num, input_features_num, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(layers_num):
            layer = _DenseLayer(
                input_features_num + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

class DenseNet(nn.Module):
    r"""DenseNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args in hyperparam:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        init_features_num (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        out_features_num (int) - the number of filter to learn in the last layer
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, hyperparam):

        super(DenseNet, self).__init__()

        growth_rate = hyperparam.get('growth_rate', 24)
        block_config = hyperparam.get('block_config', (6,6,6))
        compression = hyperparam.get('compression', 0.5)
        init_features_num = hyperparam.get('init_features_num', growth_rate*2)
        out_features_num = hyperparam.get('out_features_num', 4)
        efficient = hyperparam.get('efficient', False)
        drop_rate = hyperparam.get('drop_rate', 0)
        bn_size = hyperparam.get('bn_size', 4)
        input_features_num = hyperparam.get('input_features_num')
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(input_features_num, init_features_num, kernel_size=7, stride=1, padding=1, bias=False)),
        ]))
        self.features.add_module('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1,ceil_mode=False))

        # Each denseblock
        features_num = init_features_num
        for i, layers_num in enumerate(block_config):
            block = _DenseBlock(
                layers_num=layers_num,
                input_features_num=features_num,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            features_num = features_num + layers_num * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(input_features_num=features_num,
                                    output_features_num=int(features_num * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                features_num = int(features_num * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm1d(features_num))

        # Linear layer
        self.linear_features = nn.Linear(features_num, out_features_num)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'linear_features' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool1d(out, kernel_size=features.size(-1)).view(features.size(0), -1)
        out = self.linear_features(out)
        return out
