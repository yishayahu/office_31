from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn, Tensor
from torchvision.models.densenet import _DenseLayer
from torch.nn import functional as F

class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
            self,
            num_layers: int,
            num_input_features: int,
            bn_size: int,
            growth_rate: int,
            drop_rate: float,
            memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        self.layers = []
        self.par_layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                )
            self.layers.append(layer)
            self.par_layers.append(deepcopy(layer))
        self.layers = nn.ModuleList(self.layers)
        self.par_layers = nn.ModuleList(self.par_layers)

    def forward(self, init_features: Tensor,policy) -> Tensor:
        features = [init_features]
        for t,(layer, par_layer) in enumerate(zip(self.layers,self.par_layers)):
            action = policy[:,t].contiguous()
            action_mask = action.float().view(-1,1,1,1)
            new_features = layer(features) * (1-action_mask) + par_layer(features)* action_mask
            features.append(new_features)
        return torch.cat(features, 1)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]'):
        layers = {}
        for k,v in state_dict.items():
            try:
                layer_num = int(k.split('.')[0][-2:]) - 1
            except:
                layer_num = int(k.split('.')[0][-1]) - 1
            if layer_num not in layers:
                layers[layer_num] = {}
            layers[layer_num]['.'.join(k.split('.')[1:])] = v
        for k,v in layers.items():
            self.layers[k].load_state_dict(v)
            self.par_layers[k].load_state_dict(deepcopy(v))


class StDenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    def __init__(
            self,
            growth_rate: int = 32,
            block_config = (6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 4,
            drop_rate: float = 0,
            num_classes: int = 28,
            memory_efficient: bool = False
    ) -> None:

        super(StDenseNet, self).__init__()
        self.num_layers_to_policy = 0
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU()),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        cc = 0
        self.par_features = deepcopy(self.features) #0
        self.blocks = []
        self.num_layers_to_policy += 1
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.blocks.append((block,self.num_layers_to_policy,self.num_layers_to_policy+num_layers))
            self.add_module(f'{cc}',block)
            cc+=1
            self.num_layers_to_policy += num_layers

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                trans2 = deepcopy(trans)
                self.blocks.append((trans,trans2,self.num_layers_to_policy))
                self.add_module(f'{cc}',trans)
                cc+=1
                self.add_module(f'{cc}',trans2)
                cc+=1
                self.num_layers_to_policy+=1
                num_features = num_features // 2

        # Final batch norm
        bn = nn.BatchNorm2d(num_features)
        self.blocks.append((bn,None,None))
        self.add_module(f'{cc}',bn)


        # Linear layer
        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Sigmoid()
        )



    def forward(self, x: Tensor,policy) -> Tensor:
        first_action = policy[:,0].contiguous()
        first_action_mask = first_action.float().view(-1,1,1,1)
        x = self.features(x) *(1-first_action_mask) + self.features(x)* first_action_mask
        for block, start,end in self.blocks:
            if start is  None:
                x = block(x)

            elif type(start) is int:
                x= block(x,policy[:,start:end])
            else:
                action = policy[:,end].contiguous()
                action_mask = action.float().view(-1,1,1,1)
                x = block(x) * (1-action_mask) + start(x) * action_mask

        out = F.relu(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]'):
        first_feats = {}
        trans = {}
        bn5 ={}
        classifieer ={}
        blocks = {}
        for k,v in state_dict.items():
            if 'densenet121.features.conv0' in k or 'densenet121.features.norm0' in k:
                first_feats[k.replace('densenet121.features.','')] = v
            elif 'denseblock' in k:
                k = k.replace('densenet121.features.','')
                block_num = int(k.split('.')[0][-1]) - 1
                if block_num not in blocks:
                    blocks[block_num] = {}
                blocks[block_num]['.'.join(k.split('.')[1:])] = v
            elif 'transition' in k:
                k = k.replace('densenet121.features.','')
                trans_num = int(k.split('.')[0][-1]) - 1
                if trans_num not in trans:
                    trans[trans_num] = {}
                trans[trans_num]['.'.join(k.split('.')[1:])] = v
            elif 'densenet121.features.norm5' in k:
                bn5[k.replace('densenet121.features.norm5.','')] = v
            elif 'densenet121.classifier' in k:
                classifieer[k.replace('densenet121.classifier.','')] = v
            else:
                assert False

        self.features.load_state_dict(first_feats)
        self.par_features.load_state_dict(deepcopy(first_feats))
        self.blocks[-1][0].load_state_dict(bn5)
        self.classifier.load_state_dict(classifieer)
        for i,blk in enumerate(self.blocks):
            if blk[1] is None:
                continue
            if i % 2 == 0:
                blk[0].load_state_dict(blocks[i//2])
            else:
                blk[0].load_state_dict(trans[i//2])
                blk[1].load_state_dict(deepcopy(trans[i//2]))
