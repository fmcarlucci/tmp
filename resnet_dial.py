import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, n_domains=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, n_domains)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, n_domains)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, set_id):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, set_id)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, set_id)
        if self.downsample is not None:
            identity = self.downsample((x, set_id))

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, n_domains=1):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.num_domains = n_domains
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes, n_domains)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DIALSequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, num_domains=self.num_domains),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, n_domains=self.num_domains))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, n_domains=self.num_domains))

        return DIALSequential(*layers)

    
    def forward(self, x):
        x, set_id = x
        x = self.conv1(x)
        x = self.bn1(x,set_id)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1((x,set_id))
        x = self.layer2((x,set_id))
        x = self.layer3((x,set_id))
        x = self.layer4((x,set_id))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

import torch
import torch.nn as nn

class DIALSequential(nn.Sequential):
    def forward(self, x):
        x, set_id = x
        for module in self._modules.values():
            if isinstance(module, MSAutoDIAL):
                x = module(x, set_id)
            elif isinstance(module, BasicBlock):
                x = module(x, set_id)
            else:
                x = module(x)
        return x

def _index_mean(x, idx, num):
    if x.ndimension() > 2:
        x = x.view(x.size(0), x.size(1), -1).mean(-1)

    mean = x.new_zeros(num, x.size(1))
    mean.index_add_(0, idx, x)

    count = x.new_zeros(num)
    count.index_add_(0, idx, count.new_ones(idx.numel()))

    mean = mean * torch.clamp(1. / count, max=1.).view(-1, 1)
    return mean, count


def _global_mean(x):
    if x.ndimension() > 2:
        x = x.view(x.size(0), x.size(1), -1).mean(-1)
    return x.mean(0)


def _broadcast_shapes(x):
    return (x.size(0), x.size(1)) + (1,) * (x.ndimension() - 2), (1, x.size(1)) + (1,) * (x.ndimension() - 2)


def _spatial_size(x):
    size = 1
    for i in range(2, x.ndimension()):
        size *= x.size(i)
    return size

class MSAutoDIAL(nn.Module):
    def __init__(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True):
        super(MSAutoDIAL, self).__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.eps = eps
        self.momentum = 1. - momentum
        self.affine = affine
        self.alpha = nn.Parameter(torch.ones(()))
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_domains + 1, num_features))
        self.register_buffer('running_var', torch.zeros(num_domains + 1, num_features))
        self.register_buffer('running_num', torch.zeros(num_domains + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.alpha.data.fill_(1)
        self.running_mean.zero_()
        self.running_var.zero_()
        self.running_num.zero_()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def extra_repr(self):
        return '{num_domains} x {num_features}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Compatibility with standard BN: add alpha, running_num if not present, expand running_mean and running_var
        alpha_key = prefix + "alpha"
        if alpha_key not in state_dict:
            state_dict[alpha_key] = torch.ones_like(self.alpha)

        running_num_key = prefix + "running_num"
        if running_num_key not in state_dict:
            state_dict[running_num_key] = torch.ones_like(self.running_num)

        running_mean_key = prefix + "running_mean"
        if running_mean_key in state_dict and state_dict[running_mean_key].ndimension() == 1:
            state_dict[running_mean_key] = state_dict[running_mean_key].unsqueeze(0).repeat(self.num_domains + 1, 1)

        running_var_key = prefix + "running_var"
        if running_var_key in state_dict and state_dict[running_var_key].ndimension() == 1:
            state_dict[running_var_key] = state_dict[running_var_key].unsqueeze(0).repeat(self.num_domains + 1, 1)

        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(MSAutoDIAL, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x, set_id):
        with torch.no_grad():
            self.alpha.clamp_(min=0, max=1)
        bs0, bs1 = _broadcast_shapes(x)

        # Get domain-specific
        if self.training:
            # Local stats
            mean_d, count_d = _index_mean(x, set_id, self.num_domains)
            var_d, _ = _index_mean((x - mean_d[set_id].view(bs0)) ** 2, set_id, self.num_domains)

            # Global stats
            mean_g = _global_mean(x)
            var_g = _global_mean((x - mean_g.view(bs1)) ** 2)

            # Update moving stats
            with torch.no_grad():
                mean = torch.cat([mean_g.view(1, -1), mean_d], dim=0)
                var = torch.cat([var_g.view(1, -1), var_d], dim=0)
                count = torch.cat([count_d.sum().view(-1), count_d], dim=0)

                var_norm_factor = (count * _spatial_size(x)) / (count * _spatial_size(x) - 1)

                self.running_num.mul_(self.momentum).add_(count)
                self.running_mean.mul_(self.momentum).add_(count.view(-1, 1) * mean)
                self.running_var.mul_(self.momentum).add_(
                    count.view(-1, 1) * var * torch.clamp(var_norm_factor, max=2).view(-1, 1))
        else:
            mean = self.running_mean / self.running_num.view(-1, 1)
            var = self.running_var / self.running_num.view(-1, 1)

            mean_g, mean_d = mean[0, :], mean[1:, :]
            var_g, var_d = var[0, :], var[1:, :]

        # Mixed stats
        mean = mean_d * self.alpha + mean_g * (1 - self.alpha)
        var = var_d * self.alpha + var_g * (1 - self.alpha) + self.alpha * (1 - self.alpha) * (mean_g - mean_d) ** 2

        # Normalize
        y = (x - mean[set_id].view(bs0)) * torch.rsqrt(var[set_id] + self.eps).view(bs0)
        if self.affine:
            y.mul_(self.weight.view(bs1)).add_(self.bias.view(bs1))

        return y
