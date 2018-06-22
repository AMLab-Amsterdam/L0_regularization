import torch
import torch.nn as nn
from l0_layers import L0Conv2d, L0Dense
from base_layers import MAPConv2d, MAPDense
from utils import get_flat_fts
from copy import deepcopy
import torch.nn.functional as F


class L0MLP(nn.Module):
    def __init__(self, input_dim, num_classes, layer_dims=(300, 100), N=50000, beta_ema=0.999,
                 weight_decay=1, lambas=(1., 1., 1.), local_rep=False, temperature=2./3.):
        super(L0MLP, self).__init__()
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.N = N
        self.beta_ema = beta_ema
        self.weight_decay = self.N * weight_decay
        self.lambas = lambas

        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
            droprate_init, lamba = 0.2 if i == 0 else 0.5, lambas[i] if len(lambas) > 1 else lambas[0]
            layers += [L0Dense(inp_dim, dimh, droprate_init=droprate_init, weight_decay=self.weight_decay,
                               lamba=lamba, local_rep=local_rep, temperature=temperature), nn.ReLU()]

        layers.append(L0Dense(self.layer_dims[-1], num_classes, droprate_init=0.5, weight_decay=self.weight_decay,
                              lamba=lambas[-1], local_rep=local_rep, temperature=temperature))
        self.output = nn.Sequential(*layers)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        return self.output(x)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params


class L0LeNet5(nn.Module):
    def __init__(self, num_classes, input_size=(1, 28, 28), conv_dims=(20, 50), fc_dims=500,
                 N=50000, beta_ema=0., weight_decay=1, lambas=(1., 1., 1., 1.), local_rep=False,
                 temperature=2./3.):
        super(L0LeNet5, self).__init__()
        self.N = N
        assert(len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay

        convs = [L0Conv2d(input_size[0], conv_dims[0], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[0], local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2),
                 L0Conv2d(conv_dims[0], conv_dims[1], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas[1], local_rep=local_rep),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs)
        fcs = [L0Dense(flat_fts, self.fc_dims, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[2], local_rep=local_rep, temperature=temperature), nn.ReLU(),
               L0Dense(self.fc_dims, num_classes, droprate_init=0.5, weight_decay=self.weight_decay,
                       lamba=lambas[3], local_rep=local_rep, temperature=temperature)]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense) or isinstance(m, L0Conv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        return self.fcs(o)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate_init=0.0, weight_decay=0., lamba=0.01, local_rep=False,
                 temperature=2./3.):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = L0Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
                              droprate_init=droprate_init, weight_decay=weight_decay / (1 - 0.3), local_rep=local_rep,
                              lamba=lamba, temperature=temperature)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_decay=weight_decay)

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                      weight_decay=weight_decay) or None

    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.add(out, x if self.equalInOut else self.convShortcut(x))


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate_init=0.0, weight_decay=0., lamba=0.01,
                 local_rep=False, temperature=2./3.):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate_init,
                                      weight_decay=weight_decay, lamba=lamba, local_rep=local_rep,
                                      temperature=temperature)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate_init,
                    weight_decay=0., lamba=0.01, local_rep=False, temperature=2./3.):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1,
                                droprate_init, weight_decay, lamba, local_rep=local_rep, temperature=temperature))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class L0WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, droprate_init=0.3, N=50000, beta_ema=0.99,
                 weight_decay=5e-4, local_rep=False, lamba=0.01, temperature=2./3.):
        super(L0WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        self.n = (depth - 4) // 6
        self.N = N
        self.beta_ema = beta_ema
        block = BasicBlock

        self.weight_decay = N * weight_decay
        self.lamba = lamba

        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=self.weight_decay)
        # 1st block
        self.block1 = NetworkBlock(self.n, nChannels[0], nChannels[1], block, 1, droprate_init, self.weight_decay,
                                   self.lamba, local_rep=local_rep, temperature=temperature)
        # 2nd block
        self.block2 = NetworkBlock(self.n, nChannels[1], nChannels[2], block, 2, droprate_init, self.weight_decay,
                                   self.lamba, local_rep=local_rep, temperature=temperature)
        # 3rd block
        self.block3 = NetworkBlock(self.n, nChannels[2], nChannels[3], block, 2, droprate_init, self.weight_decay,
                                   self.lamba, local_rep=local_rep, temperature=temperature)
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.fcout = MAPDense(nChannels[3], num_classes, weight_decay=self.weight_decay)

        self.layers, self.bn_params = [], []
        for m in self.modules():
            if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, L0Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        for bnw in self.bn_params:
            if self.weight_decay > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            try:
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0 += e_l0
            except:
                pass
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params
