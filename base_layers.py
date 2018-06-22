import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.nn import init


class MAPDense(Module):

    def __init__(self, in_features, out_features, bias=True, weight_decay=1., **kwargs):
        super(MAPDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weight, mode='fan_out')

        if self.bias is not None:
            self.bias.data.normal_(0, 1e-2)

    def constrain_parameters(self, **kwargs):
        pass

    def _reg_w(self, **kwargs):
        logpw = - torch.sum(self.weight_decay * .5 * (self.weight.pow(2)))
        logpb = 0
        if self.bias is not None:
            logpb = - torch.sum(self.weight_decay * .5 * (self.bias.pow(2)))
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        expected_flops = (2 * self.in_features - 1) * self.out_features
        expected_l0 = self.in_features * self.out_features
        if self.bias is not None:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        return expected_flops, expected_l0

    def forward(self, input):
        output = input.mm(self.weight)
        if self.bias is not None:
            output.add_(self.bias.view(1, self.out_features).expand_as(output))
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', weight_decay: ' \
            + str(self.weight_decay) + ')'


class MAPConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 weight_decay=1., **kwargs):
        super(MAPConv2d, self).__init__()
        self.weight_decay = weight_decay
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.input_shape = None
        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weight, mode='fan_in')

        if self.bias is not None:
            self.bias.data.normal_(0, 1e-2)

    def constrain_parameters(self, thres_std=1.):
        pass

    def _reg_w(self, **kwargs):
        logpw = - torch.sum(self.weight_decay * .5 * (self.weight.pow(2)))
        logpb = 0
        if self.bias is not None:
            logpb = - torch.sum(self.weight_decay * .5 * (self.bias.pow(2)))
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        ppos = self.out_channels
        n = self.kernel_size[0] * self.kernel_size[1] * self.in_channels  # vector_length
        flops_per_instance = n + (n - 1)  # (n: multiplications and n-1: additions)

        num_instances_per_filter = ((self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0]) + 1  # for rows
        num_instances_per_filter *= ((self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1]) + 1  # multiplying with cols

        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos  # multiply with number of filters
        expected_l0 = n * ppos

        if self.bias is not None:
            # since the gate is applied to the output we also reduce the bias computation
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos

        return expected_flops, expected_l0

    def forward(self, input_):
        if self.input_shape is None:
            self.input_shape = input_.size()
        output = F.conv2d(input_, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size} '
             ', stride={stride}, weight_decay={weight_decay}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)