import torch.nn as nn
from torch.nn.modules.utils import _single
import torchsummary as summary
import torch.nn.functional as F
import torch as t
class Conv1d_same(nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,padding_mode='zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d_same, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias,padding_mode)

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)

def conv1d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                         input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    # padding_cols = max(0, (out_rows - 1) * stride[0] +
    # (filter_rows - 1) * dilation[0] + 1 - input_rows)
    # padding_cols = 0
    # cols_odd = (padding_rows % 2 != 0)
    if rows_odd:
        input = F.pad(input, [0, int(rows_odd)])
    return F.conv1d(input, weight, bias, stride,
                    padding=padding_rows // 2,
                    dilation=dilation, groups=groups)

class MaxPool1d_same(nn.modules.pooling._MaxPoolNd):
    def pool_same_padding(self,input, kernel_size, stride=1,
                            padding=1, dilation=1,
                            ):
        input_rows = input.size(2)
        filter_rows = kernel_size
        effective_filter_size_rows = (filter_rows - 1) * dilation + 1
        out_rows = (input_rows + stride - 1) // stride
        padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows -
                             input_rows)
        padding_rows = max(0, (out_rows - 1) * stride +
                           (filter_rows - 1) * dilation + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        # padding_cols = max(0, (out_rows - 1) * stride[0] +
        # (filter_rows - 1) * dilation[0] + 1 - input_rows)
        # padding_cols = 0
        # cols_odd = (padding_rows % 2 != 0)
        if rows_odd:
            input = F.pad(input, [0, int(rows_odd)])
        return F.max_pool1d(input, kernel_size, stride,
                            padding=padding_rows//2, dilation=dilation, ceil_mode=self.ceil_mode,
                            return_indices=self.return_indices)
    def forward(self, input):
        return self.pool_same_padding(input, self.kernel_size, self.stride,self.padding, self.dilation)

class DFModel(nn.Module):
    def __init__(self,out_features):
        super().__init__()
        filter_num = ['None', 32, 64, 128, 256]
        kernel_size = ['None', 8, 8, 8, 8]
        conv_stride_size = ['None', 1, 1, 1, 1]
        pool_stride_size = ['None', 4, 4, 4, 4]
        pool_size = ['None', 8, 8, 8, 8]

        self.con1 = nn.Sequential()
        self.con1.add_module('block1_conv1', Conv1d_same(1, filter_num[1], kernel_size=kernel_size[1],
                                                         stride=conv_stride_size[1]))
        self.con1.add_module('block1_bn1', nn.BatchNorm1d(filter_num[1]))
        self.con1.add_module('block1_act1', nn.ELU(alpha=1.0))
        self.con1.add_module('block1_conv2', Conv1d_same(filter_num[1], filter_num[1], kernel_size=kernel_size[1],
                                                         stride=conv_stride_size[1]))
        self.con1.add_module('block1_bn2', nn.BatchNorm1d(filter_num[1]))
        self.con1.add_module('block1_act2', nn.ELU(alpha=1.0))
        self.con1.add_module('block1_pool', MaxPool1d_same(kernel_size=pool_size[1], stride=pool_stride_size[1],
                                                           ))
        self.con1.add_module('block1_dropout', nn.Dropout(p=0.1))

        self.con2 = nn.Sequential()
        self.con2.add_module('block2_conv1', Conv1d_same(filter_num[1], filter_num[2], kernel_size=kernel_size[2],
                                                         stride=conv_stride_size[2]))
        self.con2.add_module('block2_bn1', nn.BatchNorm1d(filter_num[2]))
        self.con2.add_module('block2_act1', nn.ReLU())
        self.con2.add_module('block2_conv2', Conv1d_same(filter_num[2], filter_num[2], kernel_size=kernel_size[2],
                                                         stride=conv_stride_size[2]))
        self.con2.add_module('block2_bn2', nn.BatchNorm1d(filter_num[2]))
        self.con2.add_module('block2_act2', nn.ReLU())
        self.con2.add_module('block2_pool', MaxPool1d_same(kernel_size=pool_size[2], stride=pool_stride_size[2],
                                                           ))
        self.con2.add_module('block2_dropout', nn.Dropout(p=0.1))

        self.con3 = nn.Sequential()
        self.con3.add_module('block3_conv1', Conv1d_same(filter_num[2], filter_num[3], kernel_size=kernel_size[3],
                                                         stride=conv_stride_size[3]))
        self.con3.add_module('block3_bn1', nn.BatchNorm1d(filter_num[3]))
        self.con3.add_module('block3_act1', nn.ReLU())
        self.con3.add_module('block3_conv2', Conv1d_same(filter_num[3], filter_num[3], kernel_size=kernel_size[3],
                                                         stride=conv_stride_size[3]))
        self.con3.add_module('block3_bn2', nn.BatchNorm1d(filter_num[3]))
        self.con3.add_module('block3_act2', nn.ReLU())
        self.con3.add_module('block3_pool', MaxPool1d_same(kernel_size=pool_size[3], stride=pool_stride_size[3],
                                                           ))
        self.con3.add_module('block3_dropout', nn.Dropout(p=0.1))

        self.con4 = nn.Sequential()
        self.con4.add_module('block4_conv1', Conv1d_same(filter_num[3], filter_num[4], kernel_size=kernel_size[4],
                                                         stride=conv_stride_size[4]))
        self.con4.add_module('block4_bn1', nn.BatchNorm1d(filter_num[4]))
        self.con4.add_module('block4_act1', nn.ReLU())
        self.con4.add_module('block4_conv2', Conv1d_same(filter_num[4], filter_num[4], kernel_size=kernel_size[4],
                                                         stride=conv_stride_size[4]))
        self.con4.add_module('block4_bn2', nn.BatchNorm1d(filter_num[4]))
        self.con4.add_module('block4_act2', nn.ReLU())
        self.con4.add_module('block4_pool', MaxPool1d_same(kernel_size=pool_size[4], stride=pool_stride_size[4],
                                                           ))
        self.con4.add_module('block4_dropout', nn.Dropout(p=0.1))

        self.fc1 = nn.Sequential()
        self.fc1.add_module('flatten', nn.Flatten())
        #改成了CosineLinear 以前是Linear
        self.fc1.add_module('fc1', nn.Linear(filter_num[4]*20, out_features))
    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x = self.con3(x)
        x = self.con4(x)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    model = DFModel(64).cuda()
    summary.summary(model,(1,5000))