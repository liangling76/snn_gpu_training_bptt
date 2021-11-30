'''
this file build the interface for each snn layer
'''

import torch
import torch.nn as nn
import math
import snnlib_func as sf


class Fire(nn.Module):
    def __init__(self):
        super(Fire, self).__init__()

    def forward(self, x):
        return sf.fire_func(x)


class Bn(nn.Module):
    def __init__(self, cin, dim_bn, eps_bn=1e-5):
        super(Bn, self).__init__()

        self.cin = cin
        self.dim_bn = dim_bn
        self.eps_bn = eps_bn
        self.weight_bn = nn.Parameter(torch.ones(cin))
        self.bias_bn = nn.Parameter(torch.zeros(cin))

        self.running_var = nn.Parameter(torch.ones(cin))
        self.running_mean = nn.Parameter(torch.zeros(cin))

        self.factor = None
        self.train_bn = True

    def forward(self, x):
        if self.factor is None:
            factor = x.numel()/self.cin
            self.factor = factor/(factor - 1)

            self.running_var = self.running_var.to(x.device)
            self.running_mean = self.running_mean.to(x.device)

        if self.train_bn:
            var, mean = torch.var_mean(x, dim=self.dim_bn, unbiased=False)
            self.running_var = nn.Parameter(self.running_var * 0.9 + var * (0.1 * self.factor))
            self.running_mean = nn.Parameter(self.running_mean * 0.9 + mean * 0.1)
        else:
            var, mean = self.running_var, self.running_mean

        return sf.bn_func(x, self.weight_bn, self.bias_bn, var, mean, self.dim_bn, self.eps_bn)


class BnFire(nn.Module):
    def __init__(self, cin, dim_bn, eps_bn=1e-5):
        super(BnFire, self).__init__()

        self.cin = cin
        self.dim_bn = dim_bn
        self.eps_bn = eps_bn
        self.weight_bn = nn.Parameter(torch.ones(cin))
        self.bias_bn = nn.Parameter(torch.zeros(cin))

        self.running_var = nn.Parameter(torch.ones(cin))
        self.running_mean = nn.Parameter(torch.zeros(cin))

        self.factor = None
        self.train_bn = True

    def forward(self, x):
        if self.factor is None:
            factor = x.numel()/self.cin
            self.factor = factor/(factor - 1)

            self.running_var = self.running_var.to(x.device)
            self.running_mean = self.running_mean.to(x.device)

        if self.train_bn:
            var, mean = torch.var_mean(x, dim=self.dim_bn, unbiased=False)
            self.running_var = nn.Parameter(self.running_var * 0.9 + var * (0.1 * self.factor))
            self.running_mean = nn.Parameter(self.running_mean * 0.9 + mean * 0.1)
        else:
            var, mean = self.running_var, self.running_mean

        return sf.bn_fire_func(x, self.weight_bn, self.bias_bn, var, mean, self.dim_bn, self.eps_bn)


class BnFireFc(nn.Module):
    def __init__(self, cin, cout, dim_bn, eps_bn=1e-5, cin_bn=None):
        super(BnFireFc, self).__init__()

        self.cin = cin
        self.dim_bn = dim_bn
        self.eps_bn = eps_bn

        if cin_bn is None:
            self.cin_bn = cin
        else:
            self.cin_bn =cin_bn

        self.weight_bn = nn.Parameter(torch.ones(self.cin_bn))
        self.bias_bn = nn.Parameter(torch.zeros(self.cin_bn))

        self.running_var = nn.Parameter(torch.ones(self.cin_bn))
        self.running_mean = nn.Parameter(torch.zeros(self.cin_bn))

        fc = nn.Linear(cin, cout, bias=True)
        self.weight_fc = nn.Parameter(fc.weight)
        self.bias_fc = nn.Parameter(fc.bias)

        self.factor = None
        self.train_bn = True

    def forward(self, x):
        if self.factor is None:
            factor = x.numel()/self.cin_bn
            self.factor = factor/(factor - 1)

            self.running_var = self.running_var.to(x.device)
            self.running_mean = self.running_mean.to(x.device)
        
        if self.train_bn:
            var, mean = torch.var_mean(x, dim=self.dim_bn, unbiased=False)
            self.running_var = nn.Parameter(self.running_var * 0.9 + var * (0.1 * self.factor))
            self.running_mean = nn.Parameter(self.running_mean * 0.9 + mean * 0.1)
        else:
            var, mean = self.running_var, self.running_mean
        
        return sf.bn_fire_fc_func(x, self.weight_bn, self.bias_bn, var, mean, self.dim_bn, self.eps_bn, self.weight_fc, self.bias_fc)


class FireFc(nn.Module):
    def __init__(self, cin, cout):
        super(FireFc, self).__init__()

        self.weight_fc = nn.Parameter(nn.init.normal_(torch.zeros(cout, cin), 0, 0.75/math.sqrt(cin)))
        self.bias_fc = nn.Parameter(torch.zeros(cout))

    def forward(self, x):
        return sf.fire_fc_func(x, self.weight_fc, self.bias_fc)


class Fc(nn.Module):
    def __init__(self, cin, cout):
        super(Fc, self).__init__()

        self.weight_fc = nn.Parameter(nn.init.normal_(torch.zeros(cout, cin), 0, 0.75/math.sqrt(cin)))
        self.bias_fc = nn.Parameter(torch.zeros(cout))

    def forward(self, x):
        return sf.fc_func(x, self.weight_fc, self.bias_fc)


class BnFireConv(nn.Module):
    def __init__(self, cin, cout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), eps_bn=1e-5, if_bias_conv=True):
        super(BnFireConv, self).__init__()

        self.cin = cin
        self.stride = stride
        self.padding = padding
        self.eps_bn = eps_bn
        self.if_bias_conv = if_bias_conv

        self.weight_bn = nn.Parameter(torch.ones(cin))
        self.bias_bn = nn.Parameter(torch.zeros(cin))
        self.running_var = nn.Parameter(torch.ones(cin))
        self.running_mean = nn.Parameter(torch.zeros(cin))

        conv = nn.Conv2d(cin, cout, kernel_size, stride, padding)
        self.weight_conv = nn.Parameter(conv.weight)
        self.bias_conv = nn.Parameter(conv.bias)

        self.factor = None
        self.train_bn = True

    def forward(self, x):
        if self.factor is None:
            factor = x.numel()/self.cin
            self.factor = factor/(factor - 1)

            self.running_var = self.running_var.to(x.device)
            self.running_mean = self.running_mean.to(x.device)
        
        if self.train_bn:
            var, mean = torch.var_mean(x, dim=(0, 1, 3, 4), unbiased=False)
            self.running_var = nn.Parameter(self.running_var * 0.9 + var * (0.1 * self.factor))
            self.running_mean = nn.Parameter(self.running_mean * 0.9 + mean * 0.1)
        else:
            var, mean = self.running_var, self.running_mean
        
        return sf.bn_fire_conv_func(
            x, self.weight_bn, self.bias_bn, var, mean, self.eps_bn, 
            self.weight_conv, self.bias_conv, self.stride, self.padding, self.if_bias_conv)


class FireConv(nn.Module):
    def __init__(self, cin, cout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), if_bias_conv=True):
        super(FireConv, self).__init__()

        self.cin = cin
        self.stride = stride
        self.padding = padding
        self.if_bias_conv = if_bias_conv

        conv = nn.Conv2d(cin, cout, kernel_size, stride, padding)
        self.weight_conv = nn.Parameter(conv.weight)
        self.bias_conv = nn.Parameter(conv.bias)


    def forward(self, x):
        return sf.fire_conv_func(x, self.weight_conv, self.bias_conv, self.stride, self.padding, self.if_bias_conv)


class Conv(nn.Module):
    def __init__(self, cin, cout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), if_bias_conv=True):
        super(Conv, self).__init__()

        self.cin = cin
        self.stride = stride
        self.padding = padding
        self.if_bias_conv = if_bias_conv

        self.weight_conv = nn.Parameter(nn.init.normal_(torch.zeros(cout, cin, kernel_size[0], kernel_size[1]), 0, 0.75/math.sqrt(cin)))
        self.bias_conv = nn.Parameter(torch.zeros(cout))


    def forward(self, x):
        
        return sf.conv_func(x, self.weight_conv, self.bias_conv, self.stride, self.padding, self.if_bias_conv)


class BnDist(nn.Module):
    def __init__(self, cin, dim_bn, eps_bn=1e-5):
        super(BnDist, self).__init__()

        self.cin = cin
        self.dim_bn = dim_bn
        self.eps_bn = eps_bn
        self.weight_bn = nn.Parameter(torch.ones(cin))
        self.bias_bn = nn.Parameter(torch.zeros(cin))

        self.running_var = nn.Parameter(torch.ones(cin))
        self.running_mean = nn.Parameter(torch.zeros(cin))

        self.factor = None
        self.train_bn = True

        if hasattr(torch.distributed, 'ReduceOp'):
            self.ReduceOp = torch.distributed.ReduceOp
        elif hasattr(torch.distributed, 'reduce_op'):
            self.ReduceOp = torch.distributed.reduce_op
        else:
            self.ReduceOp = torch.distributed.deprecated.reduce_op

    def forward(self, x):
        if self.factor is None:
            factor = x.numel()/self.cin
            self.factor = factor/(factor - 1)

            self.running_var = nn.Parameter(self.running_var.to(x.device))
            self.running_mean = nn.Parameter(self.running_mean.to(x.device))
            self.process_group = torch.distributed.group.WORLD
            self.world_size = torch.distributed.get_world_size(self.process_group)
        
        if self.train_bn:
            local_mean = torch.mean(x, dim=self.dim_bn)
            local_mean_sqrt = torch.mean(x**2, dim=self.dim_bn)

            torch.distributed.all_reduce(local_mean, self.ReduceOp.SUM, self.process_group)
            torch.distributed.all_reduce(local_mean_sqrt, self.ReduceOp.SUM, self.process_group)

            mean = local_mean / self.world_size
            var = local_mean_sqrt / self.world_size - mean**2

            self.running_var = nn.Parameter(self.running_var * 0.95 + var * (0.05 * self.factor))
            self.running_mean = nn.Parameter(self.running_mean * 0.95 + mean * 0.05) 
        else:
            var, mean = self.running_var, self.running_mean
        
        return sf.bn_dist_func(
            x, self.weight_bn, self.bias_bn, var, mean, self.dim_bn, self.eps_bn, 
            [self.ReduceOp, self.process_group, self.world_size])


class BnFireFcDist(nn.Module):
    def __init__(self, cin, cout, dim_bn, eps_bn=1e-5, cin_bn=None):
        super(BnFireFcDist, self).__init__()

        self.cin = cin
        self.dim_bn = dim_bn
        self.eps_bn = eps_bn

        if cin_bn is None:
            self.cin_bn = cin
        else:
            self.cin_bn =cin_bn

        self.weight_bn = nn.Parameter(torch.ones(self.cin_bn))
        self.bias_bn = nn.Parameter(torch.zeros(self.cin_bn))

        self.running_var = nn.Parameter(torch.ones(self.cin_bn))
        self.running_mean = nn.Parameter(torch.zeros(self.cin_bn))

        fc = nn.Linear(cin, cout, bias=True)
        self.weight_fc = nn.Parameter(fc.weight)
        self.bias_fc = nn.Parameter(fc.bias)

        self.factor = None
        self.train_bn = True

        if hasattr(torch.distributed, 'ReduceOp'):
            self.ReduceOp = torch.distributed.ReduceOp
        elif hasattr(torch.distributed, 'reduce_op'):
            self.ReduceOp = torch.distributed.reduce_op
        else:
            self.ReduceOp = torch.distributed.deprecated.reduce_op

    def forward(self, x):
        if self.factor is None:
            factor = x.numel()/self.cin_bn
            self.factor = factor/(factor - 1)

            self.running_var = nn.Parameter(self.running_var.to(x.device))
            self.running_mean = nn.Parameter(self.running_mean.to(x.device))
            self.process_group = torch.distributed.group.WORLD
            self.world_size = torch.distributed.get_world_size(self.process_group)
        
        if self.train_bn:
            local_mean = torch.mean(x, dim=self.dim_bn)
            local_mean_sqrt = torch.mean(x**2, dim=self.dim_bn)

            torch.distributed.all_reduce(local_mean, self.ReduceOp.SUM, self.process_group)
            torch.distributed.all_reduce(local_mean_sqrt, self.ReduceOp.SUM, self.process_group)

            mean = local_mean / self.world_size
            var = local_mean_sqrt / self.world_size - mean**2

            self.running_var = nn.Parameter(self.running_var * 0.95 + var * (0.05 * self.factor))
            self.running_mean = nn.Parameter(self.running_mean * 0.95 + mean * 0.05) 
        else:
            var, mean = self.running_var, self.running_mean
        
        return sf.bn_fire_fc_dist_func(
            x, self.weight_bn, self.bias_bn, var, mean, self.dim_bn, self.eps_bn, 
            self.weight_fc, self.bias_fc, [self.ReduceOp, self.process_group, self.world_size])


class BnFireConvDist(nn.Module):
    def __init__(self, cin, cout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), eps_bn=1e-5, if_bias_conv=True):
        super(BnFireConvDist, self).__init__()

        self.cin = cin
        self.stride = stride
        self.padding = padding
        self.eps_bn = eps_bn
        self.if_bias_conv = if_bias_conv

        self.weight_bn = nn.Parameter(torch.ones(cin))
        self.bias_bn = nn.Parameter(torch.zeros(cin))

        self.running_var = nn.Parameter(torch.ones(cin))
        self.running_mean = nn.Parameter(torch.zeros(cin))

        conv = nn.Conv2d(cin, cout, kernel_size, stride, padding)
        self.weight_conv = nn.Parameter(conv.weight)
        self.bias_conv = nn.Parameter(conv.bias)

        self.factor = None
        self.train_bn = True

        if hasattr(torch.distributed, 'ReduceOp'):
            self.ReduceOp = torch.distributed.ReduceOp
        elif hasattr(torch.distributed, 'reduce_op'):
            self.ReduceOp = torch.distributed.reduce_op
        else:
            self.ReduceOp = torch.distributed.deprecated.reduce_op

    def forward(self, x):
        if self.factor is None:
            factor = x.numel()/self.cin
            self.factor = factor/(factor - 1)

            self.running_var = nn.Parameter(self.running_var.to(x.device))
            self.running_mean = nn.Parameter(self.running_mean.to(x.device))
            self.process_group = torch.distributed.group.WORLD
            self.world_size = torch.distributed.get_world_size(self.process_group)
        
        if self.train_bn:
            local_mean = torch.mean(x, dim=(0, 1, 3, 4))
            local_mean_sqrt = torch.mean(x**2, dim=(0, 1, 3, 4))

            torch.distributed.all_reduce(local_mean, self.ReduceOp.SUM, self.process_group)
            torch.distributed.all_reduce(local_mean_sqrt, self.ReduceOp.SUM, self.process_group)

            mean = local_mean / self.world_size
            var = local_mean_sqrt / self.world_size - mean**2

            self.running_var = nn.Parameter(self.running_var * 0.95 + var * (0.05 * self.factor))
            self.running_mean = nn.Parameter(self.running_mean * 0.95 + mean * 0.05) 
        else:
            var, mean = self.running_var, self.running_mean
        
        return sf.bn_fire_conv_dist_func(
            x, self.weight_bn, self.bias_bn, var, mean, self.eps_bn, 
            self.weight_conv, self.bias_conv, self.stride, self.padding, self.if_bias_conv,
            [self.ReduceOp, self.process_group, self.world_size])

