''''
bechmark with original pytorch implementation
'''

import torch
import torch.nn as nn

alpha     = 0.25
beta      = 1.0
interval  = 0.5
thresh    = 0.25 
time_step = 6

'''
fire
'''
class GoldFireFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < interval
        return grad_input * temp.float() * beta

gold_fire_func = GoldFireFunc.apply


def gold_fire(x):
    u = []
    s = []
    for t in range(time_step):
        u.append(x[t]+0)
        if t > 0:
            u[t] += (u[t - 1] * (1 - s[t - 1]) * alpha)

        s.append(gold_fire_func(u[t]))

    return torch.stack(u, dim=0), torch.stack(s, dim=0)

'''
bn
'''
def gold_bn(x, bn):
    x_shape = x.shape
    x_len = len(x_shape)
    if x_len == 5:
        x = x.view(x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4])
        x = bn(x)
        x = x.view(x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4])
    else:
        x = x.view(x_shape[0] * x_shape[1], x_shape[2])
        x = bn(x)
        x = x.view(x_shape[0], x_shape[1], x_shape[2])

    return x


'''
bn fire
'''
def gold_bn_fire(x, bn):
    x_shape = x.shape
    x_len = len(x_shape)
    if x_len == 5:
        x = x.view(x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4])
        x = bn(x)
        x = x.view(x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4])
    else:
        x = x.view(x_shape[0] * x_shape[1], x_shape[2])
        x = bn(x)
        x = x.view(x_shape[0], x_shape[1], x_shape[2])


    u = []
    s = []
    for t in range(time_step):
        u.append(x[t]+0)
        if t > 0:
            u[t] += (u[t - 1] * (1 - s[t - 1]) * alpha)

        s.append(gold_fire_func(u[t]))

    return torch.stack(u, dim=0), torch.stack(s, dim=0)



class Bn(nn.Module):
    def __init__(self, c, layer_type):
        super(Bn, self).__init__()
        if layer_type == 'fc':
            self.bn = nn.BatchNorm1d(c)
        else:
            self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        return gold_bn(x, self.bn)



class BnFire(nn.Module):
    def __init__(self, c, layer_type):
        super(BnFire, self).__init__()
        if layer_type == 'fc':
            self.bn = nn.BatchNorm1d(c)
        else:
            self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        return gold_bn_fire(x, self.bn)


class BnFireFc(nn.Module):
    def __init__(self, cin, cout, bn_fire=False, bn_layer_type=None, fire=False):
        super(BnFireFc, self).__init__()

        self.bn_fire = bn_fire
        self.fire = fire
        self.cout = cout

        if bn_fire:
            if bn_layer_type == 'fc':
                self.bn = nn.BatchNorm1d(cin)
            else:
                self.bn = nn.BatchNorm2d(cin)

        self.fc = nn.Linear(cin, cout, bias=True)

    def forward(self, x):

        if self.bn_fire:
            _, x = gold_bn_fire(x, self.bn)

        elif self.fire:
            _, x = gold_fire(x)

        if len(x.shape) == 5:
            x = x.view(x.shape[0], x.shape[1], -1)

        result = torch.empty(x.shape[0], x.shape[1], self.cout).to(x.device)

        for t in range(time_step):
            result[t] = self.fc(x[t])

        return result


class BnFireConv(nn.Module):
    def __init__(self, cin, cout, kernel_size=(3, 3), bn_fire=False, fire=False):
        super(BnFireConv, self).__init__()

        self.bn_fire = bn_fire
        self.fire = fire
        self.cout = cout

        if bn_fire:
            self.bn = nn.BatchNorm2d(cin)

        self.conv = nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=(1,1), padding=(1,1))

    def forward(self, x):

        if self.bn_fire:
            _, x = gold_bn_fire(x, self.bn)

        elif self.fire:
            _, x = gold_fire(x)

        result = []

        for t in range(time_step):
            result.append(self.conv(x[t]))

        return torch.stack(result, dim=0)

