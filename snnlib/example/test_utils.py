import torch
import torch.nn as nn
from torch.autograd import Variable

batch = 2
time_step = 6
cin = 8 
cout = 8
fmh = 8
fmw = 8

criterion = nn.CrossEntropyLoss().cuda()
label = torch.ones(batch).type(torch.long).cuda()
label[0] = 1
label[1] = 0

# input for fc
x_fc = torch.randn(time_step, batch, cin).cuda()
gold_x_fc = x_fc.clone()
x_fc = Variable(x_fc, requires_grad=True)
gold_x_fc = Variable(gold_x_fc, requires_grad=True)

# input for conv
x_conv = torch.randn(time_step, batch, cin, fmh, fmw).cuda()
gold_x_conv = x_conv.clone()
x_conv = Variable(x_conv, requires_grad=True)
gold_x_conv = Variable(gold_x_conv, requires_grad=True)

# parameter for bn
bn_w = torch.randn(cin).cuda()
bn_b = torch.randn(cin).cuda()
bn_v = torch.randn(cin).cuda()
bn_m = torch.randn(cin).cuda()

gold_bn_w = bn_w.clone()
gold_bn_b = bn_b.clone()
gold_bn_v = bn_v.clone()
gold_bn_m = bn_m.clone()

bn_w = nn.Parameter(bn_w)
bn_b = nn.Parameter(bn_b)
bn_v = nn.Parameter(bn_v)
bn_m = nn.Parameter(bn_m)

gold_bn_w = nn.Parameter(gold_bn_w)
gold_bn_b = nn.Parameter(gold_bn_b)

# parameter for fc
fc_w_fc = torch.randn(cout, cin).cuda()
fc_w_conv = torch.randn(cout, cin * fmh * fmw).cuda()
fc_b = torch.randn(cout).cuda()

gold_fc_w_fc = fc_w_fc.clone()
gold_fc_w_conv = fc_w_conv.clone()
gold_fc_b = fc_b.clone()

fc_w_fc = nn.Parameter(fc_w_fc)
fc_w_conv = nn.Parameter(fc_w_conv)
fc_b = nn.Parameter(fc_b)

gold_fc_w_fc = nn.Parameter(gold_fc_w_fc)
gold_fc_w_conv = nn.Parameter(gold_fc_w_conv)
gold_fc_b = nn.Parameter(gold_fc_b)


# parameter for conv
conv_w = torch.randn(cout, cin, 3, 3).cuda()
conv_b = torch.randn(cout).cuda()

gold_conv_w = conv_w.clone()
gold_conv_b = conv_b.clone()

conv_w = nn.Parameter(conv_w)
conv_b = nn.Parameter(conv_b)

gold_conv_w = nn.Parameter(gold_conv_w)
gold_conv_b = nn.Parameter(gold_conv_b)


def check_diff(org, ref, name):
    diff = abs(org - ref)
    print(name, ':')
    print('# of diff : ', torch.nonzero(diff).size(0))
    print('non 0 ref : ', torch.nonzero(ref).size(0))
    print('total size: ', diff.numel())

    # print('org min   : ', abs(org[org!=0]).min().item()) 
    # org_nonzero_idx = org.nonzero()
    # org_min_idx = torch.argmin(abs(org[org!=0]))
    # ref_idx = tuple(org_nonzero_idx[org_min_idx].tolist())
    # print('ref min   : ', abs(ref[ref_idx]).item())

    print('max diff  : ', diff.max().item())
    max_diff_idx = diff.argmax()
    print('max diff org: ', org.view(-1)[max_diff_idx].item())
    print('max diff ref: ', ref.view(-1)[max_diff_idx].item())
    print('\n')