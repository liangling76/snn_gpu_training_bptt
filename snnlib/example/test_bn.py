import torch
import torch.nn as nn

import snnlib_op as sop
import gold_op as gop
import test_utils as ut
import sys

input_type = sys.argv[1]


if input_type == 'fc':
    sop_obj = sop.Bn(ut.cin, (0,1)).cuda()
    gop_obj = gop.Bn(ut.cin, 'fc').cuda()
elif input_type == 'conv':
    sop_obj = sop.Bn(ut.cin, (0,1,3,4)).cuda()
    gop_obj = gop.Bn(ut.cin, 'conv').cuda()

sop_obj.weight_bn, sop_obj.bias_bn, sop_obj.running_var, sop_obj.running_mean = ut.bn_w, ut.bn_b, ut.bn_v, ut.bn_m
gop_obj.bn.weight, gop_obj.bn.bias, gop_obj.bn.running_var, gop_obj.bn.running_mean = ut.gold_bn_w, ut.gold_bn_b, ut.gold_bn_v, ut.gold_bn_m

if input_type == 'fc':
    sop_result = sop_obj(ut.x_fc)
    gop_result = gop_obj(ut.gold_x_fc)
elif input_type == 'conv':
    sop_result = sop_obj(ut.x_conv)
    gop_result = gop_obj(ut.gold_x_conv)


sop_result = sop_result.view(sop_result.shape[0], sop_result.shape[1], -1)
gop_result = gop_result.view(gop_result.shape[0], gop_result.shape[1], -1)

sop_final = sop_result.mean(dim=0)
gop_final = gop_result.mean(dim=0)

sop_loss = ut.criterion(sop_final, ut.label)
sop_loss.retain_grad()
sop_loss.backward()

if input_type == 'fc':
    sop_x_grad = ut.x_fc.grad
elif input_type == 'conv':
    sop_x_grad = ut.x_conv.grad


gop_loss = ut.criterion(gop_final, ut.label)
gop_loss.retain_grad()
gop_loss.backward()

if input_type == 'fc':
    gop_x_grad = ut.gold_x_fc.grad
elif input_type == 'conv':
    gop_x_grad = ut.gold_x_conv.grad



ut.check_diff(sop_final, gop_final, 'bn fp')
ut.check_diff(sop_x_grad, gop_x_grad, 'bn bp')

ut.check_diff(sop_obj.weight_bn.grad, gop_obj.bn.weight.grad, 'bn weight grad')
ut.check_diff(sop_obj.bias_bn.grad, gop_obj.bn.bias.grad, 'bn bias grad')
ut.check_diff(sop_obj.running_var, gop_obj.bn.running_var, 'bn var update')
ut.check_diff(sop_obj.running_mean, gop_obj.bn.running_mean, 'bn mean update')

