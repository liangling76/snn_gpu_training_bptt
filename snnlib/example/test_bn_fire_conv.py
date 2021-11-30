import torch
import torch.nn as nn

import snnlib_op as sop
import gold_op as gop
import test_utils as ut



sop_obj = sop.BnFireConv(ut.cin, ut.cout).cuda()
gop_obj = gop.BnFireConv(ut.cin, ut.cout, bn_fire=True).cuda()
sop_obj.weight_conv, sop_obj.bias_conv = ut.conv_w, ut.conv_b
gop_obj.conv.weight, gop_obj.conv.bias = ut.gold_conv_w, ut.gold_conv_b

sop_obj.weight_bn, sop_obj.bias_bn, sop_obj.running_var, sop_obj.running_mean = ut.bn_w, ut.bn_b, ut.bn_v, ut.bn_m
gop_obj.bn.weight, gop_obj.bn.bias, gop_obj.bn.running_var, gop_obj.bn.running_mean = ut.gold_bn_w, ut.gold_bn_b, ut.gold_bn_v, ut.gold_bn_m


sop_result = sop_obj(ut.x_conv)
gop_result = gop_obj(ut.gold_x_conv)

sop_result = sop_result.view(sop_result.shape[0], sop_result.shape[1], -1)
gop_result = gop_result.view(gop_result.shape[0], gop_result.shape[1], -1)

sop_final = sop_result.mean(dim=0)
gop_final = gop_result.mean(dim=0)

sop_loss = ut.criterion(sop_final, ut.label)
sop_loss.retain_grad()
sop_loss.backward()

sop_x_grad = ut.x_conv.grad


gop_loss = ut.criterion(gop_final, ut.label)
gop_loss.retain_grad()
gop_loss.backward()

gop_x_grad = ut.gold_x_conv.grad


ut.check_diff(sop_final, gop_final, 'bn fire conv fp')
ut.check_diff(sop_x_grad, gop_x_grad, 'bn fire conv bp')

ut.check_diff(sop_obj.weight_bn.grad, gop_obj.bn.weight.grad, 'bn weight grad')
ut.check_diff(sop_obj.bias_bn.grad, gop_obj.bn.bias.grad, 'bn bias grad')
ut.check_diff(sop_obj.running_var, gop_obj.bn.running_var, 'bn var update')
ut.check_diff(sop_obj.running_mean, gop_obj.bn.running_mean, 'bn mean update')
ut.check_diff(sop_obj.weight_conv.grad, gop_obj.conv.weight.grad, 'conv weight update')
ut.check_diff(sop_obj.bias_conv.grad, gop_obj.conv.bias.grad, 'conv bias update')


