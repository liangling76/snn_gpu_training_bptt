import torch
import torch.nn as nn

import snnlib_op as sop
import gold_op as gop
import test_utils as ut



sop_obj = sop.Conv(ut.cin, ut.cout).cuda()
gop_obj = gop.BnFireConv(ut.cin, ut.cout).cuda()
sop_obj.weight_conv, sop_obj.bias_conv = ut.conv_w, ut.conv_b
gop_obj.conv.weight, gop_obj.conv.bias = ut.gold_conv_w, ut.gold_conv_b

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


ut.check_diff(sop_final, gop_final, 'conv fp')
ut.check_diff(sop_x_grad, gop_x_grad, 'conv bp')

ut.check_diff(sop_obj.weight_conv.grad, gop_obj.conv.weight.grad, 'conv weight update')
ut.check_diff(sop_obj.bias_conv.grad, gop_obj.conv.bias.grad, 'conv bias update')


