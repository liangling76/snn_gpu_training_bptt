import torch
import torch.nn as nn

import snnlib_op as sop
import gold_op as gop
import test_utils as ut
import sys

input_type = sys.argv[1]


if input_type == 'fc':
    sop_obj = sop.FireFc(ut.cin, ut.cout).cuda()
    gop_obj = gop.BnFireFc(ut.cin, ut.cout, fire=True).cuda()
    sop_obj.weight_fc, sop_obj.bias_fc = ut.fc_w_fc, ut.fc_b

    gop_obj.fc.weight, gop_obj.fc.bias = ut.gold_fc_w_fc, ut.gold_fc_b
    sop_result = sop_obj(ut.x_fc)
    gop_result = gop_obj(ut.gold_x_fc)


elif input_type == 'conv':
    sop_obj = sop.FireFc(ut.cin, ut.cout).cuda()
    gop_obj = gop.BnFireFc(ut.cin, ut.cout, fire=True).cuda()
    sop_obj.weight_fc, sop_obj.bias_fc = ut.fc_w_conv, ut.fc_b
    gop_obj.fc.weight, gop_obj.fc.bias = ut.gold_fc_w_conv, ut.gold_fc_b

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



ut.check_diff(sop_final, gop_final, 'fire fc fp')
ut.check_diff(sop_x_grad, gop_x_grad, 'fire fc bp')

ut.check_diff(sop_obj.weight_fc.grad, gop_obj.fc.weight.grad, 'fc weight update')
ut.check_diff(sop_obj.bias_fc.grad, gop_obj.fc.bias.grad, 'fc bias update')


