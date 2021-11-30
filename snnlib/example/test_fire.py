import torch
import torch.nn as nn

import snnlib_op as sop
import gold_op as gop
import test_utils as ut
import sys

input_type = sys.argv[1]


sop_obj = sop.Fire().cuda()


if input_type == 'fc':
    sop_result = sop_obj(ut.x_fc)
    _, gop_result = gop.gold_fire(ut.gold_x_fc)
elif input_type == 'conv':
    sop_result = sop_obj(ut.x_conv)
    _, gop_result = gop.gold_fire(ut.gold_x_conv)


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


ut.check_diff(sop_final, gop_final, 'fire fp')
ut.check_diff(sop_x_grad, gop_x_grad, 'fire bp')

