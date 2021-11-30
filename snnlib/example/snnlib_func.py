'''
this file build the forward and backward for each single function
also the hyper parameters are set in this file
'''

import torch
import torch.nn as nn
import snnlib

# hyper parameter setting
alpha     = 0.25
beta      = 1.0
interval  = 0.5
thresh    = 0.25 
time_step = 6

conv_benchmark = False # if true the conv becomes faster but consume more memory

'''
fire
'''
class FireFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spatial):
        ctx.save_for_backward(spatial)
        spike = snnlib.fire_s_fp(spatial, alpha, thresh)
        return spike

    @staticmethod
    def backward(ctx, grad_spike):
        spatial, = ctx.saved_tensors
        grad_spatial = snnlib.fire_bp(spatial.contiguous(), grad_spike.contiguous(), alpha, thresh, beta, interval)
        return grad_spatial    

fire_func = FireFunc.apply


'''
bn 
'''
class BnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_bn, weight_bn, bias_bn, var, mean, dim_bn, eps_bn):
        ivar, y_bn = snnlib.bn_fp(x_bn, weight_bn, bias_bn, var, mean, eps_bn)
        ctx.save_for_backward(x_bn, weight_bn, bias_bn, ivar, mean)
        ctx.dim_bn = dim_bn
        return y_bn

    @staticmethod
    def backward(ctx, grad_y_bn):
        x_bn, weight_bn, bias_bn, ivar, mean, = ctx.saved_tensors
        dim_bn = ctx.dim_bn
        grad_x_bn, grad_weight_bn, grad_bias_bn = snnlib.bn_bp(
            x_bn, weight_bn, bias_bn, ivar, mean, dim_bn, grad_y_bn)

        return grad_x_bn, grad_weight_bn, grad_bias_bn, None, None, None, None       

bn_func = BnFunc.apply



'''
bn fire
'''
class BnFireFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spatial, weight_bn, bias_bn, var, mean, dim_bn, eps_bn):
        ivar, spike = snnlib.bn_fire_s_fp(spatial, weight_bn, bias_bn, var, mean, eps_bn, alpha, thresh)
        ctx.save_for_backward(spatial, weight_bn, bias_bn, ivar, mean)
        ctx.dim_bn = dim_bn
        return spike

    @staticmethod
    def backward(ctx, grad_spike):
        spatial, weight_bn, bias_bn, ivar, mean, = ctx.saved_tensors
        dim_bn = ctx.dim_bn
        grad_spatial, grad_weight_bn, grad_bias_bn = snnlib.bn_fire_bp(
            spatial, weight_bn, bias_bn, ivar, mean, dim_bn, grad_spike, alpha, thresh, beta, interval)

        return grad_spatial, grad_weight_bn, grad_bias_bn, None, None, None, None       

bn_fire_func = BnFireFunc.apply


'''
bn fire fc
'''
class BnFireFcFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spatial, weight_bn, bias_bn, var, mean, dim_bn, eps_bn, weight_fc, bias_fc):
        ivar, fc = snnlib.bn_fire_fc_fp(spatial, weight_bn, bias_bn, var, mean, eps_bn, weight_fc, bias_fc, alpha, thresh)
        ctx.save_for_backward(spatial, weight_bn, bias_bn, ivar, mean, weight_fc, bias_fc)
        ctx.dim_bn = dim_bn
        return fc

    @staticmethod
    def backward(ctx, grad_fc):
        spatial, weight_bn, bias_bn, ivar, mean, weight_fc, bias_fc, = ctx.saved_tensors
        dim_bn = ctx.dim_bn
        grad_spatial, grad_weight_bn, grad_bias_bn, grad_weight_fc, grad_bias_fc = snnlib.bn_fire_fc_bp(
            spatial, weight_bn, bias_bn, ivar, mean, dim_bn, 
            grad_fc, weight_fc, bias_fc, (0,1), alpha, thresh, beta, interval)

        return grad_spatial, grad_weight_bn, grad_bias_bn, None, None, None, None, grad_weight_fc, grad_bias_fc       

bn_fire_fc_func = BnFireFcFunc.apply


'''
fire fc
'''
class FireFcFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spatial, weight_fc, bias_fc):
        fc = snnlib.fire_fc_fp(spatial, weight_fc, bias_fc, alpha, thresh)
        ctx.save_for_backward(spatial, weight_fc, bias_fc)
        return fc

    @staticmethod
    def backward(ctx, grad_fc):
        spatial, weight_fc, bias_fc, = ctx.saved_tensors
        grad_spatial, grad_weight_fc, grad_bias_fc = snnlib.fire_fc_bp(
            spatial, grad_fc, weight_fc, bias_fc, (0,1), alpha, thresh, beta, interval)

        return grad_spatial, grad_weight_fc, grad_bias_fc       

fire_fc_func = FireFcFunc.apply


'''
fc
'''
class FcFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spatial, weight_fc, bias_fc):
        fc = snnlib.fc_fp(spatial, weight_fc, bias_fc)
        ctx.save_for_backward(spatial, weight_fc, bias_fc)
        return fc

    @staticmethod
    def backward(ctx, grad_fc):
        spatial, weight_fc, bias_fc, = ctx.saved_tensors
        grad_spatial, grad_weight_fc, grad_bias_fc = snnlib.fc_bp(
            spatial, grad_fc, weight_fc, bias_fc, (0,1))

        return grad_spatial, grad_weight_fc, grad_bias_fc       


fc_func = FcFunc.apply


'''
bn fire conv
'''
class BnFireConvFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, spatial, weight_bn, bias_bn, var, mean, eps_bn, 
        weight_conv, bias_conv, stride, padding, if_bias_conv):

        ivar, conv = snnlib.bn_fire_conv_fp(
            spatial, weight_bn, bias_bn, var, mean, eps_bn, 
            weight_conv, bias_conv, stride, padding, (1, 1), 1, conv_benchmark, False, alpha, thresh)

        ctx.save_for_backward(spatial, weight_bn, bias_bn, ivar, mean, weight_conv, bias_conv)
        ctx.stride, ctx.padding, ctx.if_bias_conv = stride, padding, if_bias_conv
        return conv

    @staticmethod
    def backward(ctx, grad_conv):
        spatial, weight_bn, bias_bn, ivar, mean, weight_conv, bias_conv = ctx.saved_tensors
        stride, padding, if_bias_conv = ctx.stride, ctx.padding, ctx.if_bias_conv

        grad_spatial, grad_weight_bn, grad_bias_bn, grad_weight_conv, grad_bias_conv = snnlib.bn_fire_conv_bp(
            spatial.contiguous(), weight_bn, bias_bn, ivar, mean, (0, 1, 3, 4), 
            grad_conv.contiguous(), weight_conv, stride, padding, (1, 1), 1, conv_benchmark, False, True, (0, 1, 3, 4), alpha, thresh, beta, interval)

        if if_bias_conv:
            return grad_spatial, grad_weight_bn, grad_bias_bn, None, None, None, grad_weight_conv, grad_bias_conv, None, None, None     
        else:
            return grad_spatial, grad_weight_bn, grad_bias_bn, None, None, None, grad_weight_conv, None, None, None, None


bn_fire_conv_func = BnFireConvFunc.apply


'''
fire conv
'''
class FireConvFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spatial, weight_conv, bias_conv, stride, padding, if_bias_conv):

        conv = snnlib.fire_conv_fp(
            spatial, weight_conv, bias_conv, stride, padding, (1, 1), 1, conv_benchmark, False, alpha, thresh)

        ctx.save_for_backward(spatial, weight_conv, bias_conv)
        ctx.stride, ctx.padding, ctx.if_bias_conv = stride, padding, if_bias_conv
        return conv

    @staticmethod
    def backward(ctx, grad_conv):
        spatial, weight_conv, bias_conv, = ctx.saved_tensors
        stride, padding, if_bias_conv = ctx.stride, ctx.padding, ctx.if_bias_conv

        grad_spatial, grad_weight_conv, grad_bias_conv = snnlib.fire_conv_bp(
            spatial.contiguous(), grad_conv.contiguous(), weight_conv, stride, padding, (1, 1), 1, conv_benchmark, False, True, (0, 1, 3, 4), alpha, thresh, beta, interval)

        if if_bias_conv:
            return grad_spatial, grad_weight_conv, grad_bias_conv, None, None, None 
        else:
            return grad_spatial, grad_weight_conv, None, None, None, None     


fire_conv_func = FireConvFunc.apply


'''
conv
'''
class ConvFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spatial, weight_conv, bias_conv, stride, padding, if_bias_conv):

        conv = snnlib.conv_fp(
            spatial, weight_conv, bias_conv, stride, padding, (1, 1), 1, conv_benchmark, False)

        ctx.save_for_backward(spatial, weight_conv, bias_conv)
        ctx.stride, ctx.padding, ctx.if_bias_conv = stride, padding, if_bias_conv
        return conv

    @staticmethod
    def backward(ctx, grad_conv):
        spatial, weight_conv, bias_conv, = ctx.saved_tensors
        stride, padding, if_bias_conv = ctx.stride, ctx.padding, ctx.if_bias_conv

        grad_spatial, grad_weight_conv, grad_bias_conv = snnlib.conv_bp(
            spatial.contiguous(), grad_conv.contiguous(), weight_conv, stride, padding, (1, 1), 1, conv_benchmark, False, True, (0, 1, 3, 4))

        if if_bias_conv:
            return grad_spatial, grad_weight_conv, grad_bias_conv, None, None, None       
        else:
            return grad_spatial, grad_weight_conv, None, None, None, None


conv_func = ConvFunc.apply


'''
bn dist
'''
class BnDistFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_bn, weight_bn, bias_bn, var, mean, dim_bn, eps_bn, reduce_op):
        ivar, y_bn = snnlib.bn_fp(x_bn, weight_bn, bias_bn, var, mean, eps_bn)
        ctx.save_for_backward(x_bn, weight_bn, bias_bn, ivar, mean)
        ctx.dim_bn, ctx.reduce_op = dim_bn, reduce_op
        return y_bn

    @staticmethod
    def backward(ctx, grad_y_bn):
        x_bn, weight_bn, bias_bn, ivar, mean = ctx.saved_tensors
        dim_bn, reduce_op = ctx.dim_bn, ctx.reduce_op

        local_grad_xhat, local_grad_xhatx, grad_weight_bn, grad_bias_bn = snnlib.bn_bp_dist1(
            x_bn.contiguous(), weight_bn, bias_bn, ivar, mean, dim_bn, grad_y_bn.contiguous())

        torch.distributed.all_reduce(local_grad_xhat,  reduce_op[0].SUM, reduce_op[1])
        torch.distributed.all_reduce(local_grad_xhatx, reduce_op[0].SUM, reduce_op[1])

        grad_xhat  = local_grad_xhat  / reduce_op[2]
        grad_xhatx = local_grad_xhatx / reduce_op[2]

        grad_x_bn = snnlib.bn_bp_dist2(
            x_bn.contiguous(), weight_bn, bias_bn, ivar, mean, dim_bn, grad_y_bn.contiguous(), 
            grad_xhat, grad_xhatx)
        
        return grad_x_bn, grad_weight_bn, grad_bias_bn, None, None, None, None, None       

bn_dist_func = BnDistFunc.apply



'''
bn fire fc dist
'''
class BnFireFcDistFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spatial, weight_bn, bias_bn, var, mean, dim_bn, eps_bn, weight_fc, bias_fc, reduce_op):
        ivar, fc = snnlib.bn_fire_fc_fp(spatial, weight_bn, bias_bn, var, mean, eps_bn, weight_fc, bias_fc, alpha, thresh)
        ctx.save_for_backward(spatial, weight_bn, bias_bn, ivar, mean, weight_fc, bias_fc)
        ctx.dim_bn, ctx.reduce_op = dim_bn, reduce_op
        return fc

    @staticmethod
    def backward(ctx, grad_fc):
        spatial, weight_bn, bias_bn, ivar, mean, weight_fc, bias_fc = ctx.saved_tensors
        dim_bn, reduce_op = ctx.dim_bn, ctx.reduce_op

        grad_spike, local_grad_xhat, local_grad_xhatx, grad_weight_bn, grad_bias_bn, grad_weight_fc, grad_bias_fc = snnlib.bn_fire_fc_bp_dist1(
            spatial.contiguous(), weight_bn, bias_bn, ivar, mean, dim_bn, 
            grad_fc.contiguous(), weight_fc, bias_fc, (0, 1), alpha, thresh, beta, interval)

        torch.distributed.all_reduce(local_grad_xhat,  reduce_op[0].SUM, reduce_op[1])
        torch.distributed.all_reduce(local_grad_xhatx, reduce_op[0].SUM, reduce_op[1])

        grad_xhat  = local_grad_xhat  / reduce_op[2]
        grad_xhatx = local_grad_xhatx / reduce_op[2]

        grad_spatial = snnlib.bn_fire_bp_dist2(
            spatial.contiguous(), weight_bn, bias_bn, ivar, mean,
            grad_spike.contiguous(), grad_xhat, grad_xhatx,
            alpha, thresh, beta, interval
        )
        
        return grad_spatial, grad_weight_bn, grad_bias_bn, None, None, None, None, grad_weight_fc, grad_bias_fc, None       

bn_fire_fc_dist_func = BnFireFcDistFunc.apply



'''
bn fire conv dist
'''
class BnFireConvDistFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, spatial, weight_bn, bias_bn, var, mean, eps_bn, 
        weight_conv, bias_conv, stride, padding, if_bias_conv, reduce_op):

        ivar, conv = snnlib.bn_fire_conv_fp(
            spatial, weight_bn, bias_bn, var, mean, eps_bn, 
            weight_conv, bias_conv, stride, padding, (1, 1), 1, conv_benchmark, False, alpha, thresh)

        ctx.save_for_backward(spatial, weight_bn, bias_bn, ivar, mean, weight_conv, bias_conv)
        ctx.stride, ctx.padding, ctx.if_bias_conv, ctx.reduce_op = stride, padding, if_bias_conv, reduce_op
        return conv

    @staticmethod
    def backward(ctx, grad_conv):
        spatial, weight_bn, bias_bn, ivar, mean, weight_conv, bias_conv = ctx.saved_tensors
        stride, padding, if_bias_conv, reduce_op = ctx.stride, ctx.padding, ctx.if_bias_conv, ctx.reduce_op

        grad_spike, local_grad_xhat, local_grad_xhatx, grad_weight_bn, grad_bias_bn, grad_weight_conv, grad_bias_conv = snnlib.bn_fire_conv_bp_dist1(
            spatial.contiguous(), weight_bn, bias_bn, ivar, mean, (0, 1, 3, 4), 
            grad_conv.contiguous(), weight_conv, stride, padding, (1, 1), 1, conv_benchmark, False, True,
            (0, 1, 3, 4), alpha, thresh, beta, interval
        )

        torch.distributed.all_reduce(local_grad_xhat,  reduce_op[0].SUM, reduce_op[1])
        torch.distributed.all_reduce(local_grad_xhatx, reduce_op[0].SUM, reduce_op[1])

        grad_xhat  = local_grad_xhat  / reduce_op[2]
        grad_xhatx = local_grad_xhatx / reduce_op[2]

        grad_spatial = snnlib.bn_fire_bp_dist2(
            spatial.contiguous(), weight_bn, bias_bn, ivar, mean,
            grad_spike.contiguous(), grad_xhat, grad_xhatx,
            alpha, thresh, beta, interval
        )
        
        if if_bias_conv:
            return grad_spatial, grad_weight_bn, grad_bias_bn, None, None, None, grad_weight_conv, grad_bias_conv, None, None, None, None  
        else:
            return grad_spatial, grad_weight_bn, grad_bias_bn, None, None, None, grad_weight_conv, None, None, None, None, None

bn_fire_conv_dist_func = BnFireConvDistFunc.apply


