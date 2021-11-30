#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <vector>


#define BLOCKS(N, T) (N + T - 1) / T
#define THREAD_NUM 128

/*
fire forward s kernel
*/

template <typename scalar_t>
__global__ void fire_s_fp_kernel(
    const scalar_t* __restrict__ spatial,   
    scalar_t* __restrict__ spike, 

    const float alpha,
    const float thresh,
    const int T,     
    const int B,
    const int CHW,   
    const int BCHW
    
){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    scalar_t u_t_pre;
    int s_t_pre;
    int t_idx;

    
    for(int i = idx; i < CHW; i += stride){

        for(int b = 0; b < B; b++){

            u_t_pre = 0;
            s_t_pre = 0;
            
            for(int t = 0; t < T; t++){
                t_idx   = t * BCHW + b * CHW + i;
                u_t_pre = alpha * u_t_pre * (1 - s_t_pre) + spatial[t_idx];
                s_t_pre = u_t_pre > thresh;

                spike[t_idx] = s_t_pre;
            }
        }
    }
}

torch::Tensor fire_s_fp_cuda(
    torch::Tensor spatial,
    float alpha,
    float thresh
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;

    auto spike = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "fire_s_fp_kernel",([&]{
        fire_s_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), spike.data<scalar_t>(), 
            alpha, thresh, T, B, CHW, BCHW);}));

    return spike;
}


/*
bn forward kernel
*/

template <typename scalar_t>
__global__ void bn_fp_kernel(
    const scalar_t* __restrict__ x_bn, 
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,

    scalar_t* __restrict__ y_bn,  
    scalar_t* __restrict__ ivar,
     
    const int T,     
    const int B,
    const int HW,
    const int CHW,   
    const int BCHW,
    const float eps_bn
){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int t_idx, c_idx;
    float tmp_m, tmp_w, tmp_b;
    
    for(int i = idx; i < CHW; i += stride){

        c_idx   = int(i / HW);
        tmp_m   = mean[c_idx];
        tmp_w   = weight[c_idx] / std::sqrt(var[c_idx] + eps_bn);
        tmp_b   = bias[c_idx];

        ivar[c_idx] = tmp_w / weight[c_idx];

        for(int b = 0; b < B; b++){

            for(int t = 0; t < T; t++){
                t_idx   = t * BCHW + b * CHW + i;
                y_bn[t_idx] = (x_bn[t_idx] - tmp_m) * tmp_w + tmp_b;
            }
        }
    }
}

std::vector<torch::Tensor> bn_fp_cuda(
    torch::Tensor x_bn,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor var,
    torch::Tensor mean,
    float eps_bn
){
    int T = x_bn.size(0);
    int B = x_bn.size(1);
    int C = x_bn.size(2);
    int BCHW = x_bn.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    auto y_bn  = torch::empty_like(x_bn, x_bn.device());
    auto ivar   = torch::empty_like(var, var.device());

    AT_DISPATCH_ALL_TYPES(x_bn.type(), "bn_fp_kernel",([&]{
        bn_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            x_bn.data<scalar_t>(), 
            var.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), bias_bn.data<scalar_t>(),
            y_bn.data<scalar_t>(), ivar.data<scalar_t>(),
            T, B, HW, CHW, BCHW, eps_bn);}));

    return {ivar, y_bn};
}


/*
bn fire forward s kernel
*/

template <typename scalar_t>
__global__ void bn_fire_s_fp_kernel(
    const scalar_t* __restrict__ spatial, 
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,

    scalar_t* __restrict__ spike,  
    scalar_t* __restrict__ ivar,
     
    const float alpha,
    const float thresh,
    const int T,     
    const int B,
    const int HW,
    const int CHW,   
    const int BCHW,
    const float eps_bn
){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int t_idx, c_idx;
    scalar_t u_t_pre;
    int s_t_pre;
    float tmp_m, tmp_w, tmp_b;
    
    for(int i = idx; i < CHW; i += stride){

        c_idx   = int(i / HW);
        tmp_m   = mean[c_idx];
        tmp_w   = weight[c_idx] / std::sqrt(var[c_idx] + eps_bn);
        tmp_b   = bias[c_idx];

        ivar[c_idx] = tmp_w / weight[c_idx];

        for(int b = 0; b < B; b++){

            u_t_pre = 0;
            s_t_pre = 0;

            for(int t = 0; t < T; t++){
                t_idx   = t * BCHW + b * CHW + i;
                u_t_pre = alpha * u_t_pre * (1 - s_t_pre) + (spatial[t_idx] - tmp_m) * tmp_w + tmp_b;
                s_t_pre = u_t_pre > thresh;

                spike[t_idx] = s_t_pre;
            }

        }
    }
}

std::vector<torch::Tensor> bn_fire_s_fp_cuda(
    torch::Tensor spatial,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor var,
    torch::Tensor mean,
    float eps_bn,
    float alpha,
    float thresh
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    auto spike  = torch::empty_like(spatial, spatial.device());
    auto ivar   = torch::empty_like(var, var.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_s_fp_kernel",([&]{
        bn_fire_s_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), 
            var.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), bias_bn.data<scalar_t>(),
            spike.data<scalar_t>(), ivar.data<scalar_t>(),
            alpha, thresh, T, B, HW, CHW, BCHW, eps_bn);}));

    return {ivar, spike};
}


// FC layers
std::vector<torch::Tensor> bn_fire_fc_fp_cuda(
    torch::Tensor spatial,

    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor var,
    torch::Tensor mean,
    float eps_bn,

    torch::Tensor weight_fc,
    torch::Tensor bias_fc,

    float alpha,
    float thresh
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    auto spike  = torch::empty_like(spatial, spatial.device());
    auto ivar   = torch::empty_like(var, var.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_s_fp_kernel",([&]{
        bn_fire_s_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), 
            var.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), bias_bn.data<scalar_t>(),
            spike.data<scalar_t>(), ivar.data<scalar_t>(),
            alpha, thresh, T, B, HW, CHW, BCHW, eps_bn);}));

    auto fc = torch::empty({T, spatial.size(1), weight_fc.size(0)}, spatial.device());

    if(HW==1){
        for(int t = 0; t < T; t++){
            fc.data()[t] = at::linear(spike.data()[t], weight_fc, bias_fc);
        }
    }
    else{
        spike = spike.view({T, spatial.size(1), -1});
        for(int t = 0; t < T; t++){
            fc.data()[t] = at::linear(spike.data()[t], weight_fc, bias_fc);
        }
    }

    return {ivar, fc};   
}



torch::Tensor fire_fc_fp_cuda(
    torch::Tensor spatial,
    torch::Tensor weight_fc,
    torch::Tensor bias_fc,
    float alpha,
    float thresh
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    auto spike = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "fire_s_fp_kernel",([&]{
        fire_s_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(spatial.data<scalar_t>(), spike.data<scalar_t>(), alpha, thresh, T, B, CHW, BCHW);}));

    auto fc = torch::empty({T, spatial.size(1), weight_fc.size(0)}, spatial.device());

    if(HW==1){
        for(int t = 0; t < T; t++){
            fc.data()[t] = at::linear(spike.data()[t], weight_fc, bias_fc);
        }
    }
    else{
        spike = spike.view({T, spatial.size(1), -1});
        for(int t = 0; t < T; t++){
            fc.data()[t] = at::linear(spike.data()[t], weight_fc, bias_fc);
        }
    }

    return fc;   
}


torch::Tensor fc_fp_cuda(
    torch::Tensor spatial,
    torch::Tensor weight_fc,
    torch::Tensor bias_fc
){
    int T = spatial.size(0);
    int C = spatial.size(2);
    int N = spatial.numel() / T;
    int HW = N / (spatial.size(1) * C);

    auto fc = torch::empty({T, spatial.size(1), weight_fc.size(0)}, spatial.device());

    if(HW==1){
        for(int t = 0; t < T; t++){
            fc.data()[t] = at::linear(spatial.data()[t], weight_fc, bias_fc);
        }
    }
    else{
        spatial = spatial.view({T, spatial.size(1), -1});
        for(int t = 0; t < T; t++){
            fc.data()[t] = at::linear(spatial.data()[t], weight_fc, bias_fc);
        }
    }

    return fc;   
}



std::vector<torch::Tensor> bn_fire_conv_fp_cuda(
    torch::Tensor spatial,

    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor var,
    torch::Tensor mean,
    float eps_bn,

    torch::Tensor weight_conv,
    torch::Tensor bias_conv,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,

    float alpha,
    float thresh
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    auto spike  = torch::empty_like(spatial, spatial.device());
    auto ivar   = torch::empty_like(var, var.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_s_fp_kernel",([&]{
        bn_fire_s_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), 
            var.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), bias_bn.data<scalar_t>(),
            spike.data<scalar_t>(), ivar.data<scalar_t>(),
            alpha, thresh, T, B, HW, CHW, BCHW, eps_bn);}));

    int Hout = int((spatial.size(3) + 2 * padding[0] - dilation[0] * (weight_conv.size(2) - 1) - 1) / stride[0]) + 1;
    int Wout = int((spatial.size(4) + 2 * padding[1] - dilation[1] * (weight_conv.size(3) - 1) - 1) / stride[1]) + 1;
    
    auto conv = torch::empty({T, spatial.size(1), weight_conv.size(0), Hout, Wout}, spatial.device());

    for(int t = 0; t < T; t++){
        conv.data()[t] = at::cudnn_convolution(
            spike.data()[t], weight_conv, bias_conv, padding, stride, dilation, groups, benchmark, deterministic);
    }

    return {ivar, conv};   
}


torch::Tensor fire_conv_fp_cuda(
    torch::Tensor spatial,

    torch::Tensor weight_conv,
    torch::Tensor bias_conv,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,

    float alpha,
    float thresh
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;

    auto spike  = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "fire_s_fp_kernel",([&]{
        fire_s_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(spatial.data<scalar_t>(), spike.data<scalar_t>(), alpha, thresh, T, B, CHW, BCHW);}));


    int Hout = int((spatial.size(3) + 2 * padding[0] - dilation[0] * (weight_conv.size(2) - 1) - 1) / stride[0]) + 1;
    int Wout = int((spatial.size(4) + 2 * padding[1] - dilation[1] * (weight_conv.size(3) - 1) - 1) / stride[1]) + 1;
    
    auto conv = torch::empty({T, spatial.size(1), weight_conv.size(0), Hout, Wout}, spatial.device());

    for(int t = 0; t < T; t++){
        conv.data()[t] = at::cudnn_convolution(
            spike.data()[t], weight_conv, bias_conv, padding, stride, dilation, groups, benchmark, deterministic);
    }

    return conv;   
}



torch::Tensor conv_fp_cuda(
    torch::Tensor spatial,

    torch::Tensor weight_conv,
    torch::Tensor bias_conv,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic
){
    int T = spatial.size(0);

    int Hout = int((spatial.size(3) + 2 * padding[0] - dilation[0] * (weight_conv.size(2) - 1) - 1) / stride[0]) + 1;
    int Wout = int((spatial.size(4) + 2 * padding[1] - dilation[1] * (weight_conv.size(3) - 1) - 1) / stride[1]) + 1;
    
    auto conv = torch::empty({T, spatial.size(1), weight_conv.size(0), Hout, Wout}, spatial.device());

    for(int t = 0; t < T; t++){
        conv.data()[t] = at::cudnn_convolution(
            spatial.data()[t], weight_conv, bias_conv, padding, stride, dilation, groups, benchmark, deterministic);
    }

    return conv;   
}

