#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <vector>


#define BLOCKS(N, T) (N + T - 1) / T
#define THREAD_NUM 128

/*
fire forward u and backward kernel
*/

template <typename scalar_t>
__global__ void fire_u_fp_kernel(
    const scalar_t* __restrict__ spatial,   
    scalar_t* __restrict__ mem, 
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

                mem[t_idx] = u_t_pre;
            }

        }   

    }
}

template <typename scalar_t>
__global__ void fire_us_fp_kernel(
    const scalar_t* __restrict__ spatial,   
    scalar_t* __restrict__ mem, 
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

                mem[t_idx] = u_t_pre;
                spike[t_idx] = s_t_pre;
            }

        }   

    }
}

template <typename scalar_t>
__global__ void fire_bp_kernel(
    const scalar_t* __restrict__ grad_spike,
    const scalar_t* __restrict__ mem,
    scalar_t* __restrict__ grad_mem,

    const float alpha,
    const float thresh,
    const float beta,
    const float interval,
    const int T,     
    const int B,
    const int CHW,   
    const int BCHW

){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    scalar_t grad_u_last, grad_s_last, u_tmp;
    int      t_idx;

    for(int i = idx; i < CHW; i += stride){
    // int i = idx;
        for(int b = 0; b < B; b++){
            
            grad_u_last = 0;

            for(int t = T - 1; t >=0; t--){
                t_idx = t * BCHW + b * CHW + i;
                u_tmp = mem[t_idx];

                // gradient of spike
                grad_s_last = 0;
                if((u_tmp - thresh) < interval && (thresh - u_tmp) < interval){
                    grad_s_last = grad_spike[t_idx] - grad_u_last * alpha * u_tmp;
                }

                // gradient of mem
                grad_u_last = grad_s_last * beta + grad_u_last * alpha * (u_tmp <= thresh);
                grad_mem[t_idx] = grad_u_last;
            }
        }
    }
}

torch::Tensor fire_bp_cuda(
    torch::Tensor spatial,
    torch::Tensor grad_spike,
    float alpha,
    float thresh,
    float beta,
    float interval
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;

    auto mem = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "fire_u_fp_kernel",([&]{
        fire_u_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(spatial.data<scalar_t>(), mem.data<scalar_t>(), 
        alpha, thresh,T, B, CHW, BCHW);}));

    auto grad_spatial = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "fire_bp_kernel",([&]{
        fire_bp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), mem.data<scalar_t>(), grad_spatial.data<scalar_t>(), 
            alpha, thresh, beta, interval, T, B, CHW, BCHW);}));
        
    return grad_spatial;
}


/*
bn backward 
*/

template <typename scalar_t>
__global__ void bn_bp_wb_kernel(
    const scalar_t* __restrict__ grad_y_bn,
    const scalar_t* __restrict__ x_bn,
    const scalar_t* __restrict__ ivar,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ weight,

    scalar_t* __restrict__ grad_xhat,
    scalar_t* __restrict__ grad_xhatx,
    scalar_t* __restrict__ grad_weight,
    scalar_t* __restrict__ grad_bias,

    const int T,     
    const int B,
    const int HW,
    const int CHW,   
    const int BCHW
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int t_idx, c_idx;
    float tmp_m, tmp_w, tmp_ivar;
    float x_bn_tmp, grad_y_bn_tmp, grad_bias_tmp, grad_w_tmp, grad_xhat_tmp, grad_xhatx_tmp;

    for(int i = idx; i < CHW; i += stride){

        c_idx    = int(i / HW);
        tmp_m    = mean[c_idx];
        tmp_w    = weight[c_idx];
        tmp_ivar = ivar[c_idx];

        grad_bias_tmp = 0;
        grad_w_tmp = 0;
        grad_xhat_tmp = 0;
        grad_xhatx_tmp = 0;

        for(int b = 0; b < B; b++){

            for(int t = T - 1; t >=0; t--){
                t_idx = t * BCHW + b * CHW + i;

                // gradient of y_bn
                grad_y_bn_tmp = grad_y_bn[t_idx];
                x_bn_tmp = (x_bn[t_idx] - tmp_m) * tmp_ivar;

                grad_bias_tmp += grad_y_bn_tmp;
                grad_w_tmp += grad_y_bn_tmp * x_bn_tmp;
                grad_xhat_tmp += grad_y_bn_tmp * tmp_w;
                grad_xhatx_tmp += grad_y_bn_tmp * tmp_w * x_bn_tmp;
            }

        }

        grad_bias[i] = grad_bias_tmp;
        grad_weight[i] = grad_w_tmp;
        grad_xhat[i] = grad_xhat_tmp / (T*B);
        grad_xhatx[i] = grad_xhatx_tmp / (T*B);

    }
}


template <typename scalar_t>
__global__ void bn_bp_x_kernel(
    const scalar_t* __restrict__ grad_y_bn,
    const scalar_t* __restrict__ x_bn,
    const scalar_t* __restrict__ ivar,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ weight,

    const scalar_t* __restrict__ grad_xhat,
    const scalar_t* __restrict__ grad_xhatx,

    scalar_t* __restrict__ grad_x_bn,

    const int T,     
    const int B,
    const int HW,
    const int CHW,   
    const int BCHW
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int t_idx, c_idx;
    float tmp_w, tmp_ivar, tmp_m, tmp_xhat, tmp_xhatx;


    for(int i = idx; i < CHW; i += stride){

        c_idx      = int(i / HW);
        tmp_w      = weight[c_idx];
        tmp_ivar   = ivar[c_idx];
        tmp_m      = mean[c_idx];
        tmp_xhat   = grad_xhat[c_idx];
        tmp_xhatx  = grad_xhatx[c_idx];

        for(int b = 0; b < B; b++){

            for(int t = T - 1; t >=0; t--){
                t_idx = t * BCHW + b * CHW + i;

                grad_x_bn[t_idx] = grad_y_bn[t_idx] * tmp_w * tmp_ivar - \
                                tmp_ivar * ((x_bn[t_idx] - tmp_m) * tmp_ivar * tmp_xhatx + tmp_xhat);

            }
        }
    }
}


std::vector<torch::Tensor> bn_bp_cuda(
    torch::Tensor x_bn,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    torch::Tensor grad_y_bn
){
    int T = x_bn.size(0);
    int B = x_bn.size(1);
    int C = x_bn.size(2);
    int BCHW = x_bn.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    auto grad_x_bn = torch::empty_like(x_bn, x_bn.device());

    auto grad_xhat = torch::empty({CHW}, x_bn.device());
    auto grad_xhatx = torch::empty({CHW}, x_bn.device());
    auto grad_weight_bn = torch::empty({CHW}, x_bn.device());
    auto grad_bias_bn = torch::empty({CHW}, x_bn.device());

    if(HW == 1){
        grad_xhat = grad_xhat.view({1, 1, C});
        grad_xhatx = grad_xhatx.view({1, 1, C});
        grad_weight_bn = grad_weight_bn.view({1, 1, C});
        grad_bias_bn = grad_bias_bn.view({1, 1, C});
    }
    else{
        grad_xhat = grad_xhat.view({1, 1, C, x_bn.size(3), x_bn.size(4)});
        grad_xhatx = grad_xhatx.view({1, 1, C, x_bn.size(3), x_bn.size(4)});
        grad_weight_bn = grad_weight_bn.view({1, 1, C, x_bn.size(3), x_bn.size(4)});
        grad_bias_bn = grad_bias_bn.view({1, 1, C, x_bn.size(3), x_bn.size(4)});
    }
    

    AT_DISPATCH_ALL_TYPES(x_bn.type(), "bn_bp_wb_kernel",([&]{
        bn_bp_wb_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_y_bn.data<scalar_t>(), x_bn.data<scalar_t>(),  
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            grad_xhat.data<scalar_t>(), grad_xhatx.data<scalar_t>(),
            grad_weight_bn.data<scalar_t>(), grad_bias_bn.data<scalar_t>(),
            T, B, HW, CHW, BCHW);}));

    
    AT_DISPATCH_ALL_TYPES(x_bn.type(), "bn_bp_x_kernel",([&]{
        bn_bp_x_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_y_bn.data<scalar_t>(), x_bn.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            at::mean(grad_xhat, dim_bn).data<scalar_t>(),
            at::mean(grad_xhatx, dim_bn).data<scalar_t>(),
            grad_x_bn.data<scalar_t>(),
            T, B, HW, CHW, BCHW);}));


    return {grad_x_bn, at::sum(grad_weight_bn, dim_bn), at::sum(grad_bias_bn, dim_bn)};
}


std::vector<torch::Tensor> bn_bp_cuda_dist1(
    torch::Tensor x_bn,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    torch::Tensor grad_y_bn
){
    int T = x_bn.size(0);
    int B = x_bn.size(1);
    int C = x_bn.size(2);
    int BCHW = x_bn.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;


    auto grad_xhat = torch::empty({CHW}, x_bn.device());
    auto grad_xhatx = torch::empty({CHW}, x_bn.device());
    auto grad_weight_bn = torch::empty({CHW}, x_bn.device());
    auto grad_bias_bn = torch::empty({CHW}, x_bn.device());

    if(HW == 1){
        grad_xhat = grad_xhat.view({1, 1, C});
        grad_xhatx = grad_xhatx.view({1, 1, C});
        grad_weight_bn = grad_weight_bn.view({1, 1, C});
        grad_bias_bn = grad_bias_bn.view({1, 1, C});
    }
    else{
        grad_xhat = grad_xhat.view({1, 1, C, x_bn.size(3), x_bn.size(4)});
        grad_xhatx = grad_xhatx.view({1, 1, C, x_bn.size(3), x_bn.size(4)});
        grad_weight_bn = grad_weight_bn.view({1, 1, C, x_bn.size(3), x_bn.size(4)});
        grad_bias_bn = grad_bias_bn.view({1, 1, C, x_bn.size(3), x_bn.size(4)});
    }
    

    AT_DISPATCH_ALL_TYPES(x_bn.type(), "bn_bp_wb_kernel",([&]{
        bn_bp_wb_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_y_bn.data<scalar_t>(), x_bn.data<scalar_t>(),  
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            grad_xhat.data<scalar_t>(), grad_xhatx.data<scalar_t>(),
            grad_weight_bn.data<scalar_t>(), grad_bias_bn.data<scalar_t>(),
            T, B, HW, CHW, BCHW);}));

    return {at::mean(grad_xhat, dim_bn), at::mean(grad_xhatx, dim_bn), at::sum(grad_weight_bn, dim_bn), at::sum(grad_bias_bn, dim_bn)};

}

torch::Tensor bn_bp_cuda_dist2(
    torch::Tensor x_bn,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    torch::Tensor grad_y_bn,
    torch::Tensor grad_xhat,
    torch::Tensor grad_xhatx
){
    int T = x_bn.size(0);
    int B = x_bn.size(1);
    int C = x_bn.size(2);
    int BCHW = x_bn.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    auto grad_x_bn = torch::empty_like(x_bn, x_bn.device());

    AT_DISPATCH_ALL_TYPES(x_bn.type(), "bn_bp_x_kernel",([&]{
        bn_bp_x_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_y_bn.data<scalar_t>(), x_bn.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            grad_xhat.data<scalar_t>(), grad_xhatx.data<scalar_t>(),
            grad_x_bn.data<scalar_t>(),
            T, B, HW, CHW, BCHW);}));

    return grad_x_bn;
}


/*
bn fire forward u and backward kernel
*/

template <typename scalar_t>
__global__ void bn_fire_u_fp_kernel(
    const scalar_t* __restrict__ spatial, 
    scalar_t* __restrict__ mem,  
    const scalar_t* __restrict__ ivar,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
     
    const float alpha,
    const float thresh,
    const int T,     
    const int B,
    const int HW,
    const int CHW,   
    const int BCHW
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
        tmp_w   = weight[c_idx] * ivar[c_idx];
        tmp_b   = bias[c_idx];

        for(int b = 0; b < B; b++){

            u_t_pre = 0;
            s_t_pre = 0;

            for(int t = 0; t < T; t++){
                t_idx   = t * BCHW + b * CHW + i;
                u_t_pre = alpha * u_t_pre * (1 - s_t_pre) + (spatial[t_idx] - tmp_m) * tmp_w + tmp_b;
                s_t_pre = u_t_pre > thresh;

                mem[t_idx] = u_t_pre;
            }
        }
    }
}

template <typename scalar_t>
__global__ void bn_fire_us_fp_kernel(
    const scalar_t* __restrict__ spatial, 
    scalar_t* __restrict__ mem,  
    scalar_t* __restrict__ spike,
    const scalar_t* __restrict__ ivar,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
     
    const float alpha,
    const float thresh,
    const int T,     
    const int B,
    const int HW,
    const int CHW,   
    const int BCHW
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
        tmp_w   = weight[c_idx] * ivar[c_idx];
        tmp_b   = bias[c_idx];

        for(int b = 0; b < B; b++){

            u_t_pre = 0;
            s_t_pre = 0;

            for(int t = 0; t < T; t++){
                t_idx   = t * BCHW + b * CHW + i;
                u_t_pre = alpha * u_t_pre * (1 - s_t_pre) + (spatial[t_idx] - tmp_m) * tmp_w + tmp_b;
                s_t_pre = u_t_pre > thresh;

                mem[t_idx] = u_t_pre;
                spike[t_idx] = s_t_pre;
            }
        }
    }
}


template <typename scalar_t>
__global__ void bn_fire_bp_wb_kernel(
    const scalar_t* __restrict__ grad_spike,
    const scalar_t* __restrict__ spatial,
    const scalar_t* __restrict__ mem,
    const scalar_t* __restrict__ ivar,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ weight,

    scalar_t* __restrict__ grad_xhat,
    scalar_t* __restrict__ grad_xhatx,
    scalar_t* __restrict__ grad_weight,
    scalar_t* __restrict__ grad_bias,

    const float alpha,
    const float thresh,
    const float beta,
    const float interval,
    const int T,     
    const int B,
    const int HW,
    const int CHW,   
    const int BCHW
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int t_idx, c_idx;
    scalar_t grad_u_last, grad_s_last, u_tmp;
    float tmp_m, tmp_w, tmp_ivar;
    float x_bn, grad_bias_tmp, grad_w_tmp, grad_xhat_tmp, grad_xhatx_tmp;

    for(int i = idx; i < CHW; i += stride){

        c_idx    = int(i / HW);
        tmp_m    = mean[c_idx];
        tmp_w    = weight[c_idx];
        tmp_ivar = ivar[c_idx];

        grad_bias_tmp = 0;
        grad_w_tmp = 0;
        grad_xhat_tmp = 0;
        grad_xhatx_tmp = 0;

        for(int b = 0; b < B; b++){

            grad_u_last = 0;

            for(int t = T - 1; t >=0; t--){
                t_idx = t * BCHW + b * CHW + i;
                u_tmp = mem[t_idx];

                // gradient of spike
                grad_s_last = 0;
                if((u_tmp - thresh) < interval && (thresh - u_tmp) < interval){
                    grad_s_last = grad_spike[t_idx] - grad_u_last * alpha * u_tmp;
                }

                // gradient of mem
                grad_u_last = grad_s_last * beta + grad_u_last * alpha * (u_tmp <= thresh);

                x_bn = (spatial[t_idx] - tmp_m) * tmp_ivar;

                grad_bias_tmp += grad_u_last;
                grad_w_tmp += grad_u_last * x_bn;
                grad_xhat_tmp += grad_u_last * tmp_w;
                grad_xhatx_tmp += grad_u_last * tmp_w * x_bn;
            }

        }

        grad_bias[i] = grad_bias_tmp;
        grad_weight[i] = grad_w_tmp;
        grad_xhat[i] = grad_xhat_tmp / (T*B);
        grad_xhatx[i] = grad_xhatx_tmp / (T*B);

    }
}


template <typename scalar_t>
__global__ void bn_fire_bp_x_kernel(
    const scalar_t* __restrict__ grad_spike,
    const scalar_t* __restrict__ spatial,
    const scalar_t* __restrict__ mem,
    const scalar_t* __restrict__ ivar,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ weight,

    const scalar_t* __restrict__ grad_xhat,
    const scalar_t* __restrict__ grad_xhatx,

    scalar_t* __restrict__ grad_x,

    const float alpha,
    const float thresh,
    const float beta,
    const float interval,
    const int T,     
    const int B,
    const int HW,
    const int CHW,   
    const int BCHW
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int t_idx, c_idx;
    scalar_t grad_u_last, grad_s_last, u_tmp;
    float tmp_w, tmp_ivar, tmp_m, tmp_xhat, tmp_xhatx;


    for(int i = idx; i < CHW; i += stride){

        c_idx      = int(i / HW);
        tmp_w      = weight[c_idx];
        tmp_ivar   = ivar[c_idx];
        tmp_m      = mean[c_idx];
        tmp_xhat   = grad_xhat[c_idx];
        tmp_xhatx  = grad_xhatx[c_idx];

        for(int b = 0; b < B; b++){

            grad_u_last = 0;

            for(int t = T - 1; t >=0; t--){
                t_idx = t * BCHW + b * CHW + i;
                u_tmp = mem[t_idx];

                // gradient of spike
                grad_s_last = 0;
                if((u_tmp - thresh) < interval && (thresh - u_tmp) < interval){
                    grad_s_last = grad_spike[t_idx] - grad_u_last * alpha * u_tmp;
                }

                // gradient of mem
                grad_u_last = grad_s_last * beta + grad_u_last * alpha * (u_tmp <= thresh);

                grad_x[t_idx] = grad_u_last * tmp_w * tmp_ivar - \
                                tmp_ivar * ((spatial[t_idx] - tmp_m) * tmp_ivar * tmp_xhatx + tmp_xhat);

            }
        }
    }
}


std::vector<torch::Tensor> bn_fire_bp_cuda(
    torch::Tensor spatial,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    torch::Tensor grad_spike,
    float alpha,
    float thresh,
    float beta,
    float interval
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    auto mem = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_u_fp_kernel",([&]{
        bn_fire_u_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), bias_bn.data<scalar_t>(),
            alpha, thresh, T, B, HW, CHW, BCHW);}));

    
    auto grad_spatial = torch::empty_like(spatial, spatial.device());

    auto grad_xhat = torch::empty({CHW}, spatial.device());
    auto grad_xhatx = torch::empty({CHW}, spatial.device());
    auto grad_weight_bn = torch::empty({CHW}, spatial.device());
    auto grad_bias_bn = torch::empty({CHW}, spatial.device());

    if(HW == 1){
        grad_xhat = grad_xhat.view({1, 1, C});
        grad_xhatx = grad_xhatx.view({1, 1, C});
        grad_weight_bn = grad_weight_bn.view({1, 1, C});
        grad_bias_bn = grad_bias_bn.view({1, 1, C});
    }
    else{
        grad_xhat = grad_xhat.view({1, 1, C, spatial.size(3), spatial.size(4)});
        grad_xhatx = grad_xhatx.view({1, 1, C, spatial.size(3), spatial.size(4)});
        grad_weight_bn = grad_weight_bn.view({1, 1, C, spatial.size(3), spatial.size(4)});
        grad_bias_bn = grad_bias_bn.view({1, 1, C, spatial.size(3), spatial.size(4)});
    }
    

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_bp_wb_kernel",([&]{
        bn_fire_bp_wb_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            grad_xhat.data<scalar_t>(), grad_xhatx.data<scalar_t>(),
            grad_weight_bn.data<scalar_t>(), grad_bias_bn.data<scalar_t>(),
            alpha, thresh, beta, interval, T, B, HW, CHW, BCHW);}));

    
    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_bp_x_kernel",([&]{
        bn_fire_bp_x_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            at::mean(grad_xhat, dim_bn).data<scalar_t>(),
            at::mean(grad_xhatx, dim_bn).data<scalar_t>(),
            grad_spatial.data<scalar_t>(),
            alpha, thresh, beta, interval, T, B, HW, CHW, BCHW);}));


    return {grad_spatial, at::sum(grad_weight_bn, dim_bn), at::sum(grad_bias_bn, dim_bn)};
}


std::vector<torch::Tensor> bn_fire_bp_cuda_dist1(
    torch::Tensor spatial,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    torch::Tensor grad_spike,
    float alpha,
    float thresh,
    float beta,
    float interval
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    auto mem = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_u_fp_kernel",([&]{
        bn_fire_u_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), bias_bn.data<scalar_t>(),
            alpha, thresh, T, B, HW, CHW, BCHW);}));

    

    auto grad_xhat = torch::empty({CHW}, spatial.device());
    auto grad_xhatx = torch::empty({CHW}, spatial.device());
    auto grad_weight_bn = torch::empty({CHW}, spatial.device());
    auto grad_bias_bn = torch::empty({CHW}, spatial.device());

    if(HW == 1){
        grad_xhat = grad_xhat.view({1, 1, C});
        grad_xhatx = grad_xhatx.view({1, 1, C});
        grad_weight_bn = grad_weight_bn.view({1, 1, C});
        grad_bias_bn = grad_bias_bn.view({1, 1, C});
    }
    else{
        grad_xhat = grad_xhat.view({1, 1, C, spatial.size(3), spatial.size(4)});
        grad_xhatx = grad_xhatx.view({1, 1, C, spatial.size(3), spatial.size(4)});
        grad_weight_bn = grad_weight_bn.view({1, 1, C, spatial.size(3), spatial.size(4)});
        grad_bias_bn = grad_bias_bn.view({1, 1, C, spatial.size(3), spatial.size(4)});
    }
    

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_bp_wb_kernel",([&]{
        bn_fire_bp_wb_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            grad_xhat.data<scalar_t>(), grad_xhatx.data<scalar_t>(),
            grad_weight_bn.data<scalar_t>(), grad_bias_bn.data<scalar_t>(),
            alpha, thresh, beta, interval, T, B, HW, CHW, BCHW);}));

    return {at::mean(grad_xhat, dim_bn), at::mean(grad_xhatx, dim_bn), at::sum(grad_weight_bn, dim_bn), at::sum(grad_bias_bn, dim_bn)};

}

torch::Tensor bn_fire_bp_cuda_dist2(
    torch::Tensor spatial,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    torch::Tensor grad_spike,
    torch::Tensor grad_xhat,
    torch::Tensor grad_xhatx,
    float alpha,
    float thresh,
    float beta,
    float interval
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    auto mem = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_u_fp_kernel",([&]{
        bn_fire_u_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), bias_bn.data<scalar_t>(),
            alpha, thresh, T, B, HW, CHW, BCHW);}));

    auto grad_spatial = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_bp_x_kernel",([&]{
        bn_fire_bp_x_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            grad_xhat.data<scalar_t>(), grad_xhatx.data<scalar_t>(),
            grad_spatial.data<scalar_t>(),
            alpha, thresh, beta, interval, T, B, HW, CHW, BCHW);}));

    return grad_spatial;
}

/*
fc layers
*/

std::vector<torch::Tensor> bn_fire_fc_bp_cuda(
    torch::Tensor spatial,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    
    torch::Tensor grad_fc,
    torch::Tensor weight_fc,
    torch::Tensor bias_fc,
    c10::ArrayRef<int64_t> dim_fc,

    float alpha,
    float thresh,
    float beta,
    float interval
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;


    // fp for mem and spike
    auto mem = torch::empty_like(spatial, spatial.device());
    auto spike = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_us_fp_kernel",([&]{
        bn_fire_us_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), mem.data<scalar_t>(), spike.data<scalar_t>(),
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), bias_bn.data<scalar_t>(),
            alpha, thresh, T, B, HW, CHW, BCHW);}));

    
    // bp for fc
    auto grad_spike = at::empty_like(spatial, spatial.device());
    auto grad_weight_fc = at::zeros_like(weight_fc, weight_fc.device());
    
    if(HW==1){
        for(int t = 0; t < T; t++){
            grad_spike.data()[t] = at::mm(grad_fc.data()[t], weight_fc);
            grad_weight_fc += at::mm(at::transpose(grad_fc.data()[t], 0, 1), spike.data()[t]);
        }
    }
    else{
        spike = spike.view({T, spatial.size(1), -1});
        int H = spatial.size(3);
        int W = spatial.size(4);
        grad_spike = grad_spike.view({T, spatial.size(1), -1});

        for(int t = 0; t < T; t++){
            grad_spike.data()[t] = at::mm(grad_fc.data()[t], weight_fc);
            grad_weight_fc += at::mm(at::transpose(grad_fc.data()[t], 0, 1), spike.data()[t]);
        }
        
        grad_spike = grad_spike.view({T, spatial.size(1), C, H, W});
    }
            
    // bp for bn and fire
    auto grad_spatial = torch::empty_like(spatial, spatial.device());

    auto grad_xhat = torch::empty({CHW}, spatial.device());
    auto grad_xhatx = torch::empty({CHW}, spatial.device());
    auto grad_weight_bn = torch::empty({CHW}, spatial.device());
    auto grad_bias_bn = torch::empty({CHW}, spatial.device());

    if(HW == 1){
        grad_xhat = grad_xhat.view({1, 1, C});
        grad_xhatx = grad_xhatx.view({1, 1, C});
        grad_weight_bn = grad_weight_bn.view({1, 1, C});
        grad_bias_bn = grad_bias_bn.view({1, 1, C});
    }
    else{
        grad_xhat = grad_xhat.view({1, 1, C, spatial.size(3), spatial.size(4)});
        grad_xhatx = grad_xhatx.view({1, 1, C, spatial.size(3), spatial.size(4)});
        grad_weight_bn = grad_weight_bn.view({1, 1, C, spatial.size(3), spatial.size(4)});
        grad_bias_bn = grad_bias_bn.view({1, 1, C, spatial.size(3), spatial.size(4)});
    }
    

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_bp_wb_kernel",([&]{
        bn_fire_bp_wb_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            grad_xhat.data<scalar_t>(), grad_xhatx.data<scalar_t>(),
            grad_weight_bn.data<scalar_t>(), grad_bias_bn.data<scalar_t>(),
            alpha, thresh, beta, interval, T, B, HW, CHW, BCHW);}));

    
    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_bp_x_kernel",([&]{
        bn_fire_bp_x_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            at::mean(grad_xhat, dim_bn).data<scalar_t>(),
            at::mean(grad_xhatx, dim_bn).data<scalar_t>(),
            grad_spatial.data<scalar_t>(),
            alpha, thresh, beta, interval, T, B, HW, CHW, BCHW);}));

    return {grad_spatial, at::sum(grad_weight_bn, dim_bn), at::sum(grad_bias_bn, dim_bn), grad_weight_fc, at::sum(grad_fc, dim_fc)};   
}



std::vector<torch::Tensor> bn_fire_fc_bp_cuda_dist1(
    torch::Tensor spatial,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    
    torch::Tensor grad_fc,
    torch::Tensor weight_fc,
    torch::Tensor bias_fc,
    c10::ArrayRef<int64_t> dim_fc,

    float alpha,
    float thresh,
    float beta,
    float interval
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    // fp for mem and spike
    auto mem = torch::empty_like(spatial, spatial.device());
    auto spike = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_us_fp_kernel",([&]{
        bn_fire_us_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), mem.data<scalar_t>(), spike.data<scalar_t>(),
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), bias_bn.data<scalar_t>(),
            alpha, thresh, T, B, HW, CHW, BCHW);}));

    
    // bp for fc
    auto grad_spike = at::empty_like(spatial, spatial.device());
    auto grad_weight_fc = at::zeros_like(weight_fc, weight_fc.device());
    
    if(HW==1){
        for(int t = 0; t < T; t++){
            grad_spike.data()[t] = at::mm(grad_fc.data()[t], weight_fc);
            grad_weight_fc += at::mm(at::transpose(grad_fc.data()[t], 0, 1), spike.data()[t]);
        }
    }
    else{
        spike = spike.view({T, spatial.size(1), -1});
        int H = spatial.size(3);
        int W = spatial.size(4);
        grad_spike = grad_spike.view({T, spatial.size(1), -1});

        for(int t = 0; t < T; t++){
            grad_spike.data()[t] = at::mm(grad_fc.data()[t], weight_fc);
            grad_weight_fc += at::mm(at::transpose(grad_fc.data()[t], 0, 1), spike.data()[t]);
        }
        
        grad_spike = grad_spike.view({T, spatial.size(1), C, H, W});
    }
            
    // bp for bn and fire
    auto grad_xhat = torch::empty({CHW}, spatial.device());
    auto grad_xhatx = torch::empty({CHW}, spatial.device());
    auto grad_weight_bn = torch::empty({CHW}, spatial.device());
    auto grad_bias_bn = torch::empty({CHW}, spatial.device());

    if(HW == 1){
        grad_xhat = grad_xhat.view({1, 1, C});
        grad_xhatx = grad_xhatx.view({1, 1, C});
        grad_weight_bn = grad_weight_bn.view({1, 1, C});
        grad_bias_bn = grad_bias_bn.view({1, 1, C});
    }
    else{
        grad_xhat = grad_xhat.view({1, 1, C, spatial.size(3), spatial.size(4)});
        grad_xhatx = grad_xhatx.view({1, 1, C, spatial.size(3), spatial.size(4)});
        grad_weight_bn = grad_weight_bn.view({1, 1, C, spatial.size(3), spatial.size(4)});
        grad_bias_bn = grad_bias_bn.view({1, 1, C, spatial.size(3), spatial.size(4)});
    }
    

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_bp_wb_kernel",([&]{
        bn_fire_bp_wb_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            grad_xhat.data<scalar_t>(), grad_xhatx.data<scalar_t>(),
            grad_weight_bn.data<scalar_t>(), grad_bias_bn.data<scalar_t>(),
            alpha, thresh, beta, interval, T, B, HW, CHW, BCHW);}));

    
    return {
        grad_spike, at::mean(grad_xhat, dim_bn), at::mean(grad_xhatx, dim_bn), 
        at::sum(grad_weight_bn, dim_bn), at::sum(grad_bias_bn, dim_bn),
        grad_weight_fc, at::sum(grad_fc, dim_fc)
    };

}



std::vector<torch::Tensor> fire_fc_bp_cuda(
    torch::Tensor spatial,
    torch::Tensor grad_fc,
    torch::Tensor weight_fc,
    torch::Tensor bias_fc,
    c10::ArrayRef<int64_t> dim_fc,

    float alpha,
    float thresh,
    float beta,
    float interval
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;


    // fp for mem and spike
    auto mem = torch::empty_like(spatial, spatial.device());
    auto spike = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "fire_us_fp_kernel",([&]{
        fire_us_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), mem.data<scalar_t>(), spike.data<scalar_t>(),
            alpha, thresh, T, B, CHW, BCHW);}));

    
    // bp for fc
    auto grad_spike = at::empty_like(spatial, spatial.device());
    auto grad_weight_fc = at::zeros_like(weight_fc, weight_fc.device());
    
    if(HW==1){
        for(int t = 0; t < T; t++){
            grad_spike.data()[t] = at::mm(grad_fc.data()[t], weight_fc);
            grad_weight_fc += at::mm(at::transpose(grad_fc.data()[t], 0, 1), spike.data()[t]);
        }
    }
    else{
        spike = spike.view({T, B, -1});
        grad_spike = grad_spike.view({T, B, -1});

        for(int t = 0; t < T; t++){
            grad_spike.data()[t] = at::mm(grad_fc.data()[t], weight_fc);
            grad_weight_fc += at::mm(at::transpose(grad_fc.data()[t], 0, 1), spike.data()[t]);
        }
        
        int H = spatial.size(3);
        int W = spatial.size(4);
        grad_spike = grad_spike.view({T, B, C, H, W});
    }
            
    // bp for fire
    auto grad_spatial = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "fire_bp_kernel",([&]{
        fire_bp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), mem.data<scalar_t>(), grad_spatial.data<scalar_t>(), 
            alpha, thresh, beta, interval, T, B, CHW, BCHW);}));
        
    return {grad_spatial, grad_weight_fc, at::sum(grad_fc, dim_fc)};
}


std::vector<torch::Tensor> fc_bp_cuda(
    torch::Tensor spatial,
    torch::Tensor grad_fc,
    torch::Tensor weight_fc,
    torch::Tensor bias_fc,
    c10::ArrayRef<int64_t> dim_fc
){
    int T = spatial.size(0);
    int C = spatial.size(2);
    int N = spatial.numel() / T;
    int HW = N / (spatial.size(1) * C);

    // fp for mem and spike
    auto mem = torch::empty_like(spatial, spatial.device());
    auto spike = torch::empty_like(spatial, spatial.device());

    // bp for fc
    auto grad_spatial = at::empty_like(spatial, spatial.device());
    auto grad_weight_fc = at::zeros_like(weight_fc, weight_fc.device());
    
    if(HW==1){
        for(int t = 0; t < T; t++){
            grad_spatial.data()[t] = at::mm(grad_fc.data()[t], weight_fc);
            grad_weight_fc += at::mm(at::transpose(grad_fc.data()[t], 0, 1), spatial.data()[t]);
        }
    }
    else{
        int H = spatial.size(3);
        int W = spatial.size(4);
        spatial = spatial.view({T, spatial.size(1), -1});
        grad_spatial = grad_spatial.view({T, spatial.size(1), -1});


        for(int t = 0; t < T; t++){
            grad_spatial.data()[t] = at::mm(grad_fc.data()[t], weight_fc);
            grad_weight_fc += at::mm(at::transpose(grad_fc.data()[t], 0, 1), spatial.data()[t]);
        }
        
        grad_spatial = grad_spatial.view({T, spatial.size(1), C, H, W});
    }
            
    return {grad_spatial, grad_weight_fc, at::sum(grad_fc, dim_fc)};
}



/*
conv layers
*/
std::vector<torch::Tensor> bn_fire_conv_bp_cuda(
    torch::Tensor spatial,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    
    torch::Tensor grad_conv,
    const at::Tensor& weight_conv,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    c10::ArrayRef<int64_t> dim_conv,

    float alpha,
    float thresh,
    float beta,
    float interval
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    // fp for mem and spike
    auto mem = torch::empty_like(spatial, spatial.device());
    auto spike = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_us_fp_kernel",([&]{
        bn_fire_us_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), mem.data<scalar_t>(), spike.data<scalar_t>(),
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), bias_bn.data<scalar_t>(),
            alpha, thresh, T, B, HW, CHW, BCHW);}));

    
    // bp for conv
    auto grad_spike = at::empty_like(spatial, spatial.device());
    auto grad_weight_conv = at::zeros_like(weight_conv, weight_conv.device());

    for(int t = 0; t < T; t++){
        grad_spike.data()[t] = at::cudnn_convolution_backward_input(
            {spatial.size(1), spatial.size(2), spatial.size(3), spatial.size(4)}, 
            grad_conv.data()[t], weight_conv, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);

        grad_weight_conv += at::cudnn_convolution_backward_weight(
            {weight_conv.size(0), weight_conv.size(1), weight_conv.size(2), weight_conv.size(3)},
            grad_conv.data()[t], spike.data()[t], padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
    }
            
    // bp for bn and fire
    auto grad_spatial = torch::empty_like(spatial, spatial.device());
    auto grad_xhat = torch::empty({1, 1, C, spatial.size(3), spatial.size(4)}, spatial.device());
    auto grad_xhatx = torch::empty({1, 1, C, spatial.size(3), spatial.size(4)}, spatial.device());
    auto grad_weight_bn = torch::empty({1, 1, C, spatial.size(3), spatial.size(4)}, spatial.device());
    auto grad_bias_bn = torch::empty({1, 1, C, spatial.size(3), spatial.size(4)}, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_bp_wb_kernel",([&]{
        bn_fire_bp_wb_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            grad_xhat.data<scalar_t>(), grad_xhatx.data<scalar_t>(),
            grad_weight_bn.data<scalar_t>(), grad_bias_bn.data<scalar_t>(),
            alpha, thresh, beta, interval, T, B, HW, CHW, BCHW);}));

    
    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_bp_x_kernel",([&]{
        bn_fire_bp_x_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            at::mean(grad_xhat, dim_bn).data<scalar_t>(),
            at::mean(grad_xhatx, dim_bn).data<scalar_t>(),
            grad_spatial.data<scalar_t>(),
            alpha, thresh, beta, interval, T, B, HW, CHW, BCHW);}));

    return {grad_spatial, at::sum(grad_weight_bn, dim_bn), at::sum(grad_bias_bn, dim_bn), grad_weight_conv, at::sum(grad_conv, dim_conv)};   
}


std::vector<torch::Tensor> bn_fire_conv_bp_cuda_dist1(
    torch::Tensor spatial,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    
    torch::Tensor grad_conv,
    const at::Tensor& weight_conv,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    c10::ArrayRef<int64_t> dim_conv,

    float alpha,
    float thresh,
    float beta,
    float interval
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;
    int HW = CHW / C;

    // fp for mem and spike
    auto mem = torch::empty_like(spatial, spatial.device());
    auto spike = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_us_fp_kernel",([&]{
        bn_fire_us_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), mem.data<scalar_t>(), spike.data<scalar_t>(),
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), bias_bn.data<scalar_t>(),
            alpha, thresh, T, B, HW, CHW, BCHW);}));

    
    // bp for conv
    auto grad_spike = at::empty_like(spatial, spatial.device());
    auto grad_weight_conv = at::zeros_like(weight_conv, weight_conv.device());

    for(int t = 0; t < T; t++){
        grad_spike.data()[t] = at::cudnn_convolution_backward_input(
            {spatial.size(1), spatial.size(2), spatial.size(3), spatial.size(4)}, 
            grad_conv.data()[t], weight_conv, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);

        grad_weight_conv += at::cudnn_convolution_backward_weight(
            {weight_conv.size(0), weight_conv.size(1), weight_conv.size(2), weight_conv.size(3)},
            grad_conv.data()[t], spike.data()[t], padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
    }
            
    // bp for bn and fire
    auto grad_xhat = torch::empty({1, 1, C, spatial.size(3), spatial.size(4)}, spatial.device());
    auto grad_xhatx = torch::empty({1, 1, C, spatial.size(3), spatial.size(4)}, spatial.device());
    auto grad_weight_bn = torch::empty({1, 1, C, spatial.size(3), spatial.size(4)}, spatial.device());
    auto grad_bias_bn = torch::empty({1, 1, C, spatial.size(3), spatial.size(4)}, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "bn_fire_bp_wb_kernel",([&]{
        bn_fire_bp_wb_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), spatial.data<scalar_t>(), mem.data<scalar_t>(), 
            ivar.data<scalar_t>(), mean.data<scalar_t>(), weight_bn.data<scalar_t>(), 
            grad_xhat.data<scalar_t>(), grad_xhatx.data<scalar_t>(),
            grad_weight_bn.data<scalar_t>(), grad_bias_bn.data<scalar_t>(),
            alpha, thresh, beta, interval, T, B, HW, CHW, BCHW);}));

    return {
        grad_spike, at::mean(grad_xhat, dim_bn), at::mean(grad_xhatx, dim_bn), 
        at::sum(grad_weight_bn, dim_bn), at::sum(grad_bias_bn, dim_bn),
        grad_weight_conv, at::sum(grad_conv, dim_conv)
    };
}



std::vector<torch::Tensor> fire_conv_bp_cuda(
    torch::Tensor spatial,    
    torch::Tensor grad_conv,
    const at::Tensor& weight_conv,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    c10::ArrayRef<int64_t> dim_conv,

    float alpha,
    float thresh,
    float beta,
    float interval
){
    int T = spatial.size(0);
    int B = spatial.size(1);
    int C = spatial.size(2);
    int BCHW = spatial.numel() / T;
    int CHW = BCHW / B;

    // fp for mem and spike
    auto mem = torch::empty_like(spatial, spatial.device());
    auto spike = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "fire_us_fp_kernel",([&]{
        fire_us_fp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            spatial.data<scalar_t>(), mem.data<scalar_t>(), spike.data<scalar_t>(),
            alpha, thresh, T, B, CHW, BCHW);}));

    
    // bp for conv
    auto grad_spike = at::empty_like(spatial, spatial.device());
    auto grad_weight_conv = at::zeros_like(weight_conv, weight_conv.device());

    for(int t = 0; t < T; t++){
        grad_spike.data()[t] = at::cudnn_convolution_backward_input(
            {spatial.size(1), spatial.size(2), spatial.size(3), spatial.size(4)}, 
            grad_conv.data()[t], weight_conv, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);

        grad_weight_conv += at::cudnn_convolution_backward_weight(
            {weight_conv.size(0), weight_conv.size(1), weight_conv.size(2), weight_conv.size(3)},
            grad_conv.data()[t], spike.data()[t], padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
    }
            
    // bp for fire
    auto grad_spatial = torch::empty_like(spatial, spatial.device());

    AT_DISPATCH_ALL_TYPES(spatial.type(), "fire_bp_kernel",([&]{
        fire_bp_kernel<scalar_t><<<BLOCKS(CHW, THREAD_NUM), THREAD_NUM>>>(
            grad_spike.data<scalar_t>(), mem.data<scalar_t>(), grad_spatial.data<scalar_t>(), 
            alpha, thresh, beta, interval, T, B, CHW, BCHW);}));
        

    return {grad_spatial, grad_weight_conv, at::sum(grad_conv, dim_conv)};   
}



std::vector<torch::Tensor> conv_bp_cuda(
    torch::Tensor spatial,    
    torch::Tensor grad_conv,
    const at::Tensor& weight_conv,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    c10::ArrayRef<int64_t> dim_conv
){
    int T = spatial.size(0);
    
    // bp for conv
    auto grad_spatial = at::empty_like(spatial, spatial.device());
    auto grad_weight_conv = at::zeros_like(weight_conv, weight_conv.device());

    for(int t = 0; t < T; t++){
        grad_spatial.data()[t] = at::cudnn_convolution_backward_input(
            {spatial.size(1), spatial.size(2), spatial.size(3), spatial.size(4)}, 
            grad_conv.data()[t], weight_conv, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);

        grad_weight_conv += at::cudnn_convolution_backward_weight(
            {weight_conv.size(0), weight_conv.size(1), weight_conv.size(2), weight_conv.size(3)},
            grad_conv.data()[t], spatial.data()[t], padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
    }

    return {grad_spatial, grad_weight_conv, at::sum(grad_conv, dim_conv)};   
}
