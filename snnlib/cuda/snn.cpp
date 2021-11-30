#include <torch/extension.h>
#include <vector>

// fire fp
torch::Tensor fire_s_fp_cuda(
    torch::Tensor spatial,
    float alpha,
    float thresh
);

torch::Tensor fire_s_fp(
    torch::Tensor spatial,
    float alpha,
    float thresh
){
    return fire_s_fp_cuda(spatial, alpha, thresh);
}


// bn fp
std::vector<torch::Tensor> bn_fp_cuda(
    torch::Tensor x_bn,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor var,
    torch::Tensor mean,
    float eps_bn
);

std::vector<torch::Tensor> bn_fp(
    torch::Tensor x_bn,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor var,
    torch::Tensor mean,
    float eps_bn
){
    return bn_fp_cuda(x_bn, weight_bn, bias_bn, var, mean, eps_bn);
}



// bn fire fp
std::vector<torch::Tensor> bn_fire_s_fp_cuda(
    torch::Tensor spatial,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor var,
    torch::Tensor mean,
    float eps_bn,
    float alpha,
    float thresh
);

std::vector<torch::Tensor> bn_fire_s_fp(
    torch::Tensor spatial,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor var,
    torch::Tensor mean,
    float eps_bn,
    float alpha,
    float thresh
){
    return bn_fire_s_fp_cuda(spatial, weight_bn, bias_bn, var, mean, eps_bn, alpha, thresh);
}

// bn fire fc fp
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
);

std::vector<torch::Tensor> bn_fire_fc_fp(
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
    return bn_fire_fc_fp_cuda(spatial, weight_bn, bias_bn, var, mean, eps_bn, weight_fc, bias_fc, alpha, thresh);
}

// fire fc fp
torch::Tensor fire_fc_fp_cuda(
    torch::Tensor spatial,
    torch::Tensor weight_fc,
    torch::Tensor bias_fc,
    float alpha,
    float thresh
);

torch::Tensor fire_fc_fp(
    torch::Tensor spatial,
    torch::Tensor weight_fc,
    torch::Tensor bias_fc,
    float alpha,
    float thresh
){
    return fire_fc_fp_cuda(spatial, weight_fc, bias_fc, alpha, thresh);
}


// fc fp
torch::Tensor fc_fp_cuda(
    torch::Tensor spatial,
    torch::Tensor weight_fc,
    torch::Tensor bias_fc
);

torch::Tensor fc_fp(
    torch::Tensor spatial,
    torch::Tensor weight_fc,
    torch::Tensor bias_fc
){
    return fc_fp_cuda(spatial, weight_fc, bias_fc);
}


// bn fire conv
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
);

std::vector<torch::Tensor> bn_fire_conv_fp(
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
    return bn_fire_conv_fp_cuda(
        spatial, weight_bn, bias_bn, var, mean, eps_bn, 
        weight_conv, bias_conv, stride, padding, dilation, groups, benchmark, deterministic,
        alpha, thresh);
}

// fire conv
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
);

torch::Tensor fire_conv_fp(
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
    return fire_conv_fp_cuda(
        spatial, weight_conv, bias_conv, stride, padding, dilation, groups, benchmark, deterministic, alpha, thresh);
}

// conv fp
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
);

torch::Tensor conv_fp(
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
    return conv_fp_cuda(
        spatial, weight_conv, bias_conv, stride, padding, dilation, groups, benchmark, deterministic);
}


/*
******** backward function
*/

// fire bp
torch::Tensor fire_bp_cuda(
    torch::Tensor spatial,
    torch::Tensor grad_spike,
    float alpha,
    float thresh,
    float beta,
    float interval
);

torch::Tensor fire_bp(
    torch::Tensor spatial,
    torch::Tensor grad_spike,
    float alpha,
    float thresh,
    float beta,
    float interval
){
    return fire_bp_cuda(spatial, grad_spike, alpha, thresh, beta, interval);
}


// bn bp
std::vector<torch::Tensor> bn_bp_cuda(
    torch::Tensor x_bn,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    torch::Tensor grad_y_bn
);

std::vector<torch::Tensor> bn_bp(
    torch::Tensor x_bn,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    torch::Tensor grad_y_bn
){
    return bn_bp_cuda(x_bn, weight_bn, bias_bn, ivar, mean, dim_bn, grad_y_bn);
}


// bn bp distributtion 1
std::vector<torch::Tensor> bn_bp_cuda_dist1(
    torch::Tensor x_bn,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    torch::Tensor grad_y_bn
);

std::vector<torch::Tensor> bn_bp_dist1(
    torch::Tensor x_bn,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor ivar,
    torch::Tensor mean,
    c10::ArrayRef<int64_t> dim_bn,
    torch::Tensor grad_y_bn
){
    return bn_bp_cuda_dist1(x_bn, weight_bn, bias_bn, ivar, mean, dim_bn, grad_y_bn);
}


// bn bp distribution 2
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
);

torch::Tensor bn_bp_dist2(
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
    return bn_bp_cuda_dist2(x_bn, weight_bn, bias_bn, ivar, mean, dim_bn, grad_y_bn, grad_xhat, grad_xhatx);
}


// bn fire bp
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
);

std::vector<torch::Tensor> bn_fire_bp(
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
    return bn_fire_bp_cuda(spatial, weight_bn, bias_bn, ivar, mean, dim_bn, grad_spike, alpha, thresh, beta, interval);
}


// bn fire bp distribute 1
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
);

std::vector<torch::Tensor> bn_fire_bp_dist1(
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
    return bn_fire_bp_cuda_dist1(
        spatial, weight_bn, bias_bn, ivar, mean, dim_bn, grad_spike, alpha, thresh, beta, interval
    );
}



// bn fire bp distribute 2
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
);


torch::Tensor bn_fire_bp_dist2(
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
    return bn_fire_bp_cuda_dist2(
        spatial, weight_bn, bias_bn, ivar, mean, grad_spike, grad_xhat, grad_xhatx, 
        alpha, thresh, beta, interval
    );
}



// bn fire fc bp
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
);

std::vector<torch::Tensor> bn_fire_fc_bp(
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
    return bn_fire_fc_bp_cuda(
        spatial, weight_bn, bias_bn, ivar, mean, dim_bn, grad_fc, weight_fc, bias_fc, dim_fc, alpha, thresh, beta, interval);
}


// bn fire fc bp distribute 1
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
);


std::vector<torch::Tensor> bn_fire_fc_bp_dist1(
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
    return bn_fire_fc_bp_cuda_dist1(
        spatial, weight_bn, bias_bn, ivar, mean, dim_bn, 
        grad_fc, weight_fc, bias_fc, dim_fc,
        alpha, thresh, beta, interval
    );
}



// fire fc bp
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
);

std::vector<torch::Tensor> fire_fc_bp(
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
    return fire_fc_bp_cuda(spatial, grad_fc, weight_fc, bias_fc, dim_fc, alpha, thresh, beta, interval);
}


// fc bp
std::vector<torch::Tensor> fc_bp_cuda(
    torch::Tensor spatial,
    torch::Tensor grad_fc,
    torch::Tensor weight_fc,
    torch::Tensor bias_fc,
    c10::ArrayRef<int64_t> dim_fc
);

std::vector<torch::Tensor> fc_bp(
    torch::Tensor spatial,
    torch::Tensor grad_fc,
    torch::Tensor weight_fc,
    torch::Tensor bias_fc,
    c10::ArrayRef<int64_t> dim_fc
){
    return fc_bp_cuda(spatial, grad_fc, weight_fc, bias_fc, dim_fc);
}

// bn fire conv backward
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
);

std::vector<torch::Tensor> bn_fire_conv_bp(
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
    return bn_fire_conv_bp_cuda(
        spatial, weight_bn, bias_bn, ivar, mean, dim_bn,
        grad_conv, weight_conv, stride, padding, dilation, groups, benchmark, deterministic, allow_tf32, dim_conv,
        alpha, thresh, beta, interval);
}


// bn fire conv bp cuda dist1
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
);

std::vector<torch::Tensor> bn_fire_conv_bp_dist1(
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
    return bn_fire_conv_bp_cuda_dist1(
        spatial, weight_bn, bias_bn, ivar, mean, dim_bn,
        grad_conv, weight_conv, stride, padding, dilation, groups, benchmark, deterministic, allow_tf32, dim_conv,
        alpha, thresh, beta, interval
    );
}



// fire conv bp
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
);

std::vector<torch::Tensor> fire_conv_bp(
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
    return fire_conv_bp_cuda(
        spatial, grad_conv, weight_conv, stride, padding, dilation,
        groups, benchmark, deterministic, allow_tf32, dim_conv,
        alpha, thresh, beta, interval);
}

// conv bp
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
);

std::vector<torch::Tensor> conv_bp(
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
    return conv_bp_cuda(
        spatial, grad_conv, weight_conv, stride, padding, dilation, groups, benchmark, deterministic, allow_tf32, dim_conv);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("fire_s_fp", &fire_s_fp, "fire (spike) forward");
    m.def("bn_fire_s_fp", &bn_fire_s_fp, "bn fire (spike) forward");

    m.def("bn_fp", &bn_fp, "bn forward");
    m.def("bn_bp", &bn_bp, "bn backward");

    m.def("bn_fire_fc_fp", &bn_fire_fc_fp, "bn fire fc forward");
    m.def("fire_fc_fp", &fire_fc_fp, "fire fc forward");
    m.def("fc_fp", &fc_fp, "fc forward");

    m.def("bn_fire_conv_fp", &bn_fire_conv_fp, "bn fire conv forward");
    m.def("fire_conv_fp", &fire_conv_fp, "fire conv forward");
    m.def("conv_fp", &conv_fp, "conv forward");

    m.def("fire_bp", &fire_bp, "fire backward");
    m.def("bn_fire_bp", &bn_fire_bp, "bn fire backward");

    m.def("bn_fire_fc_bp", &bn_fire_fc_bp, "bn fire fc backward");
    m.def("fire_fc_bp", &fire_fc_bp, "fire fc backward");
    m.def("fc_bp", &fc_bp, "fc backward");

    m.def("bn_fire_conv_bp", &bn_fire_conv_bp, "bn fire conv backward");
    m.def("fire_conv_bp", &fire_conv_bp, "fire conv backward");
    m.def("conv_bp", &conv_bp, "conv backward");

    m.def("bn_bp_dist1", &bn_bp_dist1, "bn backward distribute 1");
    m.def("bn_bp_dist2", &bn_bp_dist2, "bn backward distribute 2");
    m.def("bn_fire_bp_dist1", &bn_fire_bp_dist1, "bn fire backward distribute 1");
    m.def("bn_fire_bp_dist2", &bn_fire_bp_dist2, "bn fire backward distribute 2");
    m.def("bn_fire_fc_bp_dist1", &bn_fire_fc_bp_dist1, "bn fire fc backward distribute 1");
    m.def("bn_fire_conv_bp_dist1", &bn_fire_conv_bp_dist1, "bn fire conv backward distrubute");
}

