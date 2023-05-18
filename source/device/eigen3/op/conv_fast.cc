#include "op.hpp"
#include "op/registry.hpp"
#include <array>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMap.h>
#include <memory>
#include <vector>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif
#include "operator/op.h"
#include "operator/prototype/convolution_param.h"
#include "graph/tensor.h"
#include "graph/graph.h"
#ifdef __cplusplus
}
#endif

class Eigen3ConvFastOp : public Eigen3OpBase<Eigen3ConvFastOp>
{
public:
    static constexpr int type = OP_CONV;

    explicit Eigen3ConvFastOp(struct node* ir)
        : Eigen3OpBase(ir), input_tensor_(nullptr), kernel_tensor_(nullptr), output_tensor_(nullptr), bias_tensor_(nullptr)
    {
        const auto input_tensor_idx = ir->input_tensors[0];
        const auto kernel_tensor_idx = ir->input_tensors[1];
        const auto output_tensor_idx = ir->output_tensors[0];
        input_tensor_ = get_ir_graph_tensor(ir->graph, input_tensor_idx);
        kernel_tensor_ = get_ir_graph_tensor(ir->graph, kernel_tensor_idx);
        output_tensor_ = get_ir_graph_tensor(ir->graph, output_tensor_idx);
        if (ir->input_num > 2)
        {
            const auto bias_tensor_idx = ir->input_tensors[2];
            bias_tensor_ = get_ir_graph_tensor(ir->graph, bias_tensor_idx);
        }

        params_ = reinterpret_cast<struct conv_param*>(ir->op.param_mem);
        kernel_size_ = params_->kernel_h * params_->kernel_w * params_->input_channel / params_->group;

        feat_map_size_ = output_tensor_->dims[2] * output_tensor_->dims[3];
        im2col_buf_.resize(kernel_size_ * feat_map_size_);
    }

    int Compute() override
    {
        const int output_c = output_tensor_->dims[1] / params_->group;
        const int input_c = params_->input_channel / params_->group;
        const int output_h = output_tensor_->dims[2];
        const int output_w = output_tensor_->dims[3];
        const auto activation = params_->activation;

        for (int batch = 0; batch < input_tensor_->dims[0]; ++batch)
            for (int group = 0; group < params_->group; ++group)
            {
                auto input = im2col_(batch, group);
                auto* kernel_data = (float*)kernel_tensor_->data + group * kernel_size_;
                auto* output_data = (float*)output_tensor_->data + batch * params_->group * output_c * output_h * output_w + group * output_c * output_h * output_w;
                auto* bias_data = (float*)bias_tensor_->data + group * output_c;

                MatrixWrap kernel(kernel_data, output_c, kernel_size_);
                MatrixWrap output(output_data, output_c, feat_map_size_);
                Eigen::Map<Eigen::VectorXf> bias(bias_data, output_c);
                if (activation >= 0)
                {
                    output = ((kernel * input).colwise() + bias).unaryExpr([activation](float total) {
                        if (total < 0 && activation != 1)
                        {
                            total = 0;
                        }
                        if (total > 1 && activation == 1)
                        {
                            total = 1;
                        }
                        if (total > 6 && activation == 6)
                        {
                            total = 6;
                        }
                        if (total < -1 && activation == 1)
                        {
                            total = -1;
                        }
                        return total;
                    });
                }
                else
                {
                    output = (kernel * input).colwise() + bias;
                }
#if 0
extern int ref_conv_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* kernel,
                         struct tensor* bias, struct conv_param* conv_param);
                const auto rows = std::min(4, (int)output.rows());
                const auto cols = std::min(4, (int)output.cols());
                std::cerr << "ir node: " << Node()->name << " conv mean: " << output.mean()
                          << ", block example:\n " << output.block(0, 0, rows, cols)
                          << ", ref mean: ";
                ref_conv_fp32(input_tensor_, output_tensor_, kernel_tensor_, bias_tensor_, params_);
                std::cerr << output.mean() << ", block example: \n"
                          << output.block(0, 0, rows, cols) << std::endl;
#endif
            }
        return 0;
    }

private:
    using MatrixWrap = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >;
    MatrixWrap im2col_(int const batch, int const group)
    {
        const auto in_dims = input_tensor_->dims;
        const auto out_dims = output_tensor_->dims;
        const auto input_c = in_dims[1] / params_->group;
        const auto input_h = in_dims[2];
        const auto input_w = in_dims[3];

        const auto output_c = out_dims[1] / params_->group;
        const auto output_h = out_dims[2];
        const auto output_w = out_dims[3];

        const auto* p = reinterpret_cast<float*>(input_tensor_->data);
        const auto* data = p + batch * params_->group * input_c * input_h * input_w
                           + group * input_c * input_h * input_w;

        auto* output_buf = reinterpret_cast<float*>(im2col_buf_.data());

        for (int c = 0; c < input_c; ++c)
        {
            const auto* src = data + c * input_h * input_w;
            auto* output_data = output_buf + c * params_->kernel_h * params_->kernel_w * feat_map_size_;
            for (int h = 0; h < output_h; ++h)
            {
                for (int w = 0; w < output_w; ++w)
                {
                    const int buf_w = h * output_w + w;
                    const int h_start = (h * params_->stride_h) - params_->pad_h0;
                    const int w_start = (w * params_->stride_w) - params_->pad_w0;

                    for (int kh = 0; kh < params_->kernel_h; ++kh)
                    {
                        for (int kw = 0; kw < params_->kernel_w; ++kw)
                        {
                            const int h_input = h_start + kh * params_->dilation_h;
                            const int w_input = w_start + kw * params_->dilation_w;
                            const int buf_h = kh * params_->kernel_w + kw;

                            if (h_input < 0 || w_input < 0)
                            {
                                output_data[buf_h * feat_map_size_ + buf_w] = .0f;
                                continue;
                            }
                            auto val = src[h_input * input_w + w_input];
                            output_data[buf_h * feat_map_size_ + buf_w] = val;
                        }
                    }
                }
            }
        }

        return MatrixWrap(output_buf, kernel_size_, feat_map_size_);
    }
    struct tensor* kernel_tensor_;
    struct tensor* input_tensor_;
    struct tensor* bias_tensor_;
    struct conv_param* params_;
    struct tensor* output_tensor_;
    int kernel_size_;
    int feat_map_size_;
    std::vector<float> im2col_buf_;
};

EIGEN3_REGISTER_OP(Eigen3ConvFastOp);
