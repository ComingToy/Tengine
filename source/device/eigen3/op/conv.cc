#include "op.hpp"
#include "op/registry.hpp"
#define EIGEN_USE_THREADS

#include <algorithm>
#include <eigen3/eigen3_device.hpp>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h>
#include <eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMap.h>
#include <eigen3/unsupported/Eigen/CXX11/src/util/EmulateArray.h>
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

class Eigen3ConvOp : public Eigen3OpBase<Eigen3ConvOp>
{
public:
    static constexpr int type = OP_CONV;
    explicit Eigen3ConvOp(struct node* ir)
        : Eigen3OpBase(ir)
    {
    }

    int Compute() override
    {
        const auto input_tensor_idx = Node()->input_tensors[0];
        const auto kernel_tensor_idx = Node()->input_tensors[1];
        const auto output_tensor_idx = Node()->output_tensors[0];

        auto* input_tensor = get_ir_graph_tensor(Node()->graph, input_tensor_idx);
        auto* kernel_tensor = get_ir_graph_tensor(Node()->graph, kernel_tensor_idx);
        auto* output_tensor = get_ir_graph_tensor(Node()->graph, output_tensor_idx);

        struct conv_param* params = reinterpret_cast<struct conv_param*>(Node()->op.param_mem);

        const int batch = input_tensor->dims[0];
        const int group = params->group;
        const int input_c = params->input_channel / group;
        const int input_h = input_tensor->dims[2];
        const int input_w = input_tensor->dims[3];

        const int output_c = output_tensor->dims[1] / group;
        const int output_h = output_tensor->dims[2];
        const int output_w = output_tensor->dims[3];

        params->kernel_h = std::max(1, params->kernel_h);
        params->kernel_w = std::max(1, params->kernel_w);

        const float* bias = NULL;
        struct tensor* bias_tensor = NULL;
        if (Node()->input_num > 2)
        {
            bias_tensor = get_ir_graph_tensor(Node()->graph, Node()->input_tensors[2]);

            bias = (float*)bias_tensor->data;
        }

        Eigen::array<Eigen::Index, 5> in_dims = {batch, group, input_c, input_h, input_w};
        Eigen::array<Eigen::Index, 5> out_dims = {batch, group, output_c, output_h, output_w};
        Eigen::array<Eigen::Index, 5> kernel_dims = {group, output_c, input_c, params->kernel_h, params->kernel_w};

        Eigen::TensorMap<Eigen::Tensor<float, 5, Eigen::RowMajor> > input_tensor_map((float*)input_tensor->data, in_dims);
        Eigen::TensorMap<Eigen::Tensor<float, 5, Eigen::RowMajor>> output_tensor_map((float*)output_tensor->data, out_dims);
        Eigen::TensorMap<Eigen::Tensor<float, 5, Eigen::RowMajor> > kernel_tensor_map((float*)kernel_tensor->data, kernel_dims);

        Eigen::ThreadPool pool(16 /* number of threads in pool */);
        Eigen::ThreadPoolDevice cpu(&pool, 16);

        for (int b = 0; b < batch; ++b)
        {
            for (int g = 0; g < group; ++g)
            {
                for (int c = 0; c < output_c; ++c)
                {
                    for (int h = 0; h < output_h; ++h)
                    {
                        for (int w = 0; w < output_w; ++w)
                        {
                            const int h_start = (h * params->stride_h) - params->pad_h0;
                            const int w_start = (w * params->stride_w) - params->pad_w0;

                            int h_offset = h_start, w_offset = w_start, kh_offset = 0, kw_offset = 0;
                            int h_extent = params->kernel_h * params->dilation_h, w_extent = params->kernel_w * params->dilation_w;

                            if (h_start < 0)
                            {
                                h_offset = 0;
                                kh_offset = -h_start;
                                h_extent = h_extent + h_start;
                            }

                            if (w_start < 0)
                            {
                                w_offset = 0;
                                kw_offset = -w_start;
                                w_extent = w_extent + w_start;
                            }


                            Eigen::array<Eigen::Index, 5> input_offsets = {b, g, 0, h_offset, w_offset};
                            Eigen::array<Eigen::Index, 5> input_ends = {b + 1, g + 1, input_c, std::min(h_offset + h_extent, input_h), std::min(w_offset + w_extent, input_w)};
                            Eigen::array<Eigen::Index, 5> input_strides = {1, 1, 1, params->dilation_h, params->dilation_w};

                            auto input_slice = input_tensor_map.stridedSlice(input_offsets, input_ends, input_strides);

                            const int kh_extent = input_ends[3] - input_offsets[3];
                            const int kw_extent = input_ends[4] - input_offsets[4];

                            Eigen::array<Eigen::Index, 5> kernel_offsets = {g, c, 0, kh_offset, kw_offset};
                            Eigen::array<Eigen::Index, 5> kernel_extents = {1, 1, input_c, kh_extent, kw_extent};
                            Eigen::Tensor<float, 5, Eigen::RowMajor> kernel_slice(kernel_extents);
                            kernel_slice.device(cpu) = kernel_tensor_map.slice(kernel_offsets, kernel_extents);

                            // std::cerr << "ir node " << Node()->name << " kernel slice: " << kernel_slice << std::endl;

                            auto mul = (input_slice * kernel_slice);
                            Eigen::Tensor<float, 0, Eigen::RowMajor> result;

                            result.device(cpu) = (input_slice * kernel_slice).sum();

                            float total = result();
                            if (bias) total += bias[g * output_c + c];

                            if (params->activation >= 0)
                            {
                                if (total < 0 && params->activation != 1)
                                {
                                    total = 0;
                                }
                                if (total > 1 && params->activation == 1)
                                {
                                    total = 1;
                                }
                                if (total > 6 && params->activation == 6)
                                {
                                    total = 6;
                                }
                                if (total < -1 && params->activation == 1)
                                {
                                    total = -1;
                                }
                            }

                            output_tensor_map(b, g, c, h, w) = total;
                        }
                    }
                }
            }
        }

        return 0;
    }
};

// EIGEN3_REGISTER_OP(Eigen3ConvOp);
