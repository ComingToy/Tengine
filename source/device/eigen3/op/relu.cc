#include "op.hpp"
#include "registry.hpp"
#include <eigen3/Eigen/Eigen>
#ifdef __cplusplus
extern "C" {
#endif
#include "graph/graph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#ifdef __cplusplus
}
#endif

class Eigen3ReluOp : public Eigen3OpBase<Eigen3ReluOp>
{
public:
    constexpr static int type = OP_RELU;
    explicit Eigen3ReluOp(struct node* ir)
        : Eigen3OpBase(ir)
    {
    }

    int Compute() override
    {
        auto* ir = Node();
        auto* input_tensor = get_ir_graph_tensor(ir->graph, ir->input_tensors[0]);
        auto* output_tensor = get_ir_graph_tensor(ir->graph, ir->output_tensors[0]);

        auto vec = Eigen::Map<Eigen::VectorXf>(static_cast<float*>(output_tensor->data), output_tensor->elem_num);
        vec.unaryExpr([](float x) { return std::max(x, .0f); });
        return 0;
    }
};
class Eigen3Relu1Op : public Eigen3OpBase<Eigen3Relu1Op>
{
public:
    constexpr static int type = OP_RELU1;
    explicit Eigen3Relu1Op(struct node* ir)
        : Eigen3OpBase(ir)
    {
    }

    int Compute() override
    {
        auto* ir = Node();
        auto* input_tensor = get_ir_graph_tensor(ir->graph, ir->input_tensors[0]);
        auto* output_tensor = get_ir_graph_tensor(ir->graph, ir->output_tensors[0]);

        auto vec = Eigen::Map<Eigen::VectorXf>(static_cast<float*>(output_tensor->data), output_tensor->elem_num);

        vec.unaryExpr([](float x) { 
				float y = x; 
				if (y > 1.0f) y = 1.0f; 
				if (y < -1.0f) y = -1.0f;
				return y; });
        return 0;
    }
};

EIGEN3_REGISTER_OP(Eigen3ReluOp);
EIGEN3_REGISTER_OP(Eigen3Relu1Op);
