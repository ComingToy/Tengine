#pragma once

#include <memory>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "graph/node.h"
#include "operator/op.h"
#include "graph/tensor.h"
#ifdef __cplusplus
}
#endif

#include <set>
#include "op/op.hpp"
#include "op/registry.hpp"

class Eigen3Graph
{
public:
    explicit Eigen3Graph(struct subgraph* subgraph)
        : subgraph_(subgraph)
    {
        for (auto i = 0; i < subgraph_->node_num; ++i)
        {
            auto ir = get_ir_graph_node(subgraph_->graph, subgraph_->node_list[i]);
            auto op = Eigen3OpRegistry::Instance().CreateOp(ir);
            nodes_.emplace_back(op);
        }
        ResetAllocator();
    }

    int PreRun()
    {
        for (auto i = 0; i < subgraph_->node_num; ++i)
        {
            auto ir = get_ir_graph_node(subgraph_->graph, subgraph_->node_list[i]);
            for (auto k = 0; k < ir->output_num; ++k)
            {
                auto tensor = get_ir_graph_tensor(subgraph_->graph, ir->output_tensors[k]);
                if (!tensor->data)
                {
                    tensor->data = allocator_->allocate(tensor->elem_size * tensor->elem_num);
                }
            }

            for (auto k = 0; k < ir->input_num; ++k)
            {
                auto tensor = get_ir_graph_tensor(subgraph_->graph, ir->input_tensors[k]);
                if (!tensor->data)
                {
                    tensor->data = allocator_->allocate(tensor->elem_size * tensor->elem_num);
                }
            }
        }
        return 0;
    }

    int Run()
    {
        for (auto& op : nodes_)
        {
            op->Compute();
        }
        return 0;
    }

private:
    struct subgraph* subgraph_;
    std::vector<std::unique_ptr<Eigen3Op> > nodes_;
    std::unique_ptr<Eigen3Allocator> allocator_;
    void ResetAllocator()
    {
        allocator_.reset(new Eigen3Allocator);
    }
};
