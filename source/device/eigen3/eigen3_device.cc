#include "eigen3_device.hpp"
#include "eigen3_graph.hpp"
#include "module/module.h"
#include "utility/vector.h"
#include "eigen3_graph.hpp"
#include "op/registry.hpp"
#include "eigen3_define.h"
#include <cstring>
#include <set>

#ifdef __cplusplus
extern "C" {
#endif
#include "api/c_api.h"
#include "operator/op.h"
#include "graph/subgraph.h"
#include "graph/graph.h"
#include "optimizer/split.h"
#include "executer/executer.h"
#include "utility/log.h"
#ifdef __cplusplus
}
#endif

static int eigen3_describe(struct device* device, struct vector* allowed_ops, struct vector* blocked_ops, struct vector* precision)
{
    (void)device;
    for (int op_type : Eigen3OpRegistry::Instance().SupportedOps())
    {
        push_vector_data(allowed_ops, &op_type);
    }

    for (int i = 0; i < OP_BUILTIN_LAST; ++i)
    {
        if (Eigen3OpRegistry::Instance().SupportedOps().count(i) == 0)
        {
            push_vector_data(blocked_ops, &i);
        }
    }

    int precision_var = TENGINE_DT_FP32;
    push_vector_data(precision, &precision_var);
    return 0;
}

static int eigen3_evaluation(struct device* device, struct subgraph* subgraph, struct vector* tensor, struct vector* node)
{
    if (NULL == device)
    {
        return -1;
    }

    (void)subgraph;
    (void)tensor;
    (void)node;
    return 0;
}

static int eigen3_allocate(struct device* device, struct subgraph* subgraph)
{
    subgraph->input_wait_count = 0;
    for (int i = 0; i < subgraph->input_num; ++i)
    {
        struct tensor* tensor = get_ir_graph_tensor(subgraph->graph, subgraph->input_tensor_list[i]);
        if (tensor->tensor_type == TENSOR_TYPE_VAR)
        {
            subgraph->input_wait_count++;
        }
    }

    return 0;
}

static int eigen3_release(struct device* device, struct subgraph* subgraph)
{
    if (NULL == device)
    {
        return -1;
    }

    (void)subgraph;
    return 0;
}

static int eigen3_dev_init(struct device* dev)
{
    (void)dev;
    return 0;
}

static int eigen3_dev_prerun(struct device* dev, struct subgraph* subgraph, void* options)
{
    //struct eigen3_option* opt = reinterpret_cast<struct eigen3_option*>(options);
    auto* devgraph = new Eigen3Graph(subgraph);
    subgraph->device_graph = devgraph;
    return devgraph->PreRun();
}

static int eigen3_dev_run(struct device* dev, struct subgraph* subgraph)
{
    auto* devgraph = reinterpret_cast<Eigen3Graph*>(subgraph->device_graph);
    return devgraph->Run();
}

static int eigen3_dev_postrun(struct device* dev, struct subgraph* subgraph)
{
    auto* devgraph = reinterpret_cast<Eigen3Graph*>(subgraph->device_graph);
    return devgraph->PostRun();
}

static int eigen3_dev_release_graph(struct device* dev, void* graph)
{
    if (graph)
    {
        auto* devgraph = reinterpret_cast<Eigen3Graph*>(graph);
        delete devgraph;
        return 0;
    }
    return -1;
}

static int eigen3_release(struct device* device)
{
    if (NULL == device)
    {
        return -1;
    }

    return 0;
}

static int eigen3_split_graph(struct graph* ir_graph)
{
    struct device* dev = ir_graph->attribute->context->device;

    if (0 != strcmp(EIGEN3_DEV_NAME, dev->name))
    {
        return -1;
    }

    struct vector* allowed_ops = create_vector(sizeof(int), nullptr);
    struct vector* blocked_ops = create_vector(sizeof(int), nullptr);
    struct vector* precision = create_vector(sizeof(int), nullptr);

    dev->allocator->describe(dev, allowed_ops, blocked_ops, precision);
    split_graph_node_to_sub_graph(ir_graph, allowed_ops, blocked_ops, precision);

    release_vector(allowed_ops);
    release_vector(blocked_ops);
    release_vector(precision);

    generate_sub_graph_io(ir_graph);
    add_sub_graph_to_ir_graph(ir_graph);

    for (int i = 0; i < get_vector_num(ir_graph->subgraph_list); ++i)
    {
        struct subgraph* subgraph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        subgraph->index = i;
        for (int j = 0; j < subgraph->node_num; ++j)
        {
            uint16_t node_id = subgraph->node_list[j];
            struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
            ir_node->subgraph_idx = subgraph->index;
        }
    }

    return 0;
}
#ifdef __cplusplus
extern "C" {
#endif

static struct interface eigen3_interface = {
    .init = eigen3_dev_init,
    .pre_run = eigen3_dev_prerun,
    .run = eigen3_dev_run,
    .post_run = eigen3_dev_postrun,
    .async_run = NULL,
    .async_wait = NULL,
    .release_graph = eigen3_dev_release_graph,
    .release_device = eigen3_release};

static struct allocator eigen3_allocator = {
    .describe = eigen3_describe,
    .evaluation = eigen3_evaluation,
    .allocate = eigen3_allocate,
    .release = eigen3_release,
};

static struct optimizer eigen3_optimizer = {
    .split_graph = eigen3_split_graph,
    .optimize_graph = NULL,
};

static struct eigen3_device eigen3_dev = {
    .base = {
        .name = EIGEN3_DEV_NAME,
        .interface = &eigen3_interface,
        .allocator = &eigen3_allocator,
        .optimizer = &eigen3_optimizer,
        .scheduler = NULL,
        .privacy = nullptr,
    },

};

int register_eigen3_device(void)
{
    int ret = register_device(&eigen3_dev.base);
    if (0 != ret)
    {
        TLOG_INFO("Tengine plugin %s registry failed.\n", eigen3_dev.base.name);
        return ret;
    }

    TLOG_INFO("Tengine plugin device %s is registered.\n", eigen3_dev.base.name);
    return 0;
}

int unregister_eigen3_device(void)
{
    int ret = unregister_device(&eigen3_dev.base);
    if (ret != 0)
    {
        TLOG_INFO("Tengine plugin %s unregister failed.\n", eigen3_dev.base.name);
        return ret;
    }

    TLOG_INFO("Tengine plugin device %s is unregistered", eigen3_dev.base.name);

    return ret;
}
#ifdef __cplusplus
}
#endif
