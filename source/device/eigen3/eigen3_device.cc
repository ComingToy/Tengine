#include "eigen3_device.hpp"
#include "eigen3_graph.hpp"
#include "utility/vector.h"
#include "eigen3_graph.hpp"
#include "op/registry.hpp"
#include "eigen3_define.h"
#include <set>

#ifdef __cplusplus
extern "C" {
#endif
#include "api/c_api.h"
#include "operator/op.h"
#include "graph/subgraph.h"
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

