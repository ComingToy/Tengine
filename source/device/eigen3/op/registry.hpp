#pragma once
#include <functional>
#include <map>
#include "op.hpp"
#include <set>

#ifdef __cplusplus
extern "C" {
#endif
#include "graph/node.h"
#ifdef __cplusplus
}
#endif

class Eigen3OpRegistry
{
public:
    static Eigen3OpRegistry& Instance()
    {
        static Eigen3OpRegistry registry;
        return registry;
    }

    void Register(int const op, std::function<Eigen3Op*(struct node*)>& creator)
    {
        creators_[op] = creator;
        op_types_.insert(op);
    }

    Eigen3Op* CreateOp(struct node* ir)
    {
        auto pos = creators_.find(ir->op.type);
        if (pos != creators_.cend())
        {
            return (pos->second)(ir);
        }
        else
        {
            return nullptr;
        }
    }

    const std::set<int>& SupportedOps()
    {
        return op_types_;
    }

private:
    Eigen3OpRegistry() = default;
    ~Eigen3OpRegistry() = default;
    std::map<int, std::function<Eigen3Op*(struct node*)> > creators_;
    std::set<int> op_types_;
};

template<typename T>
class Eigen3OpRegisterHelper
{
public:
    Eigen3OpRegisterHelper()
    {
        Eigen3OpRegistry::Instance().Register([](struct node* ir) { return T(ir); });
    };
};

#define EIGEN3_REGISTER_OP(__Op) \
    static Eigen3OpRegisterHelper<__Op> __eigen3_op_register_helper_##__Op;
