#pragma once
#include <cassert>
#include <functional>
#include <map>
#include "op.hpp"
#include <set>
#include <vector>

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
    using CreatorType = std::pair<std::function<int(struct node*)>, std::function<Eigen3Op*(struct node*)> >;

    static Eigen3OpRegistry& Instance()
    {
        static Eigen3OpRegistry registry;
        return registry;
    }

    void Register(int const op, Eigen3OpRegistry::CreatorType&& creator)
    {
        creators_[op].emplace_back(std::move(creator));
        op_types_.insert(op);
    }

    Eigen3Op* CreateOp(struct node* ir)
    {
        auto pos = creators_.find(ir->op.type);
        if (pos != creators_.cend())
        {
            auto const& cands = pos->second;
            auto it = cands.cend();
            int max_score = -1;
            for (auto pos = cands.cbegin(); pos < cands.cend(); ++pos)
            {
                auto score = pos->first(ir);
                if (max_score < score)
                {
                    it = pos;
                    max_score = score;
                }
            }

            assert(it != cands.cend());

            return it->second(ir);
        }
        return nullptr;
    }

    const std::set<int>& SupportedOps()
    {
        return op_types_;
    }

private:
    Eigen3OpRegistry() = default;
    ~Eigen3OpRegistry() = default;
    std::map<int, std::vector<CreatorType> > creators_;
    std::set<int> op_types_ = {OP_CONST};
};

template<typename T>
class Eigen3OpRegisterHelper
{
public:
    Eigen3OpRegisterHelper()
    {
        Eigen3OpRegistry::Instance().Register(T::type, {T::Score, [](struct node* ir) { return new T(ir); }});
    };
};

#define EIGEN3_REGISTER_OP(__Op) \
    static Eigen3OpRegisterHelper<__Op> __eigen3_op_register_helper_##__Op;
