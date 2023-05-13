#pragma once
#include <vector>
#include "utility/sys_port.h"

class Eigen3Allocator
{
public:
    Eigen3Allocator() = default;
    Eigen3Allocator(Eigen3Allocator const&) = delete;
    Eigen3Allocator(Eigen3Allocator&&) = delete;
    Eigen3Allocator& operator=(Eigen3Allocator const&) = delete;
    Eigen3Allocator& operator=(Eigen3Allocator&&) = delete;

    void* allocate(size_t const size)
    {
        auto p = sys_malloc(size);
        if (!p) return p;
        blocks_.push_back(p);
        return p;
    }

    virtual ~Eigen3Allocator()
    {
        for (auto* p : blocks_)
        {
            sys_free(p);
        }
    }

private:
    std::vector<void*> blocks_;
};

class Eigen3Op
{
public:
    explicit Eigen3Op(struct node* ir)
        : ir_(ir)
    {
    }

    const struct node* Node() const
    {
        return ir_;
    }

    virtual int Compute() = 0;
    virtual ~Eigen3Op() = default;

private:
    struct node* ir_;
};

