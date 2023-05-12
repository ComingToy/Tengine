#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include "api/c_api.h"
#include "device/device.h"

struct eigen3_device
{
    struct device base;
};

DLLEXPORT extern int register_eigen3_device(void);
#ifdef __cplusplus
}
#endif
