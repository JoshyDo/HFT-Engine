#pragma once
#include <cstdint>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

struct EdgeDOD {
    double   bid;
    double   ask;
    double   log_ask;
    uint32_t to;
};

struct alignas(64) NodeDOD {
    uint32_t id;
    uint32_t startEdgeIndex;
    uint32_t edgeCount;
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif