#pragma once

#include "array.h"

namespace ndarray
{

template<typename TFirst, typename TLast>
auto range(TFirst first, TLast last)
{
    auto   diff = static_cast<ptrdiff_t>(last - first);
    size_t size = diff > 0 ? size_t(diff) : 0;
    std::vector<TFirst> data(size);
    for (size_t i = 0; i < size; ++i)
        data[i] = static_cast<TFirst>(first + i);
    return array<TFirst, 1>(std::move(data), {size});
}

template<typename TFirst, typename TLast, typename TStep>
auto range(TFirst first, TLast last, TStep step)
{
    auto   diff = static_cast<ptrdiff_t>((last - first) / step);
    size_t size = diff > 0 ? size_t(diff) : 0;
    std::vector<TFirst> data(size);
    for (size_t i = 0; i < size; ++i)
        data[i] = static_cast<TFirst>(first + i * step);
    return array<TFirst, 1>(std::move(data), {size});
}

template<typename TLast>
auto range(TLast last)
{
    return range(TLast(0), last);
}

}
