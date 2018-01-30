#pragma once

#include "array.h"

namespace ndarray
{

template<size_t NewDepth, typename T, size_t D>
auto reshape(const array<T, D>& src, std::array<size_t, NewDepth> dims)
{
    array<T, NewDepth> ret(src._get_vector(), dims);
    ret._check_size();
    return ret;
}

template<size_t NewDepth, typename T, size_t D>
auto reshape(array<T, D>&& src, std::array<size_t, NewDepth> dims)
{
    array<T, NewDepth> ret(std::move(src)._get_vector(), dims);
    ret._check_size();
    return ret;
}

template<typename T, size_t D>
auto flatten(const array<T, D>& src)
{
    size_t size = src.total_size();
    return array<T, 1>(src._get_vector(), {size});
}

template<typename T, size_t D>
auto flatten(array<T, D>&& src)
{
    size_t size = src.total_size();
    return array<T, 1>(std::move(src)._get_vector(), {size});
}

}

