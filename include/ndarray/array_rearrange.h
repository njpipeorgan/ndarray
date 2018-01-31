#pragma once

#include <algorithm>

#include "array.h"
#include "array_interface.h"

namespace ndarray
{

template<size_t NewDepth, typename Array>
inline auto _reshape_impl(Array&& src, std::array<size_t, NewDepth> dims, std::integral_constant<_access_type, _access_type::vector>)
{
    using elem_t = typename remove_cvref_t<Array>::_elem_t;
    return array<elem_t, NewDepth>(get_vector(std::forward<Array>(src)), dims);
}
template<size_t NewDepth, typename Array>
inline auto _reshape_impl(Array&& src, std::array<size_t, NewDepth> dims, std::integral_constant<_access_type, _access_type::iterator>)
{
    using elem_t = typename remove_cvref_t<Array>::_elem_t;
    const size_t src_size  = src.total_size();
    std::vector<elem_t> data(src_size);
    src.copy_to(data.data(), src_size);
    return array<elem_t, NewDepth>(std::move(data), dims);
}
template<size_t NewDepth, typename Array>
inline auto _reshape_impl(Array&& src, std::array<size_t, NewDepth> dims, std::integral_constant<_access_type, _access_type::traverse>)
{
    return _reshape_impl<NewDepth>(
        std::forward<Array>(src), dims, std::integral_constant<_access_type, _access_type::iterator>{});
}

// reshape an array to a new set of dimensions
template<size_t NewDepth, typename Array>
inline auto reshape(Array&& src, std::array<size_t, NewDepth> dims)
{
    auto ret = _reshape_impl<NewDepth>(
        std::forward<Array>(src), dims, std::integral_constant<_access_type, _identify_access_type_v<Array>>{});
    ret._check_size();
    return ret;
}

// flatten an array to 1-dimension
template<typename Array>
inline auto flatten(Array&& src)
{
    auto ret = _reshape_impl<1>(
        std::forward<Array>(src), {src.total_size()}, std::integral_constant<_access_type, _identify_access_type_v<Array>>{});
    NDARRAY_ASSERT(ret._check_size()); // no necessary
    return ret;
}

// divide the first dimension of array into parts of specific length
template<size_t PartDepth, typename Array>
inline auto partition(Array&& src, size_t dim_1)
{
    size_t dim_0     = src.dimension<0>();
    size_t new_dim_0 = dim_0 / dim_1;
    size_t remainder = dim_0 % dim_1;
    assert(remainder == 0);
    
    constexpr size_t Depth    = remove_cvref_t<Array>::_depth_v;
    constexpr size_t NewDepth = Depth + 1;

    std::array<size_t, Depth> dims = src.dimensions();
    std::array<size_t, NewDepth> new_dims{new_dim_0, dim_1};
    for (size_t i = 2; i < NewDepth; ++i)
        new_dims[i] = dims[i - 1];

    auto ret = _reshape_impl<NewDepth>(
        std::forward<Array>(src), new_dims, std::integral_constant<_access_type, _identify_access_type_v<Array>>{});
    NDARRAY_ASSERT(ret._check_size()); // no necessary
    return ret;
}


}

