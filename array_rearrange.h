#pragma once

#include <algorithm>

#include "array.h"
#include "array_interface.h"

namespace ndarray
{

template<size_t NewDepth, typename T, size_t Depth>
inline auto _reshape_impl(const array<T, Depth>& src, std::array<size_t, NewDepth> dims)
{
    return array<T, NewDepth>(get_vector(src), dims);
}
template<size_t NewDepth, typename T, size_t Depth>
inline auto _reshape_impl(array<T, Depth>&& src, std::array<size_t, NewDepth> dims)
{
    return array<T, NewDepth>(get_vector(std::move(src)), dims);
}
template<size_t NewDepth, typename View>
inline auto _reshape_impl(View&& src, std::array<size_t, NewDepth> dims)
{
    using elem_t = array_elem_of_t<View>;
    const size_t src_size  = src.size();
    std::vector<elem_t> data(src_size);
    src.copy_to(data.data(), src_size);
    return array<elem_t, NewDepth>(std::move(data), dims);
}

// reshape an array to a new set of dimensions
template<size_t NewDepth, typename Array>
inline auto reshape(Array&& src, std::array<size_t, NewDepth> dims)
{
    array<array_elem_of_t<Array>, NewDepth> ret = 
        _reshape_impl<NewDepth>(std::forward<decltype(src)>(src), dims);
    ret._check_size();
    return ret;
}

// flatten an array to 1-dimension
template<typename Array>
inline auto flatten(Array&& src)
{
    array<array_elem_of_t<Array>, 1> ret = 
        _reshape_impl<1>(std::forward<Array>(src), {src.size()});
    NDARRAY_ASSERT(ret._check_size()); // no necessary
    return ret;
}

// divide the first N dimensions of array into parts of specific length
template<size_t PartDepth, typename Array>
inline auto partition(Array&& src, std::array<size_t, PartDepth> part_dims)
{
    constexpr size_t part_depth_v = PartDepth;
    constexpr size_t depth_v      = array_depth_of_v<Array>;
    static_assert(part_depth_v <= depth_v, "too many part dimensions");
    constexpr size_t new_depth_v  = depth_v + PartDepth;

    std::array<size_t, depth_v>     dims = src.dimensions();
    std::array<size_t, new_depth_v> new_dims;
    for (size_t i = 0; i < part_depth_v; ++i)
    {
        size_t dim_i      = dims[i];
        size_t part_dim_i = part_dims[i];
        size_t extra_dim  = dim_i / part_dim_i;
        size_t remainder  = dim_i % part_dim_i;
        assert(remainder == 0); // dim_i should be divisible by part_dim_i;
        new_dims[i * 2u]     = extra_dim;
        new_dims[i * 2u + 1] = part_dim_i;
    }
    for (size_t i = part_depth_v; i < depth_v; ++i)
    { // copy remaining dims
        new_dims[i + part_depth_v] = dims[i];
    }

    array<array_elem_of_t<Array>, new_depth_v> ret = 
        _reshape_impl<new_depth_v>(std::forward<Array>(src), new_dims);
    NDARRAY_ASSERT(ret._check_size()); // not necessary
    return ret;
}

// divide the first dimension of array into parts of specific length
template<typename Array>
inline auto partition(Array&& src, size_t part_dim)
{
    return partition<1>(std::forward<Array>(src), {part_dim});
}


template<typename ResultType, typename DataArray, typename IndexArray, size_t... I>
inline ResultType _element_extract_impl(const DataArray& data, const IndexArray& index, size_t pos_0, std::index_sequence<I...>)
{
    return data.at(index.at(pos_0, I)...);
}

// extract elements at index from array
template<typename DataArray, typename IndexArray>
inline auto element_extract(const DataArray& data, const IndexArray& index)
{
    using elem_t  = array_elem_of_t<DataArray>;
    constexpr size_t data_depth_v  = array_depth_of_v<DataArray>;
    constexpr size_t index_depth_v = array_depth_of_v<IndexArray>;

    static_assert(data_depth_v == 1 ? index_depth_v <= 2 : index_depth_v == 2);
    if constexpr (data_depth_v > 1 || index_depth_v == 2)
        assert(index.dimension<1>() == data_depth_v);

    size_t size = index.dimension<0>();

    std::vector<elem_t> extracted(size);

    for (size_t i = 0; i < size; ++i)
    {
        if constexpr (index_depth_v == 1)
            extracted[i] = data.at(index.at(i));
        else
            extracted[i] = _element_extract_impl<elem_t>(
                data, index, i, std::make_index_sequence<index_depth_v>{});
    }

    return make_array(extracted);
}

// extract subarray or elements from array
template<size_t Depth = size_t(-1), typename DataArray, typename IndexArray>
inline auto extract(const DataArray& data, const IndexArray& index)
{
    constexpr size_t data_depth_v  = array_depth_of_v<DataArray>;
    constexpr size_t index_depth_v = array_depth_of_v<IndexArray>;
    static_assert(data_depth_v == 1 ? index_depth_v <= 2 : index_depth_v == 2);

    if constexpr (Depth == size_t(-1))
        return element_extract(std::forward<DataArray>(data), index);
    else
        return;
}


}

