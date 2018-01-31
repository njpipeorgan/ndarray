#pragma once

#include "array.h"

namespace ndarray
{

template<typename TFirst, typename TLast>
inline auto range(TFirst first, TLast last)
{
    auto   diff = static_cast<ptrdiff_t>(last - first);
    size_t size = diff > 0 ? size_t(diff) : 0;
    std::vector<TFirst> data(size);
    for (size_t i = 0; i < size; ++i)
        data[i] = static_cast<TFirst>(first + i);
    return array<TFirst, 1>(std::move(data), {size});
}

template<typename TFirst, typename TLast, typename TStep>
inline auto range(TFirst first, TLast last, TStep step)
{
    auto   diff = static_cast<ptrdiff_t>((last - first) / step);
    size_t size = diff > 0 ? size_t(diff) : 0;
    std::vector<TFirst> data(size);
    for (size_t i = 0; i < size; ++i)
        data[i] = static_cast<TFirst>(first + i * step);
    return array<TFirst, 1>(std::move(data), {size});
}

template<typename TLast>
inline auto range(TLast last)
{
    return range(TLast(0), last);
}

template<size_t I, typename T, typename Function, typename ArrayTuple>
inline void _table_impl(T*& data_ptr, Function fn, const ArrayTuple& arrs)
{
    const auto& arr_i = std::get<I>(arrs);
    auto begin = element_begin(arr_i);
    auto end   = element_end(arr_i);

    if constexpr (I + 1 < std::tuple_size_v<ArrayTuple>)
        for (; begin != end; ++begin)
            _table_impl<I + 1>(data_ptr, [=](auto&&... args) { return fn(*begin, std::forward<decltype(args)>(args)...); }, arrs);
    else
        for (; begin != end; ++begin)
        {
            *data_ptr = fn(*begin);
            ++data_ptr;
        }
}

template<typename Function, typename... Arrays>
inline auto table(Function fn, Arrays&&... arrs)
{
    using result_t = std::invoke_result_t<Function, typename remove_cvref_t<Arrays>::_elem_t...>;
    array<result_t, sizeof...(Arrays)> ret(size_of_arrays(arrs...));
    result_t* data_ptr = ret.data();
    _table_impl<0, result_t, Function>(data_ptr, fn, std::forward_as_tuple(arrs...));
    return ret;
}


}
