#pragma once

#include "traits.h"
#include "array.h"
#include "array_view.h"
#include "range_view.h"

namespace ndarray
{

template<typename TFirst, typename TLast>
constexpr inline auto vrange(TFirst first, TLast last)
{
    return make_range_view(first, last);
}

template<typename TFirst, typename TLast, typename TStep>
constexpr inline auto vrange(TFirst first, TLast last, TStep step)
{
    return make_range_view(first, last, step);
}

template<typename TLast>
constexpr inline auto vrange(TLast last)
{
    return make_range_view(int(0), last);
}

template<typename TFirst, typename TLast>
inline auto range(TFirst first, TLast last)
{
    return make_array(vrange(first, last));
}

template<typename TFirst, typename TLast, typename TStep>
inline auto range(TFirst first, TLast last, TStep step)
{
    return make_array(vrange(first, last, step));
}

template<typename TLast>
inline auto range(TLast last)
{
    return make_array(vrange(last));
}


template<size_t I, typename T, typename Function, typename ArrayTuple>
inline void _table_impl(T*& data_ptr, Function fn, const ArrayTuple& arrays)
{
    const auto& arr_i = std::get<I>(arrays);
    auto begin = element_begin(arr_i);
    auto end   = element_end(arr_i);

    if constexpr (I + 1 < std::tuple_size_v<ArrayTuple>)
        for (; begin != end; ++begin)
            _table_impl<I + 1>(data_ptr, [=](auto&&... args) { return fn(*begin, std::forward<decltype(args)>(args)...); }, arrays);
    else
        for (; begin != end; ++begin)
        {
            static_assert(std::is_same_v<decltype(fn(*begin)), T>, "there should not be conversion");
            *data_ptr = fn(*begin);
            ++data_ptr;
        }
}

template<typename Function, typename... Arrays>
inline array<std::invoke_result_t<Function, array_or_range_elem_of_t<Arrays>...>, sizeof...(Arrays)> 
    table(Function fn, Arrays&&... arrays)
{
    using result_t = std::invoke_result_t<Function, array_or_range_elem_of_t<Arrays>...>;
    auto array_tuple = std::make_tuple(make_range_if_arithmetic(std::forward<Arrays>(arrays))...);
    array<result_t, sizeof...(Arrays)> ret(size_of_array_tuple(array_tuple));
    result_t* data_ptr = ret.data();
    _table_impl<0, result_t, Function>(data_ptr, fn, array_tuple);
    return ret;
}

template<typename Value, typename... Ints>
inline auto const_table(Value value, Ints... ints)
{
    constexpr size_t depth_v = sizeof...(Ints);
    using elem_t = decltype(value);
    std::array<size_t, depth_v> dims = {ints...};
    std::vector<elem_t> data(size_t((ints * ... * size_t(1))), value);
    return array<elem_t, depth_v>{std::move(data), dims};
}


}
