#pragma once

#include <cassert>
#include <algorithm>

#ifdef _DEBUG
#define ENABLE_NDARRAY_DEBUG
#endif

#ifdef ENABLE_NDARRAY_DEBUG
#define ENABLE_NDARRAY_ASSERT
#define ENABLE_NDARRAY_CHECK_BOUND
#define NDARRAY_DEBUG(x) (x)
#else
#define NDARRAY_DEBUG(x) (0)
#endif

#ifdef ENABLE_NDARRAY_ASSERT
#define NDARRAY_ASSERT(x) assert(x)
#else
#define NDARRAY_ASSERT(x) (0)
#endif

#ifdef ENABLE_NDARRAY_CHECK_BOUND
#define NDARRAY_CHECK_BOUND_SCALAR(i, size) assert(_check_bound_scalar(i, size))
#define NDARRAY_CHECK_BOUND_VECTOR(v, size) assert(_check_bound_vector(v, size))
#else 
#define NDARRAY_CHECK_BOUND_SCALAR(i, size) 0
#define NDARRAY_CHECK_BOUND_VECTOR(v, size) 0
#endif 

namespace ndarray
{

template<typename T>
struct remove_cvref :
    std::remove_cv<std::remove_reference_t<T>> {};
template<typename T>
using remove_cvref_t = typename remove_cvref<T>::type;


// used to give template argument dependent false for static_assert
template<typename... T>
struct _always_false
{
    static constexpr bool value = false;
};
template<typename... T>
constexpr bool _always_false_v = _always_false<T...>::value;


// constructed from arbitrary arguments and gives nothing
struct empty_struct
{
    template<typename... T>
    constexpr empty_struct(T...) noexcept {}
};

// calculate the maximum of two template arguments
template<size_t I, size_t J>
struct _mp_max
{
    static constexpr size_t value = I > J ? I : J;
};
template<size_t I, size_t J>
constexpr size_t _mp_max_v = _mp_max<I, J>::value;

// calculate the minimum of two template arguments
template<size_t I, size_t J>
struct _mp_min
{
    static constexpr size_t value = I < J ? I : J;
};
template<size_t I, size_t J>
constexpr size_t _mp_min_v = _mp_min<I, J>::value;


// tuple of T repeated n times
template<size_t N, typename T, typename Tuple = std::tuple<>>
struct repeat_tuple;
template<size_t N, typename T, typename... Ts>
struct repeat_tuple<N, T, std::tuple<Ts...>>
{
    using type = typename repeat_tuple<N - 1, T, std::tuple<Ts..., T>>::type;
};
template<typename T, typename... Ts>
struct repeat_tuple<0, T, std::tuple<Ts...>>
{
    using type = std::tuple<Ts...>;
};
template<size_t N, typename T>
using repeat_tuple_t = typename repeat_tuple<N, T>::type;

template<size_t N, typename Dropped, typename Taken = std::tuple<>>
struct takedrop_tuple_n_impl;
template<size_t N, typename Drop1, typename... Drops, typename... Takes>
struct takedrop_tuple_n_impl<N, std::tuple<Drop1, Drops...>, std::tuple<Takes...>> :
    takedrop_tuple_n_impl<N - 1, std::tuple<Drops...>, std::tuple<Takes..., Drop1>> {};
template<typename Drop1, typename... Drops, typename... Takes>
struct takedrop_tuple_n_impl<0, std::tuple<Drop1, Drops...>, std::tuple<Takes...>>
{
    using drop_t = std::tuple<Drop1, Drops...>;
    using take_t = std::tuple<Takes...>;
};
template<size_t N, typename... Takes>
struct takedrop_tuple_n_impl<N, std::tuple<>, std::tuple<Takes...>>
{
    static_assert(N == 0, "not enough tuple elements.");
    using drop_t = std::tuple<>;
    using take_t = std::tuple<Takes...>;
};
template<size_t N, typename Tuple>
struct drop_tuple_n
{
    using type = typename takedrop_tuple_n_impl<N, Tuple>::drop_t;
};
template<size_t N, typename Tuple>
struct take_tuple_n
{
    using type = typename takedrop_tuple_n_impl<N, Tuple>::take_t;
};
template<size_t N, typename Tuple>
using drop_tuple_n_t = typename drop_tuple_n<N, Tuple>::type;
template<size_t N, typename Tuple>
using take_tuple_n_t = typename take_tuple_n<N, Tuple>::type;

template<size_t R, typename Pad, typename Tuple>
struct pad_right_tuple_impl;
template<size_t R, typename Pad, typename... Ts>
struct pad_right_tuple_impl<R, Pad, std::tuple<Ts...>> :
    pad_right_tuple_impl<R - 1, Pad, std::tuple<Ts..., Pad>> {};
template<typename Pad, typename... Ts>
struct pad_right_tuple_impl<0, Pad, std::tuple<Ts...>>
{
    using type = std::tuple<Ts...>;
};
template<size_t N, typename Pad, typename Tuple>
struct pad_right_tuple :
    pad_right_tuple_impl<_mp_max_v<N, std::tuple_size_v<Tuple>> - std::tuple_size_v<Tuple>, Pad, Tuple> {};
template<size_t N, typename Pad, typename Tuple>
using pad_right_tuple_t = typename pad_right_tuple<N, Pad, Tuple>::type;


template<typename Return, typename X, typename Y>
constexpr inline Return _add_if_negative(X x, Y y)
{
    if constexpr (std::is_unsigned_v<X>)
        return Return(x);
    else
        return (x >= X(0)) ? Return(x) : Return(x) + Return(y);
}

template<typename Return, typename X, typename Y>
constexpr inline Return _add_if_non_positive(X x, Y y)
{
    return (x > X(0)) ? Return(x) : Return(x) + Return(y);
}


template<typename Index, typename Size>
constexpr inline bool _check_bound_scalar(Index index, Size size)
{
    static_assert(std::is_integral_v<Index> && std::is_same_v<Size, size_t>);
    return 0 <= index && index < size;
}

template<typename Indices, typename Size>
constexpr inline bool _check_bound_vector(Indices indices, Size size)
{
    return std::all_of(indices.begin(), indices.end(),
                       [=](auto i) { return _check_bound_scalar(i, size); });
}

template<size_t I, typename ArrayTuple>
inline void _size_of_array_tuple_impl(size_t* sizes, const ArrayTuple& arrays)
{
    sizes[I] = std::get<I>(arrays).size();
    if constexpr (I + 1 < std::tuple_size_v<ArrayTuple>)
        _size_of_array_tuple_impl<I + 1>(sizes, arrays);
}

template<typename ArrayTuple>
inline std::array<size_t, std::tuple_size_v<ArrayTuple>> size_of_array_tuple(const ArrayTuple& arrays)
{
    std::array<size_t, std::tuple_size_v<ArrayTuple>> sizes;
    _size_of_array_tuple_impl<0>(sizes.data(), arrays);
    return sizes;
}

template<typename SingleTuple, size_t... I>
auto _tuple_repeat_impl(SingleTuple val, std::index_sequence<I...>)
{
    return std::make_tuple(std::get<0u * I>(val)...);
}
template<size_t N, typename T>
repeat_tuple_t<N, T> tuple_repeat(T val)
{
    return _tuple_repeat_impl(std::make_tuple(val), std::make_index_sequence<N>{});
}

template<typename Tuple1, typename Tuple2, size_t... I1, size_t... I2>
auto _tuple_catenate_impl(Tuple1&& t1, Tuple2&& t2, std::index_sequence<I1...>, std::index_sequence<I2...>)
{
    return std::make_tuple(std::get<I1>(std::forward<decltype(t1)>(t1))...,
                           std::get<I2>(std::forward<decltype(t2)>(t2))...);
}
template<typename Tuple1, typename Tuple2>
auto tuple_catenate(Tuple1&& t1, Tuple2&& t2)
{
    return _tuple_catenate_impl(std::forward<decltype(t1)>(t1), std::forward<decltype(t2)>(t2),
                                std::make_index_sequence<std::tuple_size_v<remove_cvref_t<Tuple1>>>{},
                                std::make_index_sequence<std::tuple_size_v<remove_cvref_t<Tuple2>>>{});
}

template<size_t N, typename Pad, typename Tuple>
pad_right_tuple_t<N, Pad, remove_cvref_t<Tuple>> tuple_pad_right(Pad pad, Tuple&& tuple)
{
    constexpr size_t size_v     = std::tuple_size_v<remove_cvref_t<Tuple>>;
    constexpr size_t pad_size_v = _mp_max_v<N, size_v> -size_v;
    return tuple_catenate(std::forward<decltype(tuple)>(tuple), tuple_repeat<pad_size_v>(pad));
}

template<size_t IndexOffset = 0, typename Tuple, size_t... I>
auto tuple_takedrop_impl(Tuple&& tuple, std::index_sequence<I...>)
{
    return std::make_tuple(std::get<I + IndexOffset>(std::forward<decltype(tuple)>(tuple))...);
}

template<size_t N, typename Tuple>
take_tuple_n_t<N, remove_cvref_t<Tuple>> tuple_take(Tuple&& tuple)
{
    return tuple_takedrop_impl<0>(std::forward<decltype(tuple)>(tuple),
                                  std::make_index_sequence<N>{});
}
template<size_t N, typename Tuple>
drop_tuple_n_t<N, remove_cvref_t<Tuple>> tuple_drop(Tuple&& tuple)
{
    constexpr size_t size_v = std::tuple_size_v<remove_cvref_t<Tuple>>;
    static_assert(size_v >= N, "not enough tuple elements.");
    return tuple_takedrop_impl<N>(std::forward<decltype(tuple)>(tuple),
                                  std::make_index_sequence<size_v - N>{});
}


}
