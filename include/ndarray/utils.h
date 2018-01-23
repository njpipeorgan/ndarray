#pragma once

#include <cassert>
#include <algorithm>

#ifdef _DEBUG
#define ENABLE_NDARRAY_ASSERT 1
#define ENABLE_NDARRAY_CHECK_BOUND 1
#endif

#if ENABLE_NDARRAY_ASSERT
#define NDARRAY_ASSERT(x) assert(x)
#else 
#define NDARRAY_ASSERT(x) 0
#endif

#if ENABLE_NDARRAY_CHECK_BOUND
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
constexpr bool _always_false_v = false;

// used to print type
template<typename... T>
void static_print_type()
{
    static_assert(false);
}

// constructed from arbitrary arguments and gives nothing
struct _empty_struct
{
    template<typename... T>
    _empty_struct(T...) {}
};


// tuple of T repeated n times
template<size_t N, typename T, typename Tuple = std::tuple<>>
struct _repeat_tuple;
template<size_t N, typename T, typename... Ts>
struct _repeat_tuple<N, T, std::tuple<Ts...>>
{
    using type = typename _repeat_tuple<N - 1, T, std::tuple<Ts..., T>>::type;
};
template<typename T, typename... Ts>
struct _repeat_tuple<0, T, std::tuple<Ts...>>
{
    using type = std::tuple<Ts...>;
};
template<size_t N, typename T>
using _repeat_tuple_t = typename _repeat_tuple<N, T>::type;


template<typename Return, typename X, typename Y>
inline Return _add_if_negative(X x, Y y)
{
    if constexpr (std::is_unsigned_v<X>)
        return Return(x);
    else
        return (x >= X(0)) ? Return(x) : Return(x) + Return(y);
}

template<typename Return, typename X, typename Y>
inline Return _add_if_non_positive(X x, Y y)
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

}
