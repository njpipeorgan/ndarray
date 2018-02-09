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
constexpr bool _always_false_v = false;


// constructed from arbitrary arguments and gives nothing
struct empty_struct
{
    template<typename... T>
    constexpr empty_struct(T...) noexcept {}
};


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


}
