#pragma once

#include <tuple>
#include <vector>
#include <type_traits>

#include "decls.h"

namespace ndarray
{

enum class span_type
{
    scalar,
    all,
    simple,
    regular,
    irregular,
    invalid
};
enum class indexer_type
{
    scalar,
    all,
    simple,
    regular,
    irregular,
    invalid
};
enum class array_obj_type
{
    scalar,    // not used
    vector, 
    array,
    simple,
    regular,
    irregular,
    range,
    invalid    // not used
};
//enum class access_type
//{
//    lv_array, // lvalue array
//    rv_array, // rvalue array
//    iterator, // object that implements O(1) element_begin() and element_end()
//    traverse  // object that implements O(1) traverse(Fn)
//};


template<size_t N, typename ResultTuple = std::tuple<>>
struct n_all_indexer_tuple;
template<size_t N, typename... Results>
struct n_all_indexer_tuple<N, std::tuple<Results...>>
{
    using type = typename n_all_indexer_tuple<N - 1, std::tuple<Results..., all_indexer>>::type;
};
template<typename... Results>
struct n_all_indexer_tuple<size_t(0), std::tuple<Results...>>
{
    using type = std::tuple<Results...>;
};
template<size_t N>
using n_all_indexer_tuple_t = typename n_all_indexer_tuple<N>::type;


template<typename... Ts>
struct is_all_ints;
template<typename T1, typename... Ts>
struct is_all_ints<T1, Ts...>
{
    static constexpr bool value = std::is_integral_v<remove_cvref_t<T1>> && is_all_ints<Ts...>::value;
};
template<>
struct is_all_ints<>
{
    static constexpr bool value = true;
};
template<typename... Ts>
constexpr bool is_all_ints_v = is_all_ints<Ts...>::value;


template<typename Tuple>
struct is_all_int_tuple :
    std::false_type {};
template<typename... Ts>
struct is_all_int_tuple<std::tuple<Ts...>> :
    is_all_ints<Ts...> {};
template<typename Tuple>
constexpr bool is_all_int_tuple_v = is_all_int_tuple<Tuple>::value;


template<typename Span>
struct classify_span_type
{
    static constexpr span_type value = std::is_integral_v<Span> ? span_type::scalar : span_type::invalid;
};
template<>
struct classify_span_type<all_span>
{
    static constexpr span_type value = span_type::all;
};
template<typename... T>
struct classify_span_type<simple_span<T...>>
{
    static constexpr span_type value = span_type::simple;
};
template<typename... T>
struct classify_span_type<regular_span<T...>>
{
    static constexpr span_type value = span_type::regular;
};
template<typename... T>
struct classify_span_type<irregular_span<T...>>
{
    static constexpr span_type value = span_type::irregular;
};
template<typename Span>
constexpr span_type classify_span_type_v = classify_span_type<Span>::value;

template<typename Indexer>
struct indexer_type_of
{
    static constexpr indexer_type value =
        std::is_same_v<Indexer, scalar_indexer>    ? indexer_type::scalar    :
        std::is_same_v<Indexer, all_indexer>       ? indexer_type::all       :
        std::is_same_v<Indexer, simple_indexer>    ? indexer_type::simple    :
        std::is_same_v<Indexer, regular_indexer>   ? indexer_type::regular   :
        std::is_same_v<Indexer, irregular_indexer> ? indexer_type::irregular : indexer_type::invalid;
};
template<typename Indexer>
constexpr indexer_type indexer_type_of_v = indexer_type_of<Indexer>::value;


template<typename Indexer, typename Span>
struct indexer_collapsing
{
    using span_t = remove_cvref_t<Span>;
    using indexer_t = remove_cvref_t<Indexer>;
    static constexpr span_type    span_v = classify_span_type_v<span_t>;
    static constexpr indexer_type indexer_v = indexer_type_of_v<indexer_t>;
    static_assert(indexer_v != indexer_type::scalar  &&
                  indexer_v != indexer_type::invalid &&
                  span_v != span_type::invalid);
    using type =
        std::conditional_t<span_v == span_type::scalar, scalar_indexer,
        std::conditional_t<span_v == span_type::all, indexer_t,
        std::conditional_t<span_v == span_type::simple,
        std::conditional_t<indexer_v == indexer_type::all, simple_indexer, indexer_t>,
        std::conditional_t<span_v == span_type::regular,
        std::conditional_t<indexer_v == indexer_type::irregular, irregular_indexer, regular_indexer>,
        irregular_indexer>>>>;
};
template<typename Indexer, typename Span>
using indexer_collapsing_t = typename indexer_collapsing<Indexer, Span>::type;


template<typename IndexerTuple, typename SpanTuple, typename ResultTuple = std::tuple<>>
struct indexer_tuple_collapsing_impl;
template<typename I1, typename... Is, typename S1, typename... Ss, typename... Rs>
struct indexer_tuple_collapsing_impl<std::tuple<I1, Is...>, std::tuple<S1, Ss...>, std::tuple<Rs...>>
{
    using collapsed_t = indexer_collapsing_t<I1, S1>;
    using type = typename indexer_tuple_collapsing_impl<
        std::tuple<Is...>, std::tuple<Ss...>, std::tuple<Rs..., collapsed_t>>::type;;
};
template<typename... Is, typename S1, typename... Ss, typename... Rs>
struct indexer_tuple_collapsing_impl<std::tuple<scalar_indexer, Is...>, std::tuple<S1, Ss...>, std::tuple<Rs...>>
{
    using type = typename indexer_tuple_collapsing_impl<
        std::tuple<Is...>, std::tuple<S1, Ss...>, std::tuple<Rs..., scalar_indexer>>::type;
};
template<typename... Is, typename... Rs>
struct indexer_tuple_collapsing_impl<std::tuple<Is...>, std::tuple<>, std::tuple<Rs...>>
{
    using type = std::tuple<Rs..., Is...>;
};
template<typename S1, typename... Ss, typename... Rs>
struct indexer_tuple_collapsing_impl<std::tuple<>, std::tuple<S1, Ss...>, std::tuple<Rs...>>
{
    static_assert(_always_false_v<S1>, "Too many span specifications.");
};
template<typename IndexerTuple, typename SpanTuple>
struct indexer_tuple_collapsing :
    indexer_tuple_collapsing_impl<remove_cvref_t<IndexerTuple>, remove_cvref_t<SpanTuple>> {};
template<typename IndexerTuple, typename SpanTuple>
using indexer_tuple_collapsing_t = typename indexer_tuple_collapsing<IndexerTuple, SpanTuple>::type;



template<typename IndexerTuple, size_t N = std::tuple_size_v<IndexerTuple>>
struct indexer_tuple_depth;
template<typename Indexer1, typename... Indexers, size_t N>
struct indexer_tuple_depth<std::tuple<Indexer1, Indexers...>, N>
{
    static constexpr size_t value = indexer_tuple_depth<
        std::tuple<Indexers...>, std::is_same_v<Indexer1, scalar_indexer> ? (N - 1) : N>::value;
};
template<size_t N>
struct indexer_tuple_depth<std::tuple<>, N>
{
    static constexpr size_t value = N;
};
template<typename IndexerTuple>
constexpr size_t indexer_tuple_depth_v = indexer_tuple_depth<IndexerTuple>::value;


template<typename IndexerTuple, size_t I = 0, array_obj_type State = array_obj_type::scalar, size_t N = std::tuple_size_v<IndexerTuple>>
struct identify_view_type
{
    static_assert(I < N);
    using vt = array_obj_type;
    using it = indexer_type;
    static constexpr vt sv = State;
    static constexpr it iv = indexer_type_of_v<std::tuple_element_t<I, IndexerTuple>>;

    static constexpr array_obj_type new_sv =
        (sv == vt::invalid || iv == it::invalid) ?
        (vt::invalid) :
        (sv == vt::scalar) ?
        (iv == it::scalar ? vt::scalar :
         iv == it::all ? vt::simple :
         iv == it::simple ? vt::simple :
         iv == it::regular ? vt::regular : vt::irregular) :
         (sv == vt::simple) ?
        (iv == it::scalar ? vt::regular :
         iv == it::all ? vt::simple : vt::irregular) :
         (sv == vt::regular && iv == it::scalar) ?
        (vt::regular) :
        (vt::irregular);

    static constexpr array_obj_type value = identify_view_type<IndexerTuple, I + 1, new_sv, N>::value;
};
template<typename IndexerTuple, array_obj_type State, size_t N>
struct identify_view_type<IndexerTuple, N, State, N>
{
    static constexpr array_obj_type value = State;
};
template<typename IndexerTuple>
constexpr array_obj_type identify_view_type_v = identify_view_type<IndexerTuple>::value;


template<typename IndexerTuple, typename Sequence = std::index_sequence<>, size_t N = std::tuple_size_v<IndexerTuple>>
struct non_scalar_indexer_finder;
template<typename Indexer1, typename... Indexers, size_t... Is, size_t N>
struct non_scalar_indexer_finder<std::tuple<Indexer1, Indexers...>, std::index_sequence<Is...>, N>
{
    using type = typename non_scalar_indexer_finder<
        std::tuple<Indexers...>, std::index_sequence<Is..., N - (1 + sizeof...(Indexers))>, N>::type;
};
template<typename... Indexers, size_t... Is, size_t N>
struct non_scalar_indexer_finder<std::tuple<scalar_indexer, Indexers...>, std::index_sequence<Is...>, N>
{
    using type = typename non_scalar_indexer_finder<
        std::tuple<Indexers...>, std::index_sequence<Is...>, N>::type;
};
template<typename Sequence, size_t N>
struct non_scalar_indexer_finder<std::tuple<>, Sequence, N>
{
    using type = Sequence;
};

template<size_t... Is>
constexpr std::array<size_t, sizeof...(Is)> _make_non_scalar_indexer_table_impl(std::index_sequence<Is...>)
{
    return {Is...};
}
template<typename IndexerTuple>
constexpr std::array<size_t, indexer_tuple_depth_v<IndexerTuple>> make_non_scalar_indexer_table()
{
    return _make_non_scalar_indexer_table_impl(typename non_scalar_indexer_finder<IndexerTuple>::type{});
}


template<typename T, typename IndexerTuple, typename SpanTuple>
struct deduce_view_type
{
    using indexers_t = indexer_tuple_collapsing_t<IndexerTuple, SpanTuple>;
    static constexpr array_obj_type _view_type_v = identify_view_type_v<indexers_t>;

    using type =
        std::conditional_t<_view_type_v == array_obj_type::simple, simple_view<T, indexers_t>,
        std::conditional_t<_view_type_v == array_obj_type::regular, regular_view<T, indexers_t>,
        std::conditional_t<_view_type_v == array_obj_type::irregular, irregular_view<T, indexers_t>,
        void>>>;
};
template<typename T, typename IndexerTuple, typename SpanTuple>
using deduce_view_type_t = typename deduce_view_type<T, IndexerTuple, SpanTuple>::type;


template<typename T, typename IndexerTuple, typename SpanTuple>
struct deduce_part_array_type
{
    using collapsed_t = indexer_tuple_collapsing_t<IndexerTuple, SpanTuple>;
    using type        = array<T, indexer_tuple_depth_v<collapsed_t>>;
};
template<typename T, typename IndexerTuple, typename SpanTuple>
using deduce_part_array_type_t = typename deduce_part_array_type<T, IndexerTuple, SpanTuple>::type;


// gives type U if SpanTuple contains Depth integers
template<typename U, typename T, size_t Depth, typename IndexerTuple, typename SpanTuple>
struct deduce_view_or_elem_type
{
    using type = std::conditional_t<
        std::tuple_size_v<SpanTuple> == Depth && is_all_int_tuple_v<SpanTuple>,
        U, deduce_view_type_t<T, IndexerTuple, SpanTuple>>;
};
template<typename U, typename T, size_t Depth, typename IndexerTuple, typename SpanTuple>
using deduce_view_or_elem_type_t = typename deduce_view_or_elem_type<U, T, Depth, IndexerTuple, SpanTuple>::type;

// gives type U if SpanTuple contains Depth integers
template<typename U, typename T, size_t Depth, typename IndexerTuple, typename SpanTuple>
struct deduce_part_or_elem_type
{
    using type = std::conditional_t<
        std::tuple_size_v<SpanTuple> == Depth && is_all_int_tuple_v<SpanTuple>,
        U, deduce_part_array_type_t<T, IndexerTuple, SpanTuple>>;
};
template<typename U, typename T, size_t Depth, typename IndexerTuple, typename SpanTuple>
using deduce_part_or_elem_type_t = typename deduce_part_or_elem_type<U, T, Depth, IndexerTuple, SpanTuple>::type;



// identify_view_iter_type gives the category of iterator in terms of its view
template<size_t IterDepth, typename IndexerTuple, typename TakenTuple = std::tuple<>>
struct identify_view_iter_type;
template<size_t IterDepth, typename I1, typename... Is, typename... Takens>
struct identify_view_iter_type<IterDepth, std::tuple<I1, Is...>, std::tuple<Takens...>> :
    identify_view_iter_type<IterDepth - 1, std::tuple<Is...>, std::tuple<Takens..., I1>> {};
template<typename I1, typename... Is, typename... Takens>
struct identify_view_iter_type<0, std::tuple<I1, Is...>, std::tuple<Takens...>>
{
    using iter_indexers_t = std::tuple<Takens..., scalar_indexer /* acting as all remaining scalars */>;
    static constexpr array_obj_type value = identify_view_type_v<iter_indexers_t>;
};
template<size_t IterDepth, typename IndexerTuple>
constexpr array_obj_type identify_view_iter_type_v = identify_view_iter_type<IterDepth, IndexerTuple>::value;


// deduce_view_iter_type gives the type of iterator based on the indexer tuple
template<size_t IterDepth, typename IndexerTuple, typename BaseView, typename SubView, bool IsExplicitConst>
struct deduce_view_iter_type
{
    static constexpr array_obj_type view_iter_type_v = identify_view_iter_type_v<IterDepth, IndexerTuple>;
    using type =
        std::conditional_t<view_iter_type_v == array_obj_type::regular,
        regular_view_iter<SubView, IsExplicitConst>,
        std::conditional_t<view_iter_type_v == array_obj_type::irregular,
        irregular_view_iter<SubView, BaseView, IsExplicitConst>,
        void>>;
};
template<size_t IterDepth, typename IndexerTuple, typename BaseView, typename SubView, bool IsExplicitConst>
using deduce_view_iter_type_t = typename deduce_view_iter_type<
    IterDepth, IndexerTuple, BaseView, SubView, IsExplicitConst>::type;


//template<typename Array>
//struct access_type_of_impl2;
//template<typename Array>
//struct access_type_of_impl1 :
//    access_type_of_impl2<std::remove_reference_t<Array>> {};
//template<typename T, size_t Depth>
//struct access_type_of_impl1<array<T, Depth>&>
//{
//    static constexpr access_type value = access_type::lv_array;
//};
//template<typename T, size_t Depth>
//struct access_type_of_impl1<array<T, Depth>&&>
//{
//    static constexpr access_type value = access_type::rv_array;
//}; template<typename T, size_t Depth>
//struct access_type_of_impl1<array<T, Depth>>
//{
//    // should not reach
//};
//template<typename T, typename IndexerTuple>
//struct access_type_of_impl2<simple_view<T, IndexerTuple>>
//{
//    static constexpr access_type value = access_type::iterator;
//};
//template<typename T, typename IndexerTuple>
//struct access_type_of_impl2<regular_view<T, IndexerTuple>>
//{
//    static constexpr access_type value = access_type::iterator;
//};
//template<typename T, typename IndexerTuple>
//struct access_type_of_impl2<irregular_view<T, IndexerTuple>>
//{
//    static constexpr access_type value = access_type::traverse;
//};
//template<typename T, bool IsUnitStep>
//struct access_type_of_impl2<range_view<T, IsUnitStep>>
//{
//    static constexpr access_type value = access_type::iterator;
//};
//template<typename Array>
//struct access_type_of :
//    access_type_of_impl1<std::remove_cv_t<Array>> {};
//template<typename Array>
//constexpr access_type access_type_of_v = access_type_of<Array>::value;
//
//using lv_array_access_tag = std::integral_constant<access_type, access_type::lv_array>;
//using rv_array_access_tag = std::integral_constant<access_type, access_type::rv_array>;
//using iterator_access_tag = std::integral_constant<access_type, access_type::iterator>;
//using traverse_access_tag = std::integral_constant<access_type, access_type::traverse>;
//
//template<typename Array>
//using access_type_tag = std::integral_constant<access_type, access_type_of_v<Array>>;


template<typename Array>
struct is_array_object_impl : 
    std::false_type {};
template<typename T, size_t Depth>
struct is_array_object_impl<array<T, Depth>> : 
    std::true_type {};
template<typename T, typename IndexerTuple>
struct is_array_object_impl<simple_view<T, IndexerTuple>> : 
    std::true_type {};
template<typename T, typename IndexerTuple>
struct is_array_object_impl<regular_view<T, IndexerTuple>> : 
    std::true_type {};
template<typename T, typename IndexerTuple>
struct is_array_object_impl<irregular_view<T, IndexerTuple>> : 
    std::true_type {};
template<typename T, bool IsUnitStep>
struct is_array_object_impl<range_view<T, IsUnitStep>> : 
    std::true_type {};
template<typename Array>
struct is_array_object : 
    is_array_object_impl<remove_cvref_t<Array>> {};
template<typename Array>
constexpr bool is_array_object_v = is_array_object<Array>::value;


// array_elem_of_t returns the element type of an object when it is 
// converted to an array. 
// array_elem_of_t automatically remove cv qualifiers of the type
template<typename Array, bool IsArray = is_array_object_v<Array>>
struct array_elem_of_impl;
template<typename Array>
struct array_elem_of_impl<Array, true>
{
    using type = typename Array::_elem_t;
};
template<typename Scalar>
struct array_elem_of_impl<Scalar, false>
{
    using type = Scalar;
};
template<typename T>
struct array_elem_of_impl<std::vector<T>, false>
{
    using type = T;
};
template<typename Array>
struct array_elem :
    array_elem_of_impl<remove_cvref_t<Array>> {};
template<typename Array>
using array_elem_of_t = typename array_elem<Array>::type;


template<typename Any, bool IsArithmetic = std::is_arithmetic_v<Any>>
struct array_or_range_elem_of_impl;
template<typename Array>
struct array_or_range_elem_of_impl<Array, false>
{
    using type = array_elem_of_t<Array>;
};
template<typename Arithmetic>
struct array_or_range_elem_of_impl<Arithmetic, true>
{
    using type = decltype(int(0) + Arithmetic{});
};
template<typename Any>
struct array_or_range_elem_of :
    array_or_range_elem_of_impl<Any> {};
template<typename Any>
using array_or_range_elem_of_t = typename array_or_range_elem_of<Any>::type;


template<typename Array>
struct array_depth_of_impl
{
    static constexpr size_t value = Array::_depth_v;
};
template<typename T>
struct array_depth_of_impl<std::vector<T>>
{
    static constexpr size_t value = 1;
};
template<typename Array>
struct array_depth_of :
    array_depth_of_impl<remove_cvref_t<Array>> {};
template<typename Array>
constexpr size_t array_depth_of_v = array_depth_of<Array>::value;

template<typename T, typename IndexerTuple>
class simple_view;
template<typename T, typename IndexerTuple>
class regular_view;
template<typename T, typename IndexerTuple>
class irregular_view;
template<typename T, bool IsUnitStep>
class range_view;

template<typename Array>
struct array_obj_type_of_impl;
template<typename T>
struct array_obj_type_of_impl<std::vector<T>>
{
    static constexpr array_obj_type value = array_obj_type::vector;
};
template<typename T, size_t Depth>
struct array_obj_type_of_impl<array<T, Depth>>
{
    static constexpr array_obj_type value = array_obj_type::array;
};
template<typename T, typename IndexerTuple>
struct array_obj_type_of_impl<simple_view<T, IndexerTuple>>
{
    static constexpr array_obj_type value = array_obj_type::simple;
};
template<typename T, typename IndexerTuple>
struct array_obj_type_of_impl<regular_view<T, IndexerTuple>>
{
    static constexpr array_obj_type value = array_obj_type::regular;
};
template<typename T, typename IndexerTuple>
struct array_obj_type_of_impl<irregular_view<T, IndexerTuple>>
{
    static constexpr array_obj_type value = array_obj_type::irregular;
};
template<typename T, bool IsUnitStep>
struct array_obj_type_of_impl<range_view<T, IsUnitStep>>
{
    static constexpr array_obj_type value = array_obj_type::range;
};
template<typename Array>
struct array_obj_type_of : 
    array_obj_type_of_impl<remove_cvref_t<Array>> {};
template<typename Array>
constexpr array_obj_type array_obj_type_of_v = array_obj_type_of<Array>::value;

}
