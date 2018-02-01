#pragma once

#include <tuple>
#include <type_traits>

#include "decls.h"

namespace ndarray
{

enum class _span_type
{
    scalar,
    all,
    simple,
    regular,
    irregular,
    invalid
};
enum class _indexer_type
{
    scalar,
    all,
    simple,
    regular,
    irregular,
    invalid
};
enum class _view_type
{
    scalar,    // not used
    simple,
    regular,
    irregular,
    array,     // not used
    invalid    // not used
};
enum class _access_type
{
    vector   = 0,   // object that owns a vector
    iterator = 1, // object that implements O(1) element_begin() and element_end()
    traverse = 2  // object that implements O(1) traverse(Fn)
};

template<size_t N, typename ResultTuple = std::tuple<>>
struct _n_all_indexer_tuple;
template<size_t N, typename... Results>
struct _n_all_indexer_tuple<N, std::tuple<Results...>>
{
    using type = typename _n_all_indexer_tuple<N - 1, std::tuple<Results..., _all_indexer>>::type;
};
template<typename... Results>
struct _n_all_indexer_tuple<size_t(0), std::tuple<Results...>>
{
    using type = std::tuple<Results...>;
};
template<size_t N>
using _n_all_indexer_tuple_t = typename _n_all_indexer_tuple<N>::type;


template<typename Span>
struct _classify_span_type
{
    static constexpr _span_type value = std::is_integral_v<Span> ? _span_type::scalar : _span_type::invalid;
};
template<>
struct _classify_span_type<_all_span>
{
    static constexpr _span_type value = _span_type::all;
};
template<typename... T>
struct _classify_span_type<_simple_span<T...>>
{
    static constexpr _span_type value = _span_type::simple;
};
template<typename... T>
struct _classify_span_type<_regular_span<T...>>
{
    static constexpr _span_type value = _span_type::regular;
};
template<typename... T>
struct _classify_span_type<_irregular_span<T...>>
{
    static constexpr _span_type value = _span_type::irregular;
};
template<typename Span>
constexpr _span_type _classify_span_type_v = _classify_span_type<Span>::value;

template<typename Indexer>
struct _classify_indexer_type
{
    static constexpr _indexer_type value =
        std::is_same_v<Indexer, _scalar_indexer> ? _indexer_type::scalar :
        std::is_same_v<Indexer, _all_indexer> ? _indexer_type::all :
        std::is_same_v<Indexer, _simple_indexer> ? _indexer_type::simple :
        std::is_same_v<Indexer, _regular_indexer> ? _indexer_type::regular :
        std::is_same_v<Indexer, _irregular_indexer> ? _indexer_type::irregular : _indexer_type::invalid;
};
template<typename Indexer>
constexpr _indexer_type _classify_indexer_type_v = _classify_indexer_type<Indexer>::value;

template<typename Indexer, typename Span>
struct _indexer_collapsing
{
    using span_t = remove_cvref_t<Span>;
    using indexer_t = remove_cvref_t<Indexer>;
    static constexpr _span_type    span_v = _classify_span_type_v<span_t>;
    static constexpr _indexer_type indexer_v = _classify_indexer_type_v<indexer_t>;
    static_assert(indexer_v != _indexer_type::scalar  &&
        indexer_v != _indexer_type::invalid &&
        span_v != _span_type::invalid);
    using type =
        std::conditional_t<span_v == _span_type::scalar, _scalar_indexer,
        std::conditional_t<span_v == _span_type::all, indexer_t,
        std::conditional_t<span_v == _span_type::simple,
        std::conditional_t<indexer_v == _indexer_type::all, _simple_indexer, indexer_t>,
        std::conditional_t<span_v == _span_type::regular,
        std::conditional_t<indexer_v == _indexer_type::irregular, _irregular_indexer, _regular_indexer>,
        _irregular_indexer>>>>;
};
template<typename Indexer, typename Span>
using _indexer_collapsing_t = typename _indexer_collapsing<Indexer, Span>::type;


template<typename IndexerTuple, typename SpanTuple, typename ResultTuple = std::tuple<>>
struct _indexer_tuple_collapsing_impl;
template<typename I1, typename... Is, typename S1, typename... Ss, typename... Rs>
struct _indexer_tuple_collapsing_impl<std::tuple<I1, Is...>, std::tuple<S1, Ss...>, std::tuple<Rs...>>
{
    using collapsed_t = _indexer_collapsing_t<I1, S1>;
    using type = typename _indexer_tuple_collapsing_impl<
        std::tuple<Is...>, std::tuple<Ss...>, std::tuple<Rs..., collapsed_t>>::type;;
};
template<typename... Is, typename S1, typename... Ss, typename... Rs>
struct _indexer_tuple_collapsing_impl<std::tuple<_scalar_indexer, Is...>, std::tuple<S1, Ss...>, std::tuple<Rs...>>
{
    using type = typename _indexer_tuple_collapsing_impl<
        std::tuple<Is...>, std::tuple<S1, Ss...>, std::tuple<Rs..., _scalar_indexer>>::type;
};
template<typename... Is, typename... Rs>
struct _indexer_tuple_collapsing_impl<std::tuple<Is...>, std::tuple<>, std::tuple<Rs...>>
{
    using type = std::tuple<Rs..., Is...>;
};
template<typename S1, typename... Ss, typename... Rs>
struct _indexer_tuple_collapsing_impl<std::tuple<>, std::tuple<S1, Ss...>, std::tuple<Rs...>>
{
    static_assert(_always_false_v<S1>, "Too many span specifications.");
};
template<typename IndexerTuple, typename SpanTuple>
struct _indexer_tuple_collapsing :
    _indexer_tuple_collapsing_impl<remove_cvref_t<IndexerTuple>, remove_cvref_t<SpanTuple>> {};
template<typename IndexerTuple, typename SpanTuple>
using _indexer_tuple_collapsing_t = typename _indexer_tuple_collapsing<IndexerTuple, SpanTuple>::type;



template<typename IndexerTuple, size_t N = std::tuple_size_v<IndexerTuple>>
struct _indexer_tuple_depth;
template<typename Indexer1, typename... Indexers, size_t N>
struct _indexer_tuple_depth<std::tuple<Indexer1, Indexers...>, N>
{
    static constexpr size_t value = _indexer_tuple_depth<
        std::tuple<Indexers...>, std::is_same_v<Indexer1, _scalar_indexer> ? (N - 1) : N>::value;
};
template<size_t N>
struct _indexer_tuple_depth<std::tuple<>, N>
{
    static constexpr size_t value = N;
};
template<typename IndexerTuple>
constexpr size_t _indexer_tuple_depth_v = _indexer_tuple_depth<IndexerTuple>::value;

template<typename IndexerTuple, size_t I = 0, _view_type State = _view_type::scalar,
    size_t N = std::tuple_size_v<IndexerTuple>>
    struct _identify_view_type
{
    static_assert(I < N);
    using vt = _view_type;
    using it = _indexer_type;
    static constexpr vt sv = State;
    static constexpr it iv = _classify_indexer_type_v<std::tuple_element_t<I, IndexerTuple>>;

    static constexpr _view_type new_sv =
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

    static constexpr _view_type value = _identify_view_type<IndexerTuple, I + 1, new_sv, N>::value;
};
template<typename IndexerTuple, _view_type State, size_t N>
struct _identify_view_type<IndexerTuple, N, State, N>
{
    static constexpr _view_type value = State;
};
template<typename IndexerTuple>
constexpr _view_type _identify_view_type_v = _identify_view_type<IndexerTuple>::value;


template<typename IndexerTuple, typename Sequence = std::index_sequence<>, size_t N = std::tuple_size_v<IndexerTuple>>
struct _non_scalar_indexer_finder;
template<typename Indexer1, typename... Indexers, size_t... Is, size_t N>
struct _non_scalar_indexer_finder<std::tuple<Indexer1, Indexers...>, std::index_sequence<Is...>, N>
{
    using type = typename _non_scalar_indexer_finder<
        std::tuple<Indexers...>, std::index_sequence<Is..., N - (1 + sizeof...(Indexers))>, N>::type;
};
template<typename... Indexers, size_t... Is, size_t N>
struct _non_scalar_indexer_finder<std::tuple<_scalar_indexer, Indexers...>, std::index_sequence<Is...>, N>
{
    using type = typename _non_scalar_indexer_finder<
        std::tuple<Indexers...>, std::index_sequence<Is...>, N>::type;
};
template<typename Sequence, size_t N>
struct _non_scalar_indexer_finder<std::tuple<>, Sequence, N>
{
    using type = Sequence;
};

template<size_t... Is>
constexpr std::array<size_t, sizeof...(Is)> _make_non_scalar_indexer_table_impl(std::index_sequence<Is...>)
{
    return { Is... };
}
template<typename IndexerTuple>
constexpr std::array<size_t, _indexer_tuple_depth_v<IndexerTuple>> _make_non_scalar_indexer_table()
{
    return _make_non_scalar_indexer_table_impl(typename _non_scalar_indexer_finder<IndexerTuple>::type{});
}


template<typename T, typename IndexerTuple, typename SpanTuple>
struct _derive_view_type
{
    using indexers_t = _indexer_tuple_collapsing_t<IndexerTuple, SpanTuple>;
    static constexpr _view_type _view_type_v = _identify_view_type_v<indexers_t>;

    using type =
        std::conditional_t<_view_type_v == _view_type::simple, simple_view<T, indexers_t>,
        std::conditional_t<_view_type_v == _view_type::regular, regular_view<T, indexers_t>,
        std::conditional_t<_view_type_v == _view_type::irregular, irregular_view<T, indexers_t>,
        void>>>;
};
template<typename T, typename IndexerTuple, typename SpanTuple>
using _derive_view_type_t = typename _derive_view_type<T, IndexerTuple, SpanTuple>::type;


template<size_t IterDepth, typename IndexerTuple, typename TakenTuple = std::tuple<>>
struct _identify_view_iter_type;
template<size_t IterDepth, typename I1, typename... Is, typename... Takens>
struct _identify_view_iter_type<IterDepth, std::tuple<I1, Is...>, std::tuple<Takens...>> : 
    _identify_view_iter_type<IterDepth - 1, std::tuple<Is...>, std::tuple<Takens..., I1>> {};
template<typename I1, typename... Is, typename... Takens>
struct _identify_view_iter_type<0, std::tuple<I1, Is...>, std::tuple<Takens...>>
{
    using iter_indexers_t = std::tuple<Takens..., _scalar_indexer /* acting as all remaining scalars */>;
    static constexpr _view_type value = _identify_view_type_v<iter_indexers_t>;
};
template<size_t IterDepth, typename IndexerTuple>
constexpr _view_type _identify_view_iter_type_v = _identify_view_iter_type<IterDepth, IndexerTuple>::value;


template<size_t IterDepth, typename IndexerTuple, typename BaseView, typename SubView, bool IsExplicitConst>
struct _derive_view_iter_type
{
    static constexpr _view_type view_iter_type_v = _identify_view_iter_type_v<IterDepth, IndexerTuple>;
    using type = 
        std::conditional_t<view_iter_type_v == _view_type::regular, 
        _regular_view_iter<SubView, IsExplicitConst>, 
        std::conditional_t<view_iter_type_v == _view_type::irregular, 
        _irregular_view_iter<SubView, BaseView, IsExplicitConst>, 
        void>>;
};
template<size_t IterDepth, typename IndexerTuple, typename BaseView, typename SubView, bool IsExplicitConst>
using _derive_view_iter_type_t = typename _derive_view_iter_type<
    IterDepth, IndexerTuple, BaseView, SubView, IsExplicitConst>::type;


template<typename... Ts>
struct _is_all_ints;
template<typename T1, typename... Ts>
struct _is_all_ints<T1, Ts...>
{
    static constexpr bool value = std::is_integral_v<remove_cvref_t<T1>> && _is_all_ints<Ts...>::value;
};
template<>
struct _is_all_ints<>
{
    static constexpr bool value = true;
};
template<typename... Ts>
constexpr auto _is_all_ints_v = _is_all_ints<Ts...>::value;


template<typename Array>
struct _identify_access_type_impl;
//template<typename T>
//struct _identify_access_type_impl<std::vector<T>>
//{
//    static constexpr _access_type value = _access_type::vector;
//};
template<typename T, size_t Depth>
struct _identify_access_type_impl<array<T, Depth>>
{
    static constexpr _access_type value = _access_type::vector;
};
template<typename T, typename IndexerTuple>
struct _identify_access_type_impl<simple_view<T, IndexerTuple>>
{
    static constexpr _access_type value = _access_type::iterator;
};
template<typename T, typename IndexerTuple>
struct _identify_access_type_impl<regular_view<T, IndexerTuple>>
{
    static constexpr _access_type value = _access_type::iterator;
};
template<typename T, typename IndexerTuple>
struct _identify_access_type_impl<irregular_view<T, IndexerTuple>>
{
    static constexpr _access_type value = _access_type::traverse;
};
template<typename T>
struct _identify_access_type_impl<range_view<T>>
{
    static constexpr _access_type value = _access_type::iterator;
};
template<typename Array>
struct _identify_access_type : 
    _identify_access_type_impl<remove_cvref_t<Array>> {};
template<typename Array>
constexpr _access_type _identify_access_type_v = _identify_access_type<Array>::value;




}
