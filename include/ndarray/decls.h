#pragma once

#include "utils.h"

namespace ndarray
{

template<typename T, size_t Depth>
class array;

class _all_span;
template<typename TFirst, typename TLast>
class _simple_span;
template<typename TFirst, typename TLast, typename TStep>
class _regular_span;
template<typename Indices>
class _irregular_span;

class _scalar_indexer;
class _all_indexer;
class _simple_indexer;
class _regular_indexer;
class _irregular_indexer;

template<typename T, typename IndexerTuple>
class array_view_base;
template<typename T, typename IndexerTuple>
class simple_view;
template<typename T, typename IndexerTuple>
class regular_view;
template<typename T, typename IndexerTuple>
class irregular_view;

template<typename SubView, bool IsExplicitConst>
class _regular_view_iter;
template<typename SubView, typename BaseView, bool IsExplicitConst>
class _irregular_view_iter;

template<typename T, bool IsExplicitConst = false>
class _simple_elem_iter;
template<typename T, bool IsExplicitConst = false>
class _regular_elem_iter;
template<typename View, bool IsExplicitConst = false>
class _irregular_elem_iter;

template<typename T>
using _simple_elem_const_iter = typename _simple_elem_iter<T, true>;
template<typename T>
using _regular_elem_const_iter = typename _regular_elem_iter<T, true>;
template<typename View>
using _irregular_elem_const_iter = typename _irregular_elem_iter<View, true>;


}
