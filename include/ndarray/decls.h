#pragma once

#include "utils.h"

namespace ndarray
{

template<typename T, size_t Depth>
class array;

class all_span;
template<typename TFirst, typename TLast>
class simple_span;
template<typename TFirst, typename TLast, typename TStep>
class regular_span;
template<typename Indices>
class irregular_span;

class scalar_indexer;
class all_indexer;
class simple_indexer;
class regular_indexer;
class irregular_indexer;

template<typename T, typename IndexerTuple>
class array_view_base;
template<typename T, typename IndexerTuple>
class simple_view;
template<typename T, typename IndexerTuple>
class regular_view;
template<typename T, typename IndexerTuple>
class irregular_view;
template<typename T, bool IsUnitStep>
class range_view;

template<typename SubView, bool IsExplicitConst>
class regular_view_iter;
template<typename SubView, typename BaseView, bool IsExplicitConst>
class irregular_view_iter;
template<typename T, bool IsUnitStep, bool IsIntegral = std::is_integral_v<T>>
class range_view_iter;

template<typename ViewCref, size_t IndicesDepth>
class irregular_indices;
template<typename T, bool IsExplicitConst = false>
class simple_elem_iter;
template<typename T, bool IsExplicitConst = false>
class regular_elem_iter;
template<typename View, bool IsExplicitConst = false>
class irregular_elem_iter;

template<typename T>
using simple_elem_const_iter = typename simple_elem_iter<T, true>;
template<typename T>
using regular_elem_const_iter = typename regular_elem_iter<T, true>;
template<typename View>
using irregular_elem_const_iter = typename irregular_elem_iter<View, true>;

template<typename SrcArray, typename DstArray>
inline void data_copy(const SrcArray& src, DstArray& dst);
template<typename SrcArray, typename DstArray>
inline void aliased_data_copy(const SrcArray& src, DstArray& dst, size_t size);
template<typename SrcArray, typename DstArray>
inline void no_alias_data_copy(const SrcArray& src, DstArray& dst, size_t size);

}
