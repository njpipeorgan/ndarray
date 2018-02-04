#pragma once

#include "decls.h"
#include "traits.h"
#include "span.h"

namespace ndarray
{

//
//  indexer type     derived from        stored as
//---------------------------------------------------------------------
//  scalar           i (integer)         [empty]
//  all              [0, n)              [empty]
//  simple           [i, j)              {size: j - i}
//  regular          [i, j) step: k      {size: (j - i) / k, step: k}
//  irregular        {i1, i2, i3, ...}   std::vector<size_t>
//

class scalar_indexer
{
};

class all_indexer
{
public:
    explicit all_indexer() = default;

    size_t size(size_t base_size) const
    {
        return base_size;
    }
    ptrdiff_t step() const
    {
        return ptrdiff_t(1);
    }
    size_t operator[](size_t i) const
    {
        return i;
    }
    size_t at(size_t i) const
    {
        return (*this)[i];
    }
};

class simple_indexer : public all_indexer
{
protected:
    size_t size_{};

public:
    explicit simple_indexer() = default;

    explicit simple_indexer(size_t size) :
        all_indexer{}, size_{size} {}

    size_t size(size_t = 0) const noexcept
    {
        return size_;
    }

    size_t at(size_t i) const noexcept
    {
        NDARRAY_CHECK_BOUND_SCALAR(i, size_);
        return (*this)[i];
    }
};

class regular_indexer : public simple_indexer
{
protected:
    ptrdiff_t step_{};

public:
    explicit regular_indexer() = default;

    explicit regular_indexer(size_t size, ptrdiff_t step) :
        simple_indexer{size}, step_{step} {}

    ptrdiff_t step() const noexcept
    {
        return step_;
    }
    size_t operator[](size_t i) const noexcept
    {
        return size_t(i * step_);
    }
    size_t at(size_t i) const noexcept
    {
        NDARRAY_CHECK_BOUND_SCALAR(i, size_);
        return (*this)[i];
    }
};

class irregular_indexer
{
public:
    std::vector<size_t> list_{};

public:
    explicit irregular_indexer() = default;

    explicit irregular_indexer(std::vector<size_t>&& list) :
        list_{std::move(list)} {};

    explicit irregular_indexer(const std::vector<size_t>& list) :
        list_{list} {};

    auto size(size_t = 0) const noexcept
    {
        return list_.size();
    }
    size_t operator[](size_t i) const noexcept
    {
        return list_[i];
    }
    size_t at(size_t i) const noexcept
    {
        return list_.at(i);
    }
};


// take the i-th span from a tuple of spans
// gives implicit all_span{} if i is out of range
template<size_t I, typename SpanTuple>
auto _take_ith_span(SpanTuple&& spans)
{
    if constexpr (I < std::tuple_size_v<remove_cvref_t<SpanTuple>>)
        return std::get<I>(std::forward<decltype(spans)>(spans));
    else
        return span(); // implicit all_span
}

// collapse a span into a non-scalar indexer
template<typename Indexer, typename Span>
std::pair<size_t, indexer_collapsing_t<Indexer, Span>> collapse_indexer(
    size_t base_size, Indexer&& indexer, Span&& span)
{
    using derived_result = indexer_collapsing<Indexer, Span>;
    constexpr span_type    span_v    = derived_result::span_v;
    constexpr indexer_type indexer_v = derived_result::indexer_v;

    size_t indexer_size = indexer.size(base_size);

    if constexpr (span_v == span_type::all)
    {
        return {size_t(0), std::forward<decltype(indexer)>(indexer)};
    }
    if constexpr (span_v == span_type::scalar)
    {
        size_t pos = _add_if_negative<size_t>(span, indexer_size);
        NDARRAY_ASSERT(pos < indexer_size);
        return {size_t(indexer.at(pos)), scalar_indexer{}};
    }
    if constexpr (span_v == span_type::simple)
    {
        if constexpr (indexer_v == indexer_type::all ||
                      indexer_v == indexer_type::simple)
        {
            size_t first = span.first(indexer_size);
            size_t last  = span.last(indexer_size);
            NDARRAY_ASSERT(first <= last);
            return {size_t(indexer.at(first)), simple_indexer{last - first}};
        }
        if constexpr (indexer_v == indexer_type::regular)
        {
            size_t    first = span.first(indexer_size);
            size_t    last  = span.last(indexer_size);
            NDARRAY_ASSERT(first <= last);
            return {size_t(indexer.at(first)), regular_indexer{last - first, indexer.step()}};
        }
        if constexpr (indexer_v == indexer_type::irregular)
        {
            size_t first = span.first(indexer_size);
            size_t last  = span.last(indexer_size);
            NDARRAY_ASSERT(first <= last);
            std::vector<size_t> uindices(last - first);
            for (auto& uindex : uindices)
            {
                uindex = indexer.at(first);
                first++;
            }
            return {size_t(0), irregular_indexer{std::move(uindices)}};
        }
    }
    if constexpr (span_v == span_type::regular)
    {
        ptrdiff_t first = span.first(indexer_size);
        ptrdiff_t last  = span.last(indexer_size);
        ptrdiff_t step  = span.step();
        NDARRAY_ASSERT(step > 0 ? first <= last : step < 0 ? first >= last : false);
        size_t    size  = step > 0 ? (last - first - 1) / step + 1 : (first - last - 1) / (-step) + 1;

        if constexpr (indexer_v == indexer_type::irregular)
        {
            std::vector<size_t> uindices(size);
            for (auto& uindex : uindices)
            {
                uindex = indexer.at(first);
                first += step;
            }
            return {size_t(0), irregular_indexer{std::move(uindices)}};
        }
        else
        {
            return {size_t(indexer.at(first)), regular_indexer{size, step * indexer.step()}};
        }
    }
    if constexpr (span_v == span_type::irregular)
    {
        auto span_list = std::forward<decltype(span)>(span).vector();
        if constexpr (std::is_same_v<decltype(span_list), std::vector<size_t>>)
        {
            for (auto& index : span_list)
                index = indexer.at(index);
            return {size_t(0), irregular_indexer{std::move(span_list)}};
        }
        else
        {
            std::vector<size_t> indices(span_list.size());
            std::transform(span_list.begin(), span_list.end(), indices.begin(), [&](auto i)
            {
                size_t pos = _add_if_negative<size_t>(i, indexer_size);
                NDARRAY_ASSERT(pos < indexer_size);
                return indexer.at(pos);
            });
            return {size_t(0), irregular_indexer{std::move(indices)}};
        }
    }
}

// recursive implementation of get_collapsed_view
// updates base_offset, new_indexers, and base_stride
template<view_type ViewType, size_t IC, size_t SC, size_t SD, typename NewTuple, typename IndexerTuple, typename SpanTuple>
void get_collapsed_view_impl(size_t& base_offset, NewTuple& new_indexers, size_t& base_stride, 
                             const size_t* dims, IndexerTuple&& indexers, SpanTuple&& spans)
{
    if constexpr (IC < std::tuple_size_v<remove_cvref_t<IndexerTuple>>)
    {
        decltype(auto) level_indexer = std::get<IC>(std::forward<decltype(indexers)>(indexers));
        constexpr indexer_type indexer_type_v = classify_indexer_type_v<remove_cvref_t<decltype(level_indexer)>>;
        static_assert(indexer_type_v != indexer_type::invalid);

        size_t level_dim = dims[IC];
        base_offset *= level_dim;
        if constexpr (IC >= SD)
            base_stride *= level_dim;

        if constexpr (indexer_type_v == indexer_type::scalar)
        {
            get_collapsed_view_impl<ViewType, IC + 1, SC, SD>(
                base_offset, new_indexers, base_stride, dims, 
                std::forward<decltype(indexers)>(indexers), std::forward<decltype(spans)>(spans));
        }
        else
        {
            auto[offset, indexer] = collapse_indexer(
                level_dim, level_indexer, _take_ith_span<SC>(std::forward<decltype(spans)>(spans)));
            base_offset += offset;
            std::get<IC>(new_indexers) = std::move(indexer);
            get_collapsed_view_impl<ViewType, IC + 1, SC + 1, SD>(
                base_offset, new_indexers, base_stride, dims, 
                std::forward<decltype(indexers)>(indexers), std::forward<decltype(spans)>(spans));
        }
    }
}

// given a view by base_ptr, dims, and its original indexers, 
// derive a new view by collapsing span specifications into non-scalar indexers
template<typename T, typename IndexerTuple, typename SpanTuple>
derive_view_type_t<T, IndexerTuple, SpanTuple> get_collapsed_view(
    T* base_ptr, const size_t* dims, IndexerTuple&& indexers, SpanTuple&& spans)
{
    using derived_type = derive_view_type<T, IndexerTuple, SpanTuple>;
    using view_t       = typename derived_type::type;
    constexpr view_type view_type_v = derived_type::_view_type_v;

    typename view_t::_indexers_t new_indexers{};
    size_t base_offset = 0;
    size_t base_stride = 1;
    get_collapsed_view_impl<view_type_v, 0, 0, view_t::_stride_depth_v>(
        base_offset, new_indexers, base_stride, dims,
        std::forward<decltype(indexers)>(indexers), std::forward<decltype(spans)>(spans));

    return view_t{base_ptr + base_offset, dims, std::move(new_indexers), base_stride};
}

}
