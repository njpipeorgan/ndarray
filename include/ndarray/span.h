#pragma once

#include <algorithm>
#include <vector>

#include "decls.h"
#include "traits.h"

namespace ndarray
{

//
// Spans are used to indicate which indices to take on all levels. 
//
//  span type   called with
//-------------------------------------------------
//  [scalar]    [an integer]
//  all         span()
//  simple      span(i, j) or span(j)
//  regular     span(i, j, k) [with k as step]
//  irregular   span({i1, i2, i3, ...})
//
//
// Span specifications should be integers. A negative integer for i/j 
// means counting from the end, and a negative integer for k means a 
// negative step. 
//
// [0]  for j denotes the index following the last position. 
// [-1] for i denotes the index before the first position. 
//
// An integer in place of a span means take this position only, which
// reduces the resulting array or array view by one. 
//


class _all_span
{
public:
    constexpr explicit _all_span() {}

    size_t first(size_t size) const noexcept
    {
        return 0;
    }

    size_t last(size_t size) const noexcept
    {
        return size;
    }
};

template<typename TFirst, typename TLast>
class _simple_span
{
    static_assert(std::is_integral_v<TFirst> &&
                  std::is_integral_v<TLast>,
                  "Span specifications should have integral types.");
protected:
    TFirst first_{};
    TLast last_{};

public:
    constexpr explicit _simple_span() {}

    constexpr explicit _simple_span(TFirst first, TLast last) :
        first_(first), last_(last) {}

    size_t first(size_t size) const noexcept
    {
        size_t ret = _add_if_negative<size_t>(first_, size);
        NDARRAY_ASSERT(ret <= size);
        return ret;
    }

    size_t last(size_t size) const noexcept
    {
        size_t ret = _add_if_non_positive<size_t>(last_, size);
        NDARRAY_ASSERT(ret <= size);
        return ret;
    }
};

template<typename TFirst, typename TLast, typename TStep>
class _regular_span : public _simple_span<TFirst, TLast>
{
    static_assert(std::is_integral_v<TStep>,
                  "Span specifications should have integral types.");

protected:
    TStep step_{};

public:
    constexpr explicit _regular_span() = default;

    constexpr explicit _regular_span(TFirst first, TLast last, TStep step) :
        _simple_span<TFirst, TLast>(first, last), step_(step) {}

    ptrdiff_t first(size_t size) const noexcept
    {
        NDARRAY_ASSERT(step_ != 0);
        ptrdiff_t ret = step_ > 0 ?
            _add_if_negative<size_t>(this->first_, size) :
            _add_if_non_positive<size_t>(this->first_ + 1, size) - 1;
        //NDARRAY_ASSERT(step_ > 0 ? ret <= size : (ret < size || ret == size_t(-1)));
        return ret;
    }
    
    ptrdiff_t last(size_t size) const noexcept
    {
        NDARRAY_ASSERT(step_ != 0);
        ptrdiff_t ret = step_ > 0 ?
            _add_if_non_positive<size_t>(this->last_, size) :
            _add_if_negative<size_t>(this->last_ + 1, size) - 1;
        //NDARRAY_ASSERT(step_ > 0 ? ret <= size : (ret < size || ret == size_t(-1)));
        return ret;
    }
    
    constexpr ptrdiff_t step() const noexcept
    {
        return ptrdiff_t(step_);
    }
};

template<typename Indices>
class _irregular_span
{
protected:
    Indices indices_{};

public:
    explicit _irregular_span() = default;

    explicit _irregular_span(const Indices& indices) :
        indices_(indices) {}
    
    explicit _irregular_span(Indices&& indices) :
        indices_(std::move(indices)) {}

    size_t get_index(size_t i, size_t size)
    {
        NDARRAY_CHECK_BOUND_SCALAR(i, indices_.size());
        size_t ret = _add_if_negative<size_t>(indices_[i], size);
        NDARRAY_CHECK_BOUND_SCALAR(ret, size);
        return ret;
    }

    Indices vector() &&
    {
        return std::move(indices_);
    }

    Indices vector() const &
    {
        return indices_;
    }

};


constexpr auto span()
{
    return _all_span{};
}

template<typename TLast>
constexpr auto span(TLast last)
{
    return _simple_span<size_t, TLast>{0, last};
}

template<typename Index>
constexpr auto span(const std::vector<Index>& indices)
{
    return _irregular_span<std::vector<Index>>{indices};
}

template<typename Index>
constexpr auto span(std::vector<Index>&& indices)
{
    return _irregular_span<std::vector<Index>>{std::move(indices)};
}

template<typename Index>
constexpr auto span(std::initializer_list<Index> indices)
{
    return _irregular_span<std::vector<Index>>{std::vector<Index>(indices)};
}

template<typename TFirst, typename TLast>
constexpr auto span(TFirst first, TLast last)
{
    return _simple_span<TFirst, TLast>{first, last};
}

template<typename TFirst, typename TLast, typename TStep>
constexpr auto span(TFirst first, TLast last, TStep step)
{
    return _regular_span<TFirst, TLast, TStep>{first, last, step};
}

constexpr auto All      = span();
constexpr auto Reversed = span(-1, -1, -1);


}
