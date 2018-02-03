#pragma once

#include "traits.h"

namespace ndarray
{

template<typename TFirst, typename TLast>
constexpr inline auto make_range_view(TFirst first, TLast last)
{
    static_assert(std::is_arithmetic_v<TFirst> && std::is_arithmetic_v<TLast>);
    using elem_t = decltype(first + last);

    auto elem_first  = elem_t(first);
    auto elem_last   = elem_t(last);
    auto diff        = elem_last - elem_first;
    auto signed_size = ptrdiff_t(diff);

    // add one if elem_t is floating point type, and last is not accurately taken
    if (std::is_floating_point_v<elem_t> && diff != signed_size * elem_t{1})
        signed_size += 1;

    auto size = signed_size >= 0 ? size_t(signed_size) : size_t(0);
    return range_view<elem_t, true>{elem_first, size};
}
template<typename TFirst, typename TLast, typename TStep>
constexpr inline auto make_range_view(TFirst first, TLast last, TStep step)
{
    static_assert(std::is_arithmetic_v<TFirst> && std::is_arithmetic_v<TLast> && 
                  std::is_arithmetic_v<TStep>);
    using elem_t = decltype(first + last + step);

    auto elem_first  = elem_t(first);
    auto elem_last   = elem_t(last);
    auto elem_step   = elem_t(step);
    auto diff        = elem_last - elem_first;
    auto signed_size = ptrdiff_t(diff / elem_step);

    // subtract one if elem_t is floating point type, and last is accurately taken
    if (std::is_floating_point_v<elem_t> && diff != signed_size * elem_step)
        signed_size += 1;

    auto size = signed_size >= 0 ? size_t(signed_size) : size_t(0);
    return range_view<elem_t, false>{elem_first, size, elem_step};
}

template<typename T, bool IsUnitStep>
class range_view
{
public:
    using _my_type = range_view;
    using _elem_t  = T;
    static constexpr bool   _is_const_v     = true;
    static constexpr size_t _depth_v        = 1;
    static constexpr bool   _is_unit_step_v = IsUnitStep;
    static constexpr bool   _is_integral_v  = std::is_integral_v<_elem_t>;
    using _step_t = std::conditional_t<_is_unit_step_v, empty_struct, _elem_t>;

    static_assert(std::is_arithmetic_v<_elem_t>);

private:
    const _elem_t first_;
    const size_t  size_;
    const _step_t step_;

public:
    constexpr range_view(_elem_t first, size_t size, _step_t step ={}) :
        first_{first}, size_{size}, step_{step} {}

    constexpr size_t size() const
    {
        return size_;
    }

    template<size_t Level>
    constexpr size_t dimension() const
    {
        static_assert(Level == 0);
        return size_;
    }

    constexpr std::array<size_t, 1> dimensions() const
    {
        return {size_};
    }

    constexpr _elem_t step() const
    {
        if constexpr (_is_unit_step_v)
            return _elem_t{1};
        else
            return step_;
    }

    template<typename Int>
    _elem_t at(Int i) const
    {
        size_t pos = _add_if_negative<size_t>(i, size());
        NDARRAY_ASSERT(pos < size_);
        return _elem_t(first_ + pos * step());
    }

    template<typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
    _elem_t operator()(Int i) const
    {
        return this->at(i);
    }


};



}
