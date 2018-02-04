#pragma once

#include "traits.h"

namespace ndarray
{

// specialization for: is unit step, integral type
template<typename T>
class range_view_iter<T, true, true>
{
public:
    using _my_type = range_view_iter;
    using _elem_t  = T;
    static constexpr bool _is_const_v     = true;
    static constexpr bool _is_unit_step_v = true;
    static_assert(std::is_integral_v<T>);

protected:
    _elem_t value_;

public:
    constexpr range_view_iter(_elem_t value, _elem_t /*ignored*/ = 0) :
        value_{value} {}

    _my_type& operator+=(ptrdiff_t diff)
    {
        value += diff;
        return *this;
    }
    _my_type& operator-=(ptrdiff_t diff)
    {
        value -= diff;
        return *this;
    }
    _my_type operator+(ptrdiff_t diff) const
    {
        return _my_type{value_ + diff};
    }
    _my_type operator-(ptrdiff_t diff) const
    {
        return _my_type{value_ + diff};
    }
    _my_type& operator++()
    {
        ++value_;
        return *this;
    }
    _my_type& operator--()
    {
        --value_;
        return *this;
    }
    _my_type operator++(int)
    {
        _my_type ret = *this;
        ++(*this);
        return ret;
    }
    _my_type operator--(int)
    {
        _my_type ret = *this;
        --(*this);
        return ret;
    }

    _elem_t operator*() const
    {
        return value_;
    }
    _elem_t operator[](ptrdiff_t diff) const
    {
        return _elem_t(value_ + diff);
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return this->value_ - other.value_;
    }

    bool operator==(const _my_type& other) const
    {
        return this->value_ == other.value_;
    }
    bool operator<(const _my_type& other) const
    {
        return this->value_ < other.value_;
    }
    bool operator>(const _my_type& other) const
    {
        return this->value_ > other.value_;
    }
    bool operator!=(const _my_type& other) const
    {
        return !((*this) == other);
    }
    bool operator<=(const _my_type& other) const
    {
        return !((*this) > other);
    }
    bool operator>=(const _my_type& other) const
    {
        return !((*this) < other);
    }

};

// specialization for: is not unit step, integral type
template<typename T>
class range_view_iter<T, false, true>
{
public:
    using _my_type = range_view_iter;
    using _elem_t  = T;
    static constexpr bool _is_const_v     = true;
    static constexpr bool _is_unit_step_v = false;
    static_assert(std::is_integral_v<T>);

protected:
    const _elem_t step_;
    _elem_t       value_;

public:
    constexpr range_view_iter(_elem_t value, _elem_t step) :
        step_{step}, value_{value} {}

    _my_type& operator+=(ptrdiff_t diff)
    {
        value_ += diff * step_;
        return *this;
    }
    _my_type& operator-=(ptrdiff_t diff)
    {
        value_ -= diff * step_;
        return *this;
    }
    _my_type operator+(ptrdiff_t diff) const
    {
        return _my_type{value_ + diff * step_};
    }
    _my_type operator-(ptrdiff_t diff) const
    {
        return _my_type{value_ + diff * step_};
    }
    _my_type& operator++()
    {
        value_ += step_;
        return *this;
    }
    _my_type& operator--()
    {
        value_ -= step_;
        return *this;
    }
    _my_type operator++(int)
    {
        _my_type ret = *this;
        ++(*this);
        return ret;
    }
    _my_type operator--(int)
    {
        _my_type ret = *this;
        --(*this);
        return ret;
    }

    _elem_t operator*() const
    {
        return value_;
    }
    _elem_t operator[](ptrdiff_t diff) const
    {
        return _elem_t(value_ + diff * step_);
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return (this->value_ - other.value_) / step_;
    }

    bool operator==(const _my_type& other) const
    {
        return this->value_ == other.value_;
    }
    bool operator<(const _my_type& other) const
    {
        return (this->value_ < other.value_) ^ (step_ > 0);
    }
    bool operator>(const _my_type& other) const
    {
        return (this->value_ > other.value_) ^ (step_ > 0);
    }
    bool operator!=(const _my_type& other) const
    {
        return !((*this) == other);
    }
    bool operator<=(const _my_type& other) const
    {
        return !((*this) > other);
    }
    bool operator>=(const _my_type& other) const
    {
        return !((*this) < other);
    }

};

// specialization for: is unit step, floating point type
template<typename T>
class range_view_iter<T, true, false>
{
public:
    using _my_type = range_view_iter;
    using _elem_t  = T;
    static constexpr bool _is_const_v     = true;
    static constexpr bool _is_unit_step_v = true;
    static_assert(std::is_floating_point_v<T>);

protected:
    const _elem_t first_;
    size_t        pos_;

public:
    constexpr range_view_iter(_elem_t first, size_t pos, _elem_t /*ignored*/ = 0) :
        first_{first}, pos_{pos} {}

    _my_type& operator+=(ptrdiff_t diff)
    {
        pos_ += diff;
        return *this;
    }
    _my_type& operator-=(ptrdiff_t diff)
    {
        pos_ -= diff;
        return *this;
    }
    _my_type operator+(ptrdiff_t diff) const
    {
        return _my_type{first_, pos_ + diff};
    }
    _my_type operator-(ptrdiff_t diff) const
    {
        return _my_type{first_, pos_ - diff};
    }
    _my_type& operator++()
    {
        ++pos_;
        return *this;
    }
    _my_type& operator--()
    {
        --pos_;
        return *this;
    }
    _my_type operator++(int)
    {
        _my_type ret = *this;
        ++(*this);
        return ret;
    }
    _my_type operator--(int)
    {
        _my_type ret = *this;
        --(*this);
        return ret;
    }

    _elem_t operator*() const
    {
        return first_ + pos_;
    }
    _elem_t operator[](ptrdiff_t diff) const
    {
        return first_ + (pos_ + diff);
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return ptrdiff_t(this->pos_ - other.pos_);
    }

    bool operator==(const _my_type& other) const
    {
        NDARRAY_ASSERT(this->first_ == other.first_);
        return this->pos_ == other.pos_;
    }
    bool operator<(const _my_type& other) const
    {
        NDARRAY_ASSERT(this->first_ == other.first_);
        return this->pos_ < other.pos_;
    }
    bool operator>(const _my_type& other) const
    {
        NDARRAY_ASSERT(this->first_ == other.first_);
        return this->pos_ > other.pos_;
    }
    bool operator!=(const _my_type& other) const
    {
        return !((*this) == other);
    }
    bool operator<=(const _my_type& other) const
    {
        return !((*this) > other);
    }
    bool operator>=(const _my_type& other) const
    {
        return !((*this) < other);
    }

};

// specialization for: is not unit step, floating point type
template<typename T>
class range_view_iter<T, false, false>
{
public:
    using _my_type = range_view_iter;
    using _elem_t  = T;
    static constexpr bool _is_const_v     = true;
    static constexpr bool _is_unit_step_v = false;
    static_assert(std::is_floating_point_v<T>);

protected:
    const _elem_t first_;
    const _elem_t step_;
    size_t        pos_;

public:
    constexpr range_view_iter(_elem_t first, size_t pos, _elem_t step) :
        first_{first}, step_{step}, pos_{pos} {}

    _my_type& operator+=(ptrdiff_t diff)
    {
        pos_ += diff;
        return *this;
    }
    _my_type& operator-=(ptrdiff_t diff)
    {
        pos_ -= diff;
        return *this;
    }
    _my_type operator+(ptrdiff_t diff) const
    {
        return _my_type{first_, pos_ + diff, step_};
    }
    _my_type operator-(ptrdiff_t diff) const
    {
        return _my_type{first_, pos_ - diff, step_};
    }
    _my_type& operator++()
    {
        ++pos_;
        return *this;
    }
    _my_type& operator--()
    {
        --pos_;
        return *this;
    }
    _my_type operator++(int)
    {
        _my_type ret = *this;
        ++(*this);
        return ret;
    }
    _my_type operator--(int)
    {
        _my_type ret = *this;
        --(*this);
        return ret;
    }

    _elem_t operator*() const
    {
        return first_ + pos_ * step_;
    }
    _elem_t operator[](ptrdiff_t diff) const
    {
        return first_ + (pos_ + diff) * step_;
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return ptrdiff_t(this->pos_ - other.pos_);
    }

    bool operator==(const _my_type& other) const
    {
        NDARRAY_ASSERT(this->first_ == other.first_ && this->step_ == other.step_);
        return this->pos_ == other.pos_;
    }
    bool operator<(const _my_type& other) const
    {
        NDARRAY_ASSERT(this->first_ == other.first_ && this->step_ == other.step_);
        return this->pos_ < other.pos_;
    }
    bool operator>(const _my_type& other) const
    {
        NDARRAY_ASSERT(this->first_ == other.first_ && this->step_ == other.step_);
        return this->pos_ > other.pos_;
    }
    bool operator!=(const _my_type& other) const
    {
        return !((*this) == other);
    }
    bool operator<=(const _my_type& other) const
    {
        return !((*this) > other);
    }
    bool operator>=(const _my_type& other) const
    {
        return !((*this) < other);
    }

};

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
    using _step_t  = std::conditional_t<_is_unit_step_v, empty_struct, _elem_t>;
    static constexpr view_type _my_view_type_v = view_type::array;

    static_assert(std::is_arithmetic_v<_elem_t>);

protected:
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
            return _elem_t(1);
        else
            return step_;
    }
    constexpr _elem_t first() const
    {
        return first_;
    }
    constexpr _elem_t last() const
    {
        if constexpr (_is_unit_step_v)
            return _elem_t(first_ + size_);
        else
            return _elem_t(first_ + size_ * step_);
    }

    constexpr auto element_cbegin() const
    {
        if constexpr (_is_integral_v)
            return range_view_iter<_elem_t, _is_unit_step_v>{first(), step()};
        else // is floating point
            return range_view_iter<_elem_t, _is_unit_step_v>{first(), size_t(0), step()};
    }
    constexpr auto element_cend() const
    {
        if constexpr (_is_integral_v)
            return range_view_iter<_elem_t, _is_unit_step_v>{last(), step()};
        else // is floating point
            return range_view_iter<_elem_t, _is_unit_step_v>{first(), size(), step()};
    }
    constexpr auto element_begin() const
    {
        return element_cbegin();
    }
    constexpr auto element_end() const
    {
        return element_cend();
    }

    constexpr auto cbegin() const
    {
        return element_cbegin();
    }
    constexpr auto cend() const
    {
        return element_cend();
    }
    constexpr auto begin() const
    {
        return cbegin();
    }
    constexpr auto end() const
    {
        return cend();
    }

    template<typename Int>
    _elem_t at(Int i) const
    {
        size_t pos = _add_if_negative<size_t>(i, size());
        NDARRAY_ASSERT(pos < size_);
        return _elem_t(first_ + pos * step());
    }

    template<typename Span>
    auto vpart(Span&& span)
    {
        static constexpr span_type span_v = classify_span_type_v<Span>;
        if constexpr (span_v == span_type::all)
        {
            return *this;
        }
        else if constexpr (span_v == span_type::simple)
        {
            // get an offset, and an simple_indexer
            auto [offset, indexer] = collapse_indexer(size(), all_indexer{}, std::forward<decltype(span)>(span));
            _elem_t new_first = this->at(offset);
            size_t  new_size  = indexer.size();
            return _my_type{new_first, new_size, step()};
        }
        else if constexpr (span_v == span_type::regular)
        {
            // get an offset, and an regular_indexer
            auto [offset, indexer] = collapse_indexer(size(), all_indexer{}, std::forward<decltype(span)>(span));
            _elem_t new_first = this->at(offset);
            size_t  new_size  = indexer.size();
            _elem_t new_step  = _elem_t(step() * indexer.step());
            return range_view<_elem_t, false>{new_first, new_size, new_step}; // will not be unit step
        }
        else if constexpr (span_v == span_type::irregular)
        {
            size_t new_size  = span.get_size();
            std::vector<_elem_t> data(new_size);
            for (size_t i = 0 ; i < new_size; ++i)
                data[i] = this->at(span.get_index(i, size()));
            return make_array(std::move(data));
        }
    }
    

    template<typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
    _elem_t operator()(Int i) const
    {
        return this->at(i);
    }

    template<typename Function>
    void traverse(Function fn)
    {
        auto iter = this->element_cbegin();
        auto size = this->size();
        for (size_t i = 0; i < size; ++i)
        {
            fn(*iter);
            ++iter;
        }
    }

    // copy data to destination given size
    template<typename Iter>
    void copy_to(Iter dst, size_t size) const
    {
        auto src = this->element_cbegin();
        for (size_t i = 0; i < size; ++i)
        {
            *dst = *src;
            ++dst;
            ++src;
        }
    }

    // copy data to destination, assuming no aliasing
    template<typename Iter>
    void copy_to(Iter dst) const
    {
        this->copy_to(dst, this->size());
    }
};

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

template<typename Arithmetic, std::enable_if_t<std::is_arithmetic_v<Arithmetic>, int> = 0>
constexpr inline auto make_range_if_arithmetic(Arithmetic&& arg)
{
    return make_range_view(int(0), arg);
}

template<typename NonArithmetic, std::enable_if_t<!std::is_arithmetic_v<NonArithmetic>, int> = 0>
constexpr inline auto make_range_if_arithmetic(NonArithmetic&& arg)
{
    return arg;
}

template<typename T, bool IsUnitStep>
inline auto make_array(const range_view<T, IsUnitStep>& range)
{
    using elem_t = T;
    std::vector<elem_t> data(range.size());
    range.copy_to(data.begin());
    return make_array(std::move(data));
}

}
