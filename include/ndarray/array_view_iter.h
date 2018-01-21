#pragma once

#include "decls.h"
#include "traits.h"
#include "array_view.h"

namespace ndarray
{

//
// Array view element iterator, iterates all elements in accessing order.
//
// The constness of the elements accessed through element iterators are 
// controlled by both their source array views and IsExplicitConst argument.
//
// static constexpr bool _is_const_v indicates this constness. 
//
// _simple_elem_iter and _regular_elem_iter stores the pointer to the 
// elements and access them by patterns. _irregular_elem_iter stores an
// array of indices and access elements by calling _view_t::tuple_at(...)
//
// Arithmetic operations on _irregular_elem_iter have time complexities of
// O(_depth_v) expect for operator++/operator--.
//

template<typename T, bool IsExplicitConst>
class _simple_elem_iter
{
public:
    using _my_type    = _simple_elem_iter;
    using _elem_t     = T;
    using _elem_ptr_t = T*;
    static constexpr bool _is_const_v = IsExplicitConst || std::is_const_v<_elem_t>;
    using _ret_elem_t = std::conditional_t<_is_const_v, std::add_const_t<_elem_t>, _elem_t>;

protected:
    _elem_ptr_t ptr_{nullptr};

public:
    _simple_elem_iter() = default;
    _simple_elem_iter(_elem_ptr_t ptr) :
        ptr_{ptr} {}

    _my_type& operator+=(ptrdiff_t diff)
    {
        ptr_ += diff;
        return *this;
    }
    _my_type& operator-=(ptrdiff_t diff)
    {
        ptr_ -= diff;
        return *this;
    }
    _my_type operator+(ptrdiff_t diff) const
    {
        return _my_type{ptr_ + diff};
    }
    _my_type operator-(ptrdiff_t diff) const
    {
        return _my_type{ptr_ - diff};
    }
    _my_type& operator++()
    {
        *this += 1;
        return *this;
    }
    _my_type& operator--()
    {
        *this -= 1;
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

    _ret_elem_t& operator*() const
    {
        return *ptr_;
    }
    _ret_elem_t& operator[](ptrdiff_t diff) const
    {
        return *(ptr_ + diff);
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return this->ptr_ - other.ptr_;
    }

    bool operator==(const _my_type& other) const
    {
        return this->ptr_ == other.ptr_;
    }
    bool operator<(const _my_type& other) const
    {
        return this->ptr_ < other.ptr_;
    }
    bool operator>(const _my_type& other) const
    {
        return this->ptr_ > other.ptr_;
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

template<typename T, bool IsExplicitConst>
class _regular_elem_iter
{
public:
    using _my_type    = _regular_elem_iter;
    using _elem_t     = T;
    using _elem_ptr_t = T*;
    using _stride_t   = ptrdiff_t;
    static constexpr bool _is_const_v = IsExplicitConst || std::is_const_v<_elem_t>;
    using _ret_elem_t = std::conditional_t<_is_const_v, std::add_const_t<_elem_t>, _elem_t>;

protected:
    _elem_ptr_t ptr_{nullptr};
    _stride_t   stride_{1};

public:
    _regular_elem_iter() = default;
    _regular_elem_iter(_elem_ptr_t ptr, _stride_t stride) :
        ptr_{ptr}, stride_{stride}
    {
        NDARRAY_ASSERT(stride_ != 0);
    }

    _my_type& operator+=(ptrdiff_t diff)
    {
        ptr_ += diff * stride_;
        return *this;
    }
    _my_type& operator-=(ptrdiff_t diff)
    {
        ptr_ -= diff * stride_;
        return *this;
    }
    _my_type operator+(ptrdiff_t diff) const
    {
        return _my_type{ptr_ + diff * stride_, stride_};
    }
    _my_type operator-(ptrdiff_t diff) const
    {
        return _my_type{ptr_ - diff * stride_, stride_};
    }
    _my_type& operator++()
    {
        *this += stride_;
        return *this;
    }
    _my_type& operator--()
    {
        *this -= stride_;
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

    _ret_elem_t& operator*() const
    {
        return *ptr_;
    }
    _ret_elem_t& operator[](ptrdiff_t diff) const
    {
        return *(ptr_ + diff * stride_);
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return (this->ptr_ - other.ptr_) / stride_;
    }

    bool operator==(const _my_type& other) const
    {
        return this->ptr_ == other.ptr_;
    }
    bool operator<(const _my_type& other) const
    {
        return (this->ptr_ < other.ptr_) ^ (stride_ > 0);
    }
    bool operator>(const _my_type& other) const
    {
        return (this->ptr_ > other.ptr_) ^ (stride_ > 0);
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

template<typename View, bool IsExplicitConst>
class _irregular_elem_iter
{
public:
    using _my_type   = _irregular_elem_iter;
    using _view_t    = View;
    using _elem_t    = typename _view_t::_elem_t;
    static constexpr size_t _depth_v    = _view_t::_depth_v;
    static constexpr bool   _is_const_v = IsExplicitConst || std::is_const_v<_elem_t>;
    using _indices_t = std::array<size_t, _depth_v>;
    using _ret_elem_t = std::conditional_t<_is_const_v, std::add_const_t<_elem_t>, _elem_t>;

protected:
    const _view_t& view_cref_;
    _indices_t     indices_{};

public:
    _irregular_elem_iter(const _view_t& view_cref, _indices_t indices) :
        view_cref_{view_cref}, indices_{indices} {}

    template<typename Diff>
    _my_type& operator+=(Diff diff)
    {
        if constexpr (std::is_unsigned_v<Diff>)
            index_explicit_inc(diff);
        else if (diff >= 0)
            index_explicit_inc(diff);
        else
            index_explicit_dec(-diff);
        return *this;
    }
    template<typename Diff>
    _my_type& operator-=(Diff diff)
    {
        if constexpr (std::is_unsigned_v<Diff>)
            index_explicit_dec(diff);
        else if (diff >= 0)
            index_explicit_dec(diff);
        else
            index_explicit_inc(-diff);
        return *this;
    }
    template<typename Diff>
    _my_type operator+(Diff diff) const
    {
        _my_type ret = *this;
        ret += diff;
        return ret;
    }
    template<typename Diff>
    _my_type operator-(Diff diff) const
    {
        _my_type ret = *this;
        ret -= diff;
        return ret;
    }
    _my_type& operator++()
    {
        index_explicit_inc();
        return *this;
    }
    _my_type& operator--()
    {
        index_explicit_dec();
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

    _ret_elem_t& operator*() const
    {
        return view_cref_.tuple_at(indices_);
    }
    template<typename Diff>
    _ret_elem_t& operator[](Diff diff) const
    {
        return *((*this) + diff);
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return index_difference(this->indices_, other.indices_);
    }

    bool operator==(const _my_type& other) const
    {
        return index_cmp_equal(this->indices_, other.indices_);
    }
    bool operator<(const _my_type& other) const
    {
        return index_cmp_less(this->indices_, other.indices_);
    }
    bool operator>(const _my_type& other) const
    {
        return index_cmp_greater(this->indices_, other.indices_);
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

protected:
    // increment by 1 on Level
    template<size_t Level = _depth_v - 1>
    inline void index_explicit_inc()
    {
        const size_t dim = view_cref_.dimension<Level>();
        indices_[Level]++;
        if constexpr (Level > 0)
        {
            if (indices_[Level] == dim) // if did overflow
            {
                indices_[Level] = 0;
                index_explicit_inc<Level - 1>(); // carry
            }
        }
    }

    // increment by diff (diff >= 0) on Level
    template<size_t Level = _depth_v - 1>
    inline void index_explicit_inc(size_t diff)
    {
        const size_t dim = view_cref_.dimension<Level>();
        indices_[Level] += diff;
        if constexpr (Level > 0)
        {
            if (indices_[Level] < dim) // if did not overflow
            { // do nothing
            }
            else if (indices_[Level] < 2 * dim) // if did overflow only once
            {
                indices_[Level] -= dim;
                index_explicit_inc<Level - 1>(); // carry
            }
            else // if did overflow more than once
            {
                size_t val  = indices_[Level];
                size_t quot = val / dim;
                size_t rem  = val % dim;
                indices_[Level] = rem;
                index_explicit_inc<Level - 1>(quot); // carry
            }
        }
    }

    // decrement by 1 on Level
    template<size_t Level = _depth_v - 1>
    inline void index_explicit_dec()
    {
        const size_t dim = view_cref_.dimension<Level>();
        if constexpr (Level > 0)
        {
            if (indices_[Level] > 0) // if will not underflow
            {
                indices_[Level]--;
            }
            else
            {
                indices_[Level] = dim - 1;
                index_explicit_dec<Level - 1>(); // borrow
            }
        }
        else
        {
            indices_[Level]--;
        }
    }

    // decrement by diff (diff >= 0) on Level
    template<size_t Level = _depth_v - 1>
    inline void index_explicit_dec(size_t diff)
    {
        const size_t dim = view_cref_.dimension<Level>();
        ptrdiff_t post_sub = indices_[Level] - diff;
        if constexpr (Level > 0)
        {
            if (post_sub >= 0) // if will not underflow
            {
                indices_[Level] -= diff;
            }
            else if (size_t(-post_sub) <= dim) // if will underflow only once
            {
                indices_[Level] -= (diff - dim);
                index_explicit_dec<Level - 1>(); // carry
            }
            else // if underflow more than once
            {
                size_t val  = size_t(-post_sub) - 1;
                size_t quot = val / dim;
                size_t rem  = val % dim;
                indices_[Level] = dim - (rem + 1);
                index_explicit_dec<Level - 1>(quot);
            }
        }
        else
        {
            indices_[Level] = size_t(post_sub);
        }
    }

    // calculate the diff, where indices1 = indices2 + diff
    template<size_t Level = _depth_v - 1>
    inline ptrdiff_t index_difference(const _indices_t& indices1, const _indices_t& indices2) const
    {
        ptrdiff_t    diff = indices1[Level] - indices2[Level];
        const size_t dim  = view_cref_.dimension<Level>();
        if constexpr (Level == 0)
            return diff;
        else
            return dim * index_difference<Level - 1>(indices1, indices2) + diff;
    }

    template<size_t Level = 0>
    inline bool index_cmp_equal(const _indices_t& indices1, const _indices_t& indices2) const
    {
        auto l = Level;
        bool equal = (indices1[Level] == indices2[Level]);
        if constexpr (Level + 1 < _depth_v)
            return equal && index_cmp_equal<Level + 1>(indices1, indices2);
        else
            return equal;
    }

    template<size_t Level = 0>
    inline bool index_cmp_less(const _indices_t& indices1, const _indices_t& indices2) const
    {
        ptrdiff_t diff = indices1[Level] - indices2[Level];
        if constexpr (Level + 1 < _depth_v)
        {
            if (diff < 0)
                return true;
            else if (diff > 0)
                return false;
            else
                return index_cmp_less<Level + 1>(indices1, indices2);
        }
        else
        {
            return indices1[Level] < indices2[Level];
        }
    }

    template<size_t Level = 0>
    inline bool index_cmp_greater(const _indices_t& indices1, const _indices_t& indices2) const
    {
        ptrdiff_t diff = indices1[Level] - indices2[Level];
        if constexpr (Level + 1 < _depth_v)
        {
            if (diff > 0)
                return true;
            else if (diff < 0)
                return false;
            else
                return index_cmp_greater<Level + 1>(indices1, indices2);
        }
        else
        {
            return indices1[Level] > indices2[Level];
        }
    }

};


}
