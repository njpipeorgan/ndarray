#pragma once

#include "decls.h"
#include "traits.h"
#include "array_view.h"

namespace ndarray
{

//
// _irregular_indices is a helper class, stores a const reference to 
// a base view (for dimensions) and an array of indices, implementing 
// random iterator functionalities that operate on indices. 
//
// Arithmetic operations have complexity O(1). 
// get_base_ptr and get_element_ref have complexity O(_indices_depth_v).
//

template<typename ViewCref, size_t IndicesDepth>
class _irregular_indices
{
public:
    using _my_type     = _irregular_indices;
    using _view_cref_t = ViewCref;
    static constexpr int _indices_depth_v = IndicesDepth;
    using _indices_t   = std::array<size_t, _indices_depth_v>;

protected:
    _view_cref_t view_cref_;
    _indices_t   indices_{}; // zeros by default

public:
    explicit _irregular_indices(_view_cref_t view_cref, _indices_t indices) : 
        view_cref_{view_cref}, indices_{indices} {}

    explicit  _irregular_indices(_view_cref_t view_cref) : 
        view_cref_{view_cref} {}

    // get a reference to indices
    _indices_t& get()
    {
        return indices_;
    }

    // using the given stride to calculate the base pointer of the subview
    auto* get_base_ptr(size_t stride) const
    {
        size_t pos = view_cref_._get_position(indices_);
        return view_cref_._get_base_ptr() + pos * stride;
    }
    
    // accessing the element the correspondes to the indices
    auto& get_element_ref() const
    {
        return view_cref_.tuple_at(indices_);
    }

    // as operator++(Diff)
    template<typename Diff>
    void inc(Diff diff)
    {
        if constexpr (std::is_unsigned_v<Diff>)
            index_explicit_inc(diff);
        else if (diff >= 0)
            index_explicit_inc(diff);
        else
            index_explicit_dec(-diff);
        return *this;
    }
    
    // as operator--(Diff)
    template<typename Diff>
    void dec(Diff diff)
    {
        if constexpr (std::is_unsigned_v<Diff>)
            index_explicit_dec(diff);
        else if (diff >= 0)
            index_explicit_dec(diff);
        else
            index_explicit_inc(-diff);
        return *this;
    }

    // as operator++()
    void inc()
    {
        explicit_inc();
    }

    // as operator--()
    void dec()
    {
        explicit_dec();
    }

    // increment by 1 on Level
    template<size_t Level = _indices_depth_v - 1>
    void explicit_inc()
    {
        const size_t dim = view_cref_.dimension<Level>();
        indices_[Level]++;
        if constexpr (Level > 0)
        {
            if (indices_[Level] == dim) // if did overflow
            {
                indices_[Level] = 0;
                explicit_inc<Level - 1>(); // carry
            }
        }
    }

    // increment by diff (diff >= 0) on Level
    template<size_t Level = _indices_depth_v - 1>
    void explicit_inc(size_t diff)
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
                explicit_inc<Level - 1>(); // carry
            }
            else // if did overflow more than once
            {
                size_t val  = indices_[Level];
                size_t quot = val / dim;
                size_t rem  = val % dim;
                indices_[Level] = rem;
                explicit_inc<Level - 1>(quot); // carry
            }
        }
    }

    // decrement by 1 on Level
    template<size_t Level = _indices_depth_v - 1>
    void explicit_dec()
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
                explicit_dec<Level - 1>(); // borrow
            }
        }
        else
        {
            indices_[Level]--;
        }
    }

    // decrement by diff (diff >= 0) on Level
    template<size_t Level = _indices_depth_v - 1>
    void explicit_dec(size_t diff)
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
                explicit_dec<Level - 1>(); // carry
            }
            else // if underflow more than once
            {
                size_t val  = size_t(-post_sub) - 1;
                size_t quot = val / dim;
                size_t rem  = val % dim;
                indices_[Level] = dim - (rem + 1);
                explicit_dec<Level - 1>(quot);
            }
        }
        else
        {
            indices_[Level] = size_t(post_sub);
        }
    }

    // calculate the diff, where indices1 = indices2 + diff
    template<size_t Level = _indices_depth_v - 1>
    ptrdiff_t difference(const _my_type& other) const
    {
        ptrdiff_t    diff = this->indices_[Level] - other.indices_[Level];
        const size_t dim  = view_cref_.dimension<Level>();
        if constexpr (Level == 0)
            return diff;
        else
            return dim * difference<Level - 1>(other) + diff;
    }

    template<size_t Level = 0>
    bool cmp_equal(const _my_type& other) const
    {
        auto l = Level;
        bool equal = (this->indices_[Level] == other.indices_[Level]);
        if constexpr (Level + 1 < _indices_depth_v)
            return equal && cmp_equal<Level + 1>(other);
        else
            return equal;
    }

    template<size_t Level = 0>
    bool cmp_less(const _my_type& other) const
    {
        if constexpr (Level + 1 < _indices_depth_v)
        {
            ptrdiff_t diff = this->indices_[Level] - other.indices_[Level];
            if (diff < 0)
                return true;
            else if (diff > 0)
                return false;
            else
                return cmp_less<Level + 1>(other);
        }
        else
        {
            return this->indices_[Level] < other.indices_[Level];
        }
    }

    template<size_t Level = 0>
    bool cmp_greater(const _my_type& other) const
    {
        if constexpr (Level + 1 < _indices_depth_v)
        {
            ptrdiff_t diff = this->indices_[Level] - other.indices_[Level];
            if (diff > 0)
                return true;
            else if (diff < 0)
                return false;
            else
                return cmp_greater<Level + 1>(other);
        }
        else
        {
            return this->indices_[Level] > other.indices_[Level];
        }
    }


};


//
// _regular_view_iter and _irregular_view_iter is the generalized version
// of iterators for multi-dimensional arrays. They allow iterations over 
// various levels. 
//
// View iterators stores an sub view of the base view that can be copied, 
// or returned as const reference. Regular iterators only describe iterators 
// with fixed stride, and use the base pointer in the sub view as the 
// indicator of position. Irregular iterators stores extra indices as the 
// position, and update the base pointer when necessary.
//
// Arithmetic operations on _irregular_view_iter have the same complexities 
// as the helper class _irregular_indices's.
//

template<typename SubView, bool IsExplicitConst>
class _regular_view_iter
{
public:
    using _my_type      = _regular_view_iter;
    using _sub_view_t   = SubView;
    using _elem_t       = typename _sub_view_t::_elem_t;
    static constexpr bool _is_const_v = IsExplicitConst || std::is_const_v<_elem_t>;
    using _ret_view_t   = std::conditional_t<_is_const_v, typename _sub_view_t::_my_const_t, _sub_view_t>;
    using _base_ptr_t   = typename _ret_view_t::_base_ptr_t;
    using _ptr_stride_t = size_t;
    using _my_const_t   = _regular_view_iter<_sub_view_t, true>;

public:
    _ret_view_t   ret_view_;
    _ptr_stride_t ptr_stride_;  // always positive

public:
    explicit _regular_view_iter(_sub_view_t sub_view, _ptr_stride_t ptr_stride) : 
        ret_view_{sub_view}, ptr_stride_{ptr_stride} {}

    _base_ptr_t& my_base_ptr_ref()
    {
        return ret_view_._get_base_ptr_ref();
    }

    _my_type& operator+=(ptrdiff_t diff)
    {
        my_base_ptr_ref() += diff * ptr_stride_;
        return *this;
    }
    _my_type& operator-=(ptrdiff_t diff)
    {
        my_base_ptr_ref() -= diff * ptr_stride_;
        return *this;
    }
    _my_type operator+(ptrdiff_t diff) const
    {
        _my_type ret = *this;
        ret += diff;
        return ret;
    }
    _my_type operator-(ptrdiff_t diff) const
    {
        _my_type ret = *this;
        ret -= diff;
        return ret;
    }
    _my_type& operator++()
    {
        (*this) += 1;
        return *this;
    }
    _my_type& operator--()
    {
        (*this) += 1;
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

    const _ret_view_t& operator*() const
    {
        return ret_view_;
    }
    _ret_view_t operator[](ptrdiff_t diff) const
    {
        return *((*this) + diff);
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return (this->my_base_ptr_ref() - other.my_base_ptr_ref()) / ptr_stride_;
    }

    bool operator==(const _my_type& other) const
    {
        return this->my_base_ptr_ref() == other.my_base_ptr_ref();
    }
    bool operator!=(const _my_type& other) const
    {
        return !((*this) == other);
    }
    bool operator<(const _my_type& other) const
    {
        return this->my_base_ptr_ref() < other.my_base_ptr_ref();
    }
    bool operator>(const _my_type& other) const
    {
        return this->my_base_ptr_ref() > other.my_base_ptr_ref();
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

template<typename SubView, typename BaseView, bool IsExplicitConst>
class _irregular_view_iter
{
public:
    using _my_type      = _irregular_view_iter;
    using _sub_view_t   = SubView;
    using _base_view_t  = BaseView;
    using _view_cref_t  = const _base_view_t&;
    using _elem_t       = typename _sub_view_t::_elem_t;
    static constexpr bool _is_const_v = IsExplicitConst || std::is_const_v<_elem_t>;
    using _ret_view_t   = std::conditional_t<_is_const_v, typename _sub_view_t::_my_const_t, _sub_view_t>;
    using _base_ptr_t   = typename _ret_view_t::_base_ptr_t;
    using _ptr_stride_t = size_t;
    static constexpr size_t _iter_depth_v = _base_view_t::_depth_v - _sub_view_t::_depth_v;
    using _indices_t    = _irregular_indices<_view_cref_t, _iter_depth_v>;
    using _indices_array_t = std::array<size_t, _iter_depth_v>;
    using _my_const_t   = _irregular_view_iter<_sub_view_t, _base_view_t, true>;

public:
    _indices_t          indices_;    // zeros by default internally
    mutable _ret_view_t ret_view_;
    _ptr_stride_t       ptr_stride_; // always positive

public:
    explicit _irregular_view_iter(_view_cref_t base_view_cref, _sub_view_t sub_view, _ptr_stride_t ptr_stride) : 
        indices_{base_view_cref}, ret_view_{sub_view}, ptr_stride_{ptr_stride} {}
    
    auto& _get_indices_ref()
    {
        return indices_.get();
    }

    template<size_t IterLevel>
    size_t _index_dimension() const
    {
        static_assert(IterLevel < _iter_depth_v);
        return base_view_cref_.dimension<IterLevel>();
    }

    void _update_base_ptr() const
    {
        ret_view_._get_base_ptr_ref() = indices_.get_base_ptr(ptr_stride_);
    }

    template<typename Diff>
    _my_type& operator+=(Diff diff)
    {
        indices_.inc(diff);
        return *this;
    }
    template<typename Diff>
    _my_type& operator-=(Diff diff)
    {
        indices_.dec(diff);
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
        indices_.inc();
        return *this;
    }
    _my_type& operator--()
    {
        indices_.dec();
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

    const _ret_view_t& operator*() const
    {
        this->_update_base_ptr();
        return ret_view_;
    }
    _ret_view_t operator[](ptrdiff_t diff) const
    {
        auto new_indices = this->indices_;
        new_indices += diff;
        ret_view._get_base_ptr_ref() = new_indices.get_base_ptr(ptr_stride_);
        return ret_view;
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return this->indices_.difference(other.indices_);
    }

    bool operator==(const _my_type& other) const
    {
        return indices_.cmp_equal(other.indices_);
    }
    bool operator<(const _my_type& other) const
    {
        return indices_.cmp_less(other.indices_);
    }
    bool operator>(const _my_type& other) const
    {
        return indices_.cmp_greater(other.indices_);
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


//
// Array view element iterator, iterates all elements in accessing order.
//
// The constness of the elements accessed through element iterators are 
// controlled by both their source array views and IsExplicitConst argument.
//
// static constexpr bool _is_const_v indicates this constness. 
//
// _simple_elem_iter and _regular_elem_iter stores the pointer to the 
// elements and access them by patterns. 
//
// Arithmetic operations on _irregular_elem_iter have the same complexities 
// as the helper class _irregular_indices's.
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
    explicit _simple_elem_iter() = default;
    explicit _simple_elem_iter(_elem_ptr_t ptr) :
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
    explicit _regular_elem_iter() = default;
    explicit _regular_elem_iter(_elem_ptr_t ptr, _stride_t stride) :
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
    using _my_type     = _irregular_elem_iter;
    using _view_t      = View;
    using _view_cref_t = const _view_t&;
    using _elem_t      = typename _view_t::_elem_t;
    static constexpr size_t _depth_v    = _view_t::_depth_v;
    static constexpr bool   _is_const_v = IsExplicitConst || std::is_const_v<_elem_t>;
    using _indices_t   = _irregular_indices<_view_cref_t, _depth_v>;
    using _indices_array_t = std::array<size_t, _depth_v>;
    using _ret_elem_t  = std::conditional_t<_is_const_v, std::add_const_t<_elem_t>, _elem_t>;

protected:
    _indices_t indices_;

public:
    explicit _irregular_elem_iter(_view_cref_t view_cref, _indices_array_t indices_array) :
        indices_{view_cref, indices_array} {}

    template<typename Diff>
    _my_type& operator+=(Diff diff)
    {
        indices_.inc(diff);
        return *this;
    }
    template<typename Diff>
    _my_type& operator-=(Diff diff)
    {
        indices_.dec(diff);
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
        indices_.inc();
        return *this;
    }
    _my_type& operator--()
    {
        indices_.dec();
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
        return indices_.get_element_ref();
    }
    template<typename Diff>
    _ret_elem_t& operator[](Diff diff) const
    {
        return *((*this) + diff);
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return indices_.difference(other.indices_);
    }

    bool operator==(const _my_type& other) const
    {
        return indices_.cmp_equal(other.indices_);
    }
    bool operator<(const _my_type& other) const
    {
        return indices_.cmp_less(other.indices_);
    }
    bool operator>(const _my_type& other) const
    {
        return indices_.cmp_greater(other.indices_);
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


}
