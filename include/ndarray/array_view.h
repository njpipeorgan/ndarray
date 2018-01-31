#pragma once

#include <type_traits>

#include "decls.h"
#include "traits.h"
#include "indexer.h"

namespace ndarray
{

//
// simple_view : 
//   Elements in the view are stored continuously in the base array.
//   e.g. {all, all, all}, {scalar, simple, all}, {scalar, simple}
//
// regular_view : 
//   Elements in the view are split by a fixed stride in the base array.
//   e.g. {all, all, scalar}, {simple, all, scalar}, {scalar, regular}
//
// irregular_view : 
//   Elements in the view are stored irregularly in the base array.
//   e.g. {all, scalar, all}, {regular, all}, {irregular, scalar}
// 
//
// View type are derived by _derive_view_type_t based on indexer types.
//
// typename T (_elem_t) reflects the constness of elements in the view, 
// which is automatically derived from the constness of the base array.
//
// _elem_t* base_ptr indicates the starting point of element accessing, 
// and does not necessarily point to the first element.
// 
// const size_t* base_dims_ are used to identify whether two views are 
// extracted from the same base array.
//

template<typename T, typename IndexerTuple>
class array_view_base
{
public:
    using _elem_t          = T;
    using _no_const_elem_t = std::remove_const_t<_elem_t>;
    using _const_elem_t    = std::add_const_t<_elem_t>;
    using _indexers_t      = IndexerTuple;
    using _base_ptr_t      = _elem_t*;
    using _base_dims_t     = const size_t*;
    static constexpr bool   _is_const_v        = std::is_const_v<_elem_t>;
    static constexpr size_t _depth_v           = _indexer_tuple_depth_v<_indexers_t>;
    static constexpr size_t _base_depth_v      = std::tuple_size_v<_indexers_t>;
    static constexpr std::array<size_t, _depth_v>
        _non_scalar_indexers_table = _make_non_scalar_indexer_table<_indexers_t>();
    static constexpr size_t _stride_depth_v    = _non_scalar_indexers_table[_depth_v - 1] + 1;
    static constexpr bool   _has_base_stride_v = (_stride_depth_v != _base_depth_v);
    using _base_stride_t   = std::conditional_t<_has_base_stride_v, size_t, _empty_struct>;

public:
    _base_ptr_t          base_ptr_;       // the base pointer for element accessing
    const _base_dims_t   base_dims_;      // dimensions of the base array, also used to identify the base array
    const _indexers_t    indexers_;       // stores all indexers
    const _base_stride_t base_stride_;    // base stride between elements

public:
    explicit array_view_base(_base_ptr_t base_ptr, _base_dims_t base_dims,
                             _indexers_t indexers, size_t base_stride ={}) :
        base_ptr_{base_ptr}, base_dims_{base_dims},
        indexers_{indexers}, base_stride_{base_stride} {}

    _base_ptr_t _get_base_ptr() const
    {
        return base_ptr_;
    }
    _base_ptr_t& _get_base_ptr_ref()
    {
        return base_ptr_;
    }
    const _base_ptr_t& _get_base_ptr_ref() const
    {
        return base_ptr_;
    }

    const size_t* _identifier_ptr() const
    {
        return base_dims_;
    }

    // access i-th indexer from indexer tuple
    template<size_t BaseLevel>
    decltype(auto) _base_indexer() const noexcept
    {
        return std::get<BaseLevel>(indexers_);
    }

    // access i-th non-scalar indexer from indexer tuple
    template<size_t Level>
    decltype(auto) _level_indexer() const noexcept
    {
        static_assert(Level < _depth_v);
        return _base_indexer<_non_scalar_indexers_table[Level]>();
    }

    // base dimension of the view on the i-th base level
    template<size_t I>
    size_t _base_dimension() const
    {
        static_assert(I < _base_depth_v);
        return base_dims_[I];
    }

    // dimension of the view on the i-th level
    template<size_t I>
    size_t dimension() const
    {
        size_t bdim_i = _base_dimension<_non_scalar_indexers_table[I]>();
        return _level_indexer<I>().size(bdim_i);
    }

    // array of dimensions
    std::array<size_t, _depth_v> dimensions() const
    {
        std::array<size_t, _depth_v> dims;
        _dimensions_impl(dims.data());
        return dims;
    }

    // total size of the view
    template<size_t LastLevel = _depth_v, size_t FirstLevel = 0>
    size_t total_size() const
    {
        static_assert(FirstLevel <= LastLevel && LastLevel <= _depth_v);
        if constexpr (FirstLevel == LastLevel)
            return size_t(1);
        else
            return dimension<LastLevel - 1>() * total_size<LastLevel - 1, FirstLevel>();
    }

    // indexing with a tuple/array of integers
    template<typename Tuple>
    _elem_t& tuple_at(const Tuple& indices) const
    {
        static_assert(std::tuple_size_v<Tuple> == _depth_v, "incorrect number of indices");
        return _base_at(_get_position(indices));
    }

    // indexing with multiple integers
    template<typename... Ints>
    _elem_t& at(Ints... ints) const
    {
        return tuple_at(std::make_tuple(ints...));
    }

    template<typename SpanTuple>
    _derive_view_type_t<_elem_t, _indexers_t, SpanTuple>
        tuple_part_view(SpanTuple&& spans) const
    {
        return get_collapsed_view(
            base_ptr_, base_dims_, indexers_, std::forward<decltype(spans)>(spans));
    }

    template<typename... Spans>
    _derive_view_type_t<_elem_t, _indexers_t, std::tuple<Spans...>>
        part_view(Spans&&... spans) const
    {
        return tuple_part_view(std::forward_as_tuple(spans...));
    }

    // check whether having same dimensions with another array, starting at specific levels
    template<size_t MyStartLevel = 0, size_t OtherStartLevel = 0, typename OtherArray>
    bool has_same_dimensions(const OtherArray& other) const
    {
        if constexpr (MyStartLevel == _depth_v || OtherStartLevel == OtherArray::_depth_v)
            return false;
        else if constexpr (MyStartLevel == _depth_v - 1 && OtherStartLevel == OtherArray::_depth_v - 1)
            return this->dimension<MyStartLevel>() == other.dimension<OtherStartLevel>();
        else
            return this->dimension<MyStartLevel>() == other.dimension<OtherStartLevel>() && 
            has_same_dimensions<MyStartLevel + 1, OtherStartLevel + 1>(other);
    }

public:

    template<size_t LastBaseLevel = _base_depth_v, size_t FirstBaseLevel = 0>
    size_t _total_base_size() const
    {
        static_assert(FirstBaseLevel <= LastBaseLevel && LastBaseLevel <= _base_depth_v);
        if constexpr (FirstBaseLevel == LastBaseLevel)
            return size_t(1);
        else
            return _base_dimension<LastBaseLevel - 1>() * _total_base_size<LastBaseLevel - 1, FirstBaseLevel>();
    }
    
    _elem_t& _base_at(size_t pos) const
    {
        if constexpr (!_has_base_stride_v)
            return base_ptr_[pos];
        else
            return base_ptr_[pos * base_stride_];
    }

    template<typename Tuple, size_t LC = std::tuple_size_v<Tuple> -1, size_t BC = _non_scalar_indexers_table[LC]>
    size_t _get_position(const Tuple& tuple) const
    {
        if constexpr (_non_scalar_indexers_table[LC] == BC)
        {
            size_t dim_i  = dimension<LC>();
            size_t bdim_i = _base_dimension<BC>();
            size_t pos_i  = _add_if_negative<size_t>(std::get<LC>(tuple), dim_i);
            NDARRAY_ASSERT(pos_i < dim_i);
            size_t bpos_i = _level_indexer<LC>()[pos_i];
            if constexpr (LC == 0)
                return bpos_i;
            else
                return bpos_i + bdim_i * _get_position<Tuple, LC - 1, BC - 1>(tuple);
        }
        else // _non_scalar_indexers_table[LC] != BC
        {
            size_t bdim_i = _base_dimension<BC>();
            return bdim_i * _get_position<Tuple, LC, BC - 1>(tuple);
        }
    }

    template<size_t Level = 0>
    void _dimensions_impl(size_t* dims) const
    {
        dims[Level] = this->dimension<Level>();
        if constexpr (Level + 1 < _depth_v)
            _dimensions_impl<Level + 1>(dims);
    }

};

template<typename T, typename IndexerTuple>
class simple_view : public array_view_base<T, IndexerTuple>
{
public:
    using _my_type         = simple_view;
    using _my_base         = array_view_base<T, IndexerTuple>;
    using _elem_t          = typename _my_base::_elem_t;
    using _no_const_elem_t = typename _my_base::_no_const_elem_t;
    using _indexers_t      = typename _my_base::_indexers_t;
    using _base_ptr_t      = typename _my_base::_base_ptr_t;
    using _base_dims_t     = typename _my_base::_base_dims_t;
    using _my_const_t      = simple_view<const _elem_t, _indexers_t>;
    static constexpr bool   _is_const_v   = _my_base::_is_const_v;
    static constexpr size_t _depth_v      = _my_base::_depth_v;
    static constexpr size_t _base_depth_v = _my_base::_base_depth_v;
    static constexpr _view_type _my_view_type_v = _view_type::simple;

public:
    explicit simple_view(_base_ptr_t base_ptr, _base_dims_t base_dims,
                         _indexers_t indexers, size_t) :
        _my_base{base_ptr, base_dims, indexers} {}

    template<typename Other>
    _my_type& operator=(Other&& other)
    {
        data_copy(std::forward<Other>(other), *this);
        return *this;
    }

    ptrdiff_t stride() const noexcept
    {
        return 1;
    }

    _simple_elem_iter<_elem_t> element_begin() const
    {
        return {this->base_ptr_};
    }
    _simple_elem_iter<_elem_t> element_end() const
    {
        return {this->base_ptr_ + this->total_size()};
    }
    _simple_elem_const_iter<_elem_t> element_cbegin() const
    {
        return {this->base_ptr_};
    }
    _simple_elem_const_iter<_elem_t> element_cend() const
    {
        return {this->base_ptr_ + this->total_size()};
    }

    template<bool IsExplicitConst, size_t Level>
    auto _begin_impl() const
    {
        static_assert(0 < Level && Level <= _depth_v);
        if constexpr (Level == _depth_v)
        {
            return this->element_begin();
        }
        else
        {
            size_t ptr_stride = this->_total_base_size<
                _base_depth_v, this->_non_scalar_indexers_table[Level - 1] + 1>();
            auto   sub_view   = this->tuple_part_view(_repeat_tuple_t<Level, size_t>{});
            return _regular_view_iter<decltype(sub_view), IsExplicitConst>{std::move(sub_view), ptr_stride};
        }
    }
    template<bool IsExplicitConst, size_t Level>
    auto _end_impl() const
    {
        static_assert(0 < Level && Level <= _depth_v);
        if constexpr (Level == _depth_v)
        {
            return this->element_end();
        }
        else
        {
            auto iter = this->_begin_impl<IsExplicitConst, Level>();
            iter += this->total_size<Level>();
            return iter;
        }
    }

    template<size_t Level = 1>
    auto begin() const
    {
        return _begin_impl<false, Level>();
    }
    template<size_t Level = 1>
    auto end() const
    {
        return _end_impl<false, Level>();
    }
    template<size_t Level = 1>
    auto cbegin() const
    {
        return _begin_impl<true, Level>();
    }
    template<size_t Level = 1>
    auto cend() const
    {
        return _end_impl<true, Level>();
    }

    // for each element in the view, call fn(element) in order
    template<typename Function>
    void traverse(Function fn) const
    {
        const size_t size = this->total_size();
        for (size_t i = 0; i < size; ++i)
            fn(*(this->base_ptr_ + i));
    }

    // copy data to destination given size, assuming no aliasing
    template<typename Iter>
    void copy_to(Iter dst, size_t size) const
    {
        auto src = this->base_ptr_;
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
        this->copy_to(dst, this->total_size());
    }

    // copy data from source given size, assuming no aliasing
    template<typename Iter>
    void copy_from(Iter src, size_t size) const
    {
        static_assert(!_is_const_v);
        auto dst = this->base_ptr_;
        for (size_t i = 0; i < size; ++i)
        {
            *dst = *src;
            ++dst;
            ++src;
        }
    }

    // copy data from source, assuming no aliasing
    template<typename Iter>
    void copy_from(Iter src) const
    {
        this->copy_from(src, this->total_size());
    }

};

template<typename T, typename IndexerTuple>
class regular_view : public array_view_base<T, IndexerTuple>
{
public:
    using _my_type         = regular_view;
    using _my_base         = array_view_base<T, IndexerTuple>;
    using _elem_t          = typename _my_base::_elem_t;
    using _no_const_elem_t = typename _my_base::_no_const_elem_t;
    using _indexers_t      = typename _my_base::_indexers_t;
    using _base_ptr_t      = typename _my_base::_base_ptr_t;
    using _base_dims_t     = typename _my_base::_base_dims_t;
    using _base_stride_t   = typename _my_base::_base_stride_t;
    using _my_const_t      = regular_view<const _elem_t, _indexers_t>;
    static constexpr bool   _is_const_v   = _my_base::_is_const_v;
    static constexpr size_t _depth_v      = _my_base::_depth_v;
    static constexpr size_t _base_depth_v = _my_base::_base_depth_v;
    static constexpr _view_type _my_view_type_v = _view_type::regular;

public:
    explicit regular_view(_base_ptr_t base_ptr, _base_dims_t base_dims,
                          _indexers_t indexers, size_t base_stride) :
        _my_base{base_ptr, base_dims, indexers, base_stride} {}

    template<typename Other>
    _my_type& operator=(Other&& other)
    {
        data_copy(std::forward<Other>(other), *this);
        return *this;
    }

    ptrdiff_t stride() const noexcept
    {
        if constexpr (this->_depth_v == 1 && this->_has_base_stride_v)
            return this->base_stride_ * this->_level_indexer<0>().step();
        if constexpr (this->_depth_v != 1 && this->_has_base_stride_v)
            return this->base_stride_;
        if constexpr (this->_depth_v == 1 && !this->_has_base_stride_v)
            return this->_level_indexer<0>().step();
        if constexpr (this->_depth_v != 1 && !this->_has_base_stride_v)
            return 1;
    }

    _regular_elem_iter<_elem_t> element_begin() const
    {
        return {this->base_ptr_, stride()};
    }
    _regular_elem_iter<_elem_t> element_end() const
    {
        return {this->base_ptr_ + this->total_size() * stride(), stride()};
    }
    _regular_elem_const_iter<_elem_t> element_cbegin() const
    {
        return {this->base_ptr_, stride()};
    }
    _regular_elem_const_iter<_elem_t> element_cend() const
    {
        return {this->base_ptr_ + this->total_size() * stride(), stride()};
    }

    template<bool IsExplicitConst, size_t Level>
    auto _begin_impl() const
    {
        static_assert(0 < Level && Level <= _depth_v);
        if constexpr (Level == _depth_v)
        {
            return this->element_begin();
        }
        else
        {
            size_t ptr_stride = this->_total_base_size<
                _base_depth_v, this->_non_scalar_indexers_table[Level - 1] + 1>();
            auto   sub_view   = this->tuple_part_view(_repeat_tuple_t<Level, size_t>{});
            return _regular_view_iter<decltype(sub_view), IsExplicitConst>{std::move(sub_view), ptr_stride};
        }
    }
    template<bool IsExplicitConst, size_t Level>
    auto _end_impl() const
    {
        static_assert(0 < Level && Level <= _depth_v);
        if constexpr (Level == _depth_v)
        {
            return this->element_end();
        }
        else
        {
            auto iter = this->_begin_impl<IsExplicitConst, Level>();
            iter += this->total_size<Level>();
            return iter;
        }
    }

    template<size_t Level = 1>
    auto begin() const
    {
        return _begin_impl<false, Level>();
    }
    template<size_t Level = 1>
    auto end() const
    {
        return _end_impl<false, Level>();
    }
    template<size_t Level = 1>
    auto cbegin() const
    {
        return _begin_impl<true, Level>();
    }
    template<size_t Level = 1>
    auto cend() const
    {
        return _end_impl<true, Level>();
    }

    // for each element in the view, call fn(element) in order
    template<typename Function>
    void traverse(Function fn) const
    {
        const size_t    size   = this->total_size();
        const ptrdiff_t stride = this->stride();
        for (size_t i = 0; i < size; ++i)
            fn(*(this->base_ptr_ + i * stride));
    }

    // copy data to destination given size, assuming no aliasing
    template<typename Iter>
    void copy_to(Iter dst, size_t size) const
    {
        auto src = this->base_ptr_;
        const auto stride = this->stride();
        for (size_t i = 0; i < size; ++i)
        {
            *dst = *src;
            src += stride;
            ++dst;
        }
    }

    // copy data to destination, assuming no aliasing
    template<typename Iter>
    void copy_to(Iter dst) const
    {
        this->copy_to(dst, this->total_size());
    }

    // copy data from source given size, assuming no aliasing
    template<typename Iter>
    void copy_from(Iter src, size_t size) const
    {
        static_assert(!_is_const_v);
        auto dst = this->base_ptr_;
        const auto stride = this->stride();
        for (size_t i = 0; i < size; ++i)
        {
            *dst = *src;
            dst += stride;
            ++src;
        }
    }

    // copy data from source, assuming no aliasing
    template<typename Iter>
    void copy_from(Iter src) const
    {
        this->copy_from(src, this->total_size());
    }

};

template<typename T, typename IndexerTuple>
class irregular_view : public regular_view<T, IndexerTuple>
{
public:
    using _my_type         = irregular_view;
    using _my_base         = regular_view<T, IndexerTuple>;
    using _elem_t          = typename _my_base::_elem_t;
    using _no_const_elem_t = typename _my_base::_no_const_elem_t;
    using _indexers_t      = typename _my_base::_indexers_t;
    using _base_ptr_t      = typename _my_base::_base_ptr_t;
    using _base_dims_t     = typename _my_base::_base_dims_t;
    using _base_stride_t   = typename _my_base::_base_stride_t;
    using _my_const_t      = irregular_view<const _elem_t, _indexers_t>;
    static constexpr bool   _is_const_v   = _my_base::_is_const_v;
    static constexpr size_t _depth_v      = _my_base::_depth_v;
    static constexpr size_t _base_depth_v = _my_base::_base_depth_v;
    static constexpr _view_type _my_view_type_v = _view_type::irregular;

public:
    explicit irregular_view(_base_ptr_t base_ptr, _base_dims_t base_dims,
                            _indexers_t indexers, size_t base_stride) :
        _my_base{base_ptr, base_dims, indexers, base_stride} {}

    template<typename Other>
    _my_type& operator=(Other&& other)
    {
        data_copy(std::forward<Other>(other), *this);
        return *this;
    }

    ptrdiff_t stride() const noexcept
    {
        if constexpr (this->_has_base_stride_v)
            return this->base_stride_;
        else
            return 1;
    }

    _irregular_elem_iter<irregular_view> element_begin()
    {
        // set all indices to zero
        std::array<size_t, _depth_v> indices{};
        return {*this, indices};
    }
    _irregular_elem_iter<irregular_view> element_end()
    {
        // set the first index to dimension<0>(), set all other indices to zero
        std::array<size_t, _depth_v> indices{this->dimension<0>()};
        return {*this, indices};
    }
    _irregular_elem_const_iter<irregular_view> element_cbegin() const
    {
        // set all indices to zero
        std::array<size_t, _depth_v> indices{};
        return {*this, indices};
    }
    _irregular_elem_const_iter<irregular_view> element_cend() const
    {
        // set the first index to dimension<0>(), set all other indices to zero
        std::array<size_t, _depth_v> indices{this->dimension<0>()};
        return {*this, indices};
    }


    template<bool IsExplicitConst, size_t Level>
    auto _regular_begin_impl() const
    { // is used if the iterator on this level is regular
        size_t ptr_stride = this->_total_base_size<
            _base_depth_v, this->_non_scalar_indexers_table[Level - 1] + 1>();
        auto   sub_view   = this->tuple_part_view(_repeat_tuple_t<Level, size_t>{});
        return _regular_view_iter<decltype(sub_view), IsExplicitConst>{sub_view, ptr_stride};
    }
    template<size_t Level>
    auto _irregular_begin_impl() const
    { // is used if the iterator on this level is irregular
        size_t ptr_stride = this->_total_base_size<
            _base_depth_v, this->_non_scalar_indexers_table[Level - 1] + 1>();
        auto   sub_view   = this->tuple_part_view(_repeat_tuple_t<Level, size_t>{});
        using iter_type = _derive_view_iter_type_t<
            this->_non_scalar_indexers_table[Level], _indexers_t, irregular_view, decltype(sub_view), false>;
        return iter_type{*this, sub_view, ptr_stride};
    }

    template<bool IsExplicitConst, size_t Level>
    auto _begin_impl() const
    {
        static_assert(0 < Level && Level <= _depth_v);
        if constexpr (Level == _depth_v)
        {
            return this->element_begin();
        }
        else
        {
            constexpr _view_type iter_type_v =
                _identify_view_iter_type_v<this->_non_scalar_indexers_table[Level], _indexers_t>;
            static_assert(iter_type_v == _view_type::regular || iter_type_v == _view_type::irregular);
            if constexpr (iter_type_v == _view_type::regular)
                return this->_regular_begin_impl<IsExplicitConst, Level>();
            else
                return this->_irregular_begin_impl<IsExplicitConst, Level>();
        }
    }
    template<bool IsExplicitConst, size_t Level>
    auto _end_impl() const
    {
        static_assert(0 < Level && Level <= _depth_v);
        if constexpr (Level == _depth_v)
        {
            return this->element_end();
        }
        else
        {
            auto iter = this->_begin_impl<IsExplicitConst, Level>(); // get begin as the iterator
            constexpr _view_type iter_type_v =
                _identify_view_iter_type_v<this->_non_scalar_indexers_table[Level], _indexers_t>;
            if constexpr (iter_type_v == _view_type::regular)
                iter += this->total_size<Level>();                  // add total_size if regular
            else
                iter._get_indices_ref()[0] = this->dimension<0>();   // modify indices[0] if irregular
            return iter;                                             // return the iterator
        }
    }

    template<size_t Level = 1>
    auto begin() const
    {
        return _begin_impl<false, Level>();
    }
    template<size_t Level = 1>
    auto end() const
    {
        return _end_impl<false, Level>();
    }
    template<size_t Level = 1>
    auto cbegin() const
    {
        return _begin_impl<true, Level>();
    }
    template<size_t Level = 1>
    auto cend() const
    {
        return _end_impl<true, Level>();
    }

    // for each element in the view, call fn(element) in order
    template<typename Function>
    void traverse(Function fn) const
    {
        traverse_impl<0, 0, Function>(fn);
    }

    // copy data to destination, assuming no aliasing
    template<typename Iter>
    void copy_to(Iter dst) const
    {
        auto copy_to_fn = [dst = dst](auto src_val) mutable { *dst = src_val; ++dst; };
        //auto copy_to_fn = [](auto x) {}; 
        this->traverse<decltype((copy_to_fn))>(copy_to_fn); // pass by reference type
    }

    // copy data to destination with size ignored, assuming no aliasing
    template<typename Iter>
    void copy_to(Iter dst, size_t) const
    {
        this->copy_to(dst);
    }

    // copy data from source, assuming no aliasing
    template<typename Iter>
    void copy_from(Iter src) const
    {
        static_assert(!_is_const_v);
        auto copy_from_fn = [src = src](auto& dst_val) mutable { dst_val = *src; ++src; };
        //auto& copy_from_fn_ref = copy_from_fn;
        this->traverse<decltype((copy_from_fn))>(copy_from_fn); // pass by reference type
    }

    // copy data from source with size ignored, assuming no aliasing
    template<typename Iter>
    void copy_from(Iter src, size_t) const
    {
        this->copy_from(src);
    }




protected:
    template<size_t BC = 0, size_t LC = 0, typename Function>
    void traverse_impl(Function fn, size_t offset = 0) const
    {
        const size_t new_offset = offset * this->_base_dimension<BC>();
        if constexpr (this->_non_scalar_indexers_table[LC] > BC) // encounter _scalar_indexer
        {
            traverse_impl<BC + 1, LC, Function>(fn, new_offset);
        }
        else if constexpr (LC == _depth_v - 1) // the last Level
        {
            const size_t dim_i  = this->dimension<LC>();
            for (size_t i = 0; i < dim_i; ++i)
            {
                size_t final_offset = new_offset + (this->_base_indexer<BC>())[i];
                fn(this->_base_at(final_offset));
            }
        }
        else // before the last Level
        {
            const size_t dim_i  = this->dimension<LC>();
            for (size_t i = 0; i < dim_i; ++i)
                traverse_impl<BC + 1, LC + 1, Function>(fn, new_offset + (this->_base_indexer<BC>())[i]);
        }
    }



};

}
