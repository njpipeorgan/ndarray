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
// View type are derived by deduce_view_type_t based on indexer types.
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
    static constexpr size_t _depth_v           = indexer_tuple_depth_v<_indexers_t>;
    static constexpr size_t _base_depth_v      = std::tuple_size_v<_indexers_t>;
    static constexpr std::array<size_t, _depth_v>
        _non_scalar_indexers_table = make_non_scalar_indexer_table<_indexers_t>();
    static constexpr size_t _stride_depth_v    = _non_scalar_indexers_table[_depth_v - 1] + 1;
    static constexpr bool   _has_base_stride_v = (_stride_depth_v != _base_depth_v);
    using _base_stride_t   = std::conditional_t<_has_base_stride_v, size_t, empty_struct>;
    static_assert(_depth_v > 0);

public:
    _base_ptr_t          base_ptr_;       // the base pointer for element accessing
    const _base_dims_t   base_dims_;      // dimensions of the base array, also used to identify the base array
    const _indexers_t    indexers_;       // stores all indexers
    const _base_stride_t base_stride_;    // base stride between elements

public:
    array_view_base(_base_ptr_t base_ptr, _base_dims_t base_dims,
                    _indexers_t indexers, size_t base_stride ={}) :
        base_ptr_{base_ptr}, base_dims_{base_dims},
        indexers_{indexers}, base_stride_{base_stride} {}

    _elem_t* base_ptr() const
    {
        return base_ptr_;
    }
    _base_ptr_t& _base_ptr_ref()
    {
        return base_ptr_;
    }
    const _base_ptr_t& _base_ptr_ref() const
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
    template<size_t BI>
    size_t _base_dimension() const
    {
        static_assert(BI < _base_depth_v);
        return base_dims_[BI];
    }

    // dimension of the view on the i-th level
    template<size_t I>
    size_t dimension() const
    {
        size_t bdim_i = _base_dimension<_non_scalar_indexers_table[I]>();
        return _level_indexer<I>().size(bdim_i);
    }

    // stride of the view on the i-th level
    // equivalent to the product of dimensions from base level 
    // (_non_scalar_indexers_table[I]) to (_non_scalar_indexers_table[I - 1] + 1)
    template<size_t I, size_t BC = _non_scalar_indexers_table[I]>
    size_t _level_stride() const
    {
        static_assert(0 < I && I < _depth_v);
        size_t bdim_i = _base_dimension<BC>();
        if constexpr (BC == _non_scalar_indexers_table[I - 1] + 1)
            return bdim_i;
        else
            return _level_stride<I, BC - 1>() * bdim_i;
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
    size_t size() const
    {
        static_assert(FirstLevel <= LastLevel && LastLevel <= _depth_v);
        if constexpr (FirstLevel == LastLevel)
            return size_t(1);
        else
            return dimension<LastLevel - 1>() * size<LastLevel - 1, FirstLevel>();
    }

    // automatically calls at() or vpart(), depending on its arguments
    template<typename... Anys>
    deduce_view_or_elem_type_t<_elem_t&, _elem_t, _depth_v, _indexers_t, std::tuple<Anys...>>
        operator()(Anys&&... anys) const
    {
        constexpr bool is_complete_index = sizeof...(Anys) == _depth_v && is_all_ints_v<Anys...>;
        if constexpr (is_complete_index)
            return this->at(std::forward<decltype(anys)>(anys)...);
        else
            return this->vpart(std::forward<decltype(anys)>(anys)...);
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
    deduce_view_type_t<_elem_t, _indexers_t, SpanTuple>
        tuple_vpart(SpanTuple&& spans) const
    {
        return get_collapsed_view(
            base_ptr_, base_dims_, indexers_, std::forward<decltype(spans)>(spans));
    }

    template<typename... Spans>
    deduce_view_type_t<_elem_t, _indexers_t, std::tuple<Spans...>>
        vpart(Spans&&... spans) const
    {
        return tuple_vpart(std::forward_as_tuple(spans...));
    }

    // check whether having same dimensions with another array, starting at specific levels
    template<size_t MyStartLevel = 0, size_t OtherStartLevel = 0, typename OtherArray>
    bool check_size_with(const OtherArray& other) const
    {
        if constexpr (MyStartLevel == _depth_v || OtherStartLevel == OtherArray::_depth_v)
            return false;
        else if constexpr (MyStartLevel == _depth_v - 1 && OtherStartLevel == OtherArray::_depth_v - 1)
            return this->dimension<MyStartLevel>() == other.dimension<OtherStartLevel>();
        else
            return this->dimension<MyStartLevel>() == other.dimension<OtherStartLevel>() &&
            check_size_with<MyStartLevel + 1, OtherStartLevel + 1>(other);
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

    template<size_t LC, bool DoCheckDim = true, typename Int>
    size_t _get_level_base_position(Int pos) const
    {
        size_t dim_i  = dimension<LC>();
        size_t pos_i  = _add_if_negative<size_t>(pos, dim_i);
        if constexpr (DoCheckDim)
            NDARRAY_ASSERT(pos_i < dim_i);
        return _level_indexer<LC>()[pos_i];
    }

    template<typename Tuple, size_t LC = std::tuple_size_v<Tuple> -1, size_t BC = _non_scalar_indexers_table[LC]>
    size_t _get_position(const Tuple& tuple) const
    {
        if constexpr (_non_scalar_indexers_table[LC] == BC)
        {
            size_t bdim_i = _base_dimension<BC>();
            size_t bpos_i = _get_level_base_position<LC>(std::get<LC>(tuple));
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

public:
    simple_view(_base_ptr_t base_ptr, _base_dims_t base_dims,
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

    simple_elem_iter<_elem_t> element_begin() const
    {
        return {this->base_ptr_};
    }
    simple_elem_iter<_elem_t> element_end() const
    {
        return {this->base_ptr_ + this->size()};
    }
    simple_elem_const_iter<_elem_t> element_cbegin() const
    {
        return {this->base_ptr_};
    }
    simple_elem_const_iter<_elem_t> element_cend() const
    {
        return {this->base_ptr_ + this->size()};
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
            auto   sub_view   = this->tuple_vpart(repeat_tuple_t<Level, size_t>{});
            return regular_view_iter<decltype(sub_view), IsExplicitConst>{std::move(sub_view), ptr_stride};
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
            iter += this->size<Level>();
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

    //// for each element in the view, call fn(element) in order
    //template<typename Function>
    //void traverse(Function fn) const
    //{
    //    const size_t size = this->size();
    //    for (size_t i = 0; i < size; ++i)
    //        fn(*(this->base_ptr_ + i));
    //}

    //// return a forward-iterator-like function object
    //auto traverse_iterator() const
    //{
    //    auto iter =
    //        [base_ptr=this->base_ptr_]() mutable
    //    {
    //        _elem_t& val = *base_ptr;
    //        ++base_ptr;
    //        return val;
    //    };
    //    return iter;
    //}

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
        this->copy_to(dst, this->size());
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
        this->copy_from(src, this->size());
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

public:
    regular_view(_base_ptr_t base_ptr, _base_dims_t base_dims,
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

    regular_elem_iter<_elem_t> element_begin() const
    {
        return {this->base_ptr_, stride()};
    }
    regular_elem_iter<_elem_t> element_end() const
    {
        return {this->base_ptr_ + this->size() * stride(), stride()};
    }
    regular_elem_const_iter<_elem_t> element_cbegin() const
    {
        return {this->base_ptr_, stride()};
    }
    regular_elem_const_iter<_elem_t> element_cend() const
    {
        return {this->base_ptr_ + this->size() * stride(), stride()};
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
            auto   sub_view   = this->tuple_vpart(repeat_tuple_t<Level, size_t>{});
            return regular_view_iter<decltype(sub_view), IsExplicitConst>{std::move(sub_view), ptr_stride};
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
            iter += this->size<Level>();
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

    //// for each element in the view, call fn(element) in order
    //template<typename Function>
    //void traverse(Function fn) const
    //{
    //    const size_t    size   = this->size();
    //    const ptrdiff_t stride = this->stride();
    //    for (size_t i = 0; i < size; ++i)
    //        fn(*(this->base_ptr_ + i * stride));
    //}

    //// return a forward-iterator-like function object
    //auto traverse_iterator() const
    //{
    //    auto iter =
    //        [base_ptr=this->base_ptr_, stride=stride()]() mutable
    //    {
    //        _elem_t& val = *base_ptr;
    //        base_ptr += stride;
    //        return val;
    //    };
    //    return iter;
    //}

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
        this->copy_to(dst, this->size());
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
        this->copy_from(src, this->size());
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

public:
    irregular_view(_base_ptr_t base_ptr, _base_dims_t base_dims,
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

    irregular_elem_iter<irregular_view> element_begin()
    {
        // set all indices to zero
        std::array<size_t, _depth_v> indices{};
        return {*this, indices};
    }
    irregular_elem_iter<irregular_view> element_end()
    {
        // set the first index to dimension<0>(), set all other indices to zero
        std::array<size_t, _depth_v> indices{this->dimension<0>()};
        return {*this, indices};
    }
    irregular_elem_const_iter<irregular_view> element_cbegin() const
    {
        // set all indices to zero
        std::array<size_t, _depth_v> indices{};
        return {*this, indices};
    }
    irregular_elem_const_iter<irregular_view> element_cend() const
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
        auto   sub_view   = this->tuple_vpart(repeat_tuple_t<Level, size_t>{});
        return regular_view_iter<decltype(sub_view), IsExplicitConst>{sub_view, ptr_stride};
    }
    template<bool IsExplicitConst, size_t Level>
    auto _irregular_begin_impl() const
    { // is used if the iterator on this level is irregular
        size_t ptr_stride = this->_total_base_size<
            _base_depth_v, this->_non_scalar_indexers_table[Level - 1] + 1>();
        auto   sub_view   = this->tuple_vpart(repeat_tuple_t<Level, size_t>{});
        return irregular_view_iter<decltype(sub_view), irregular_view, IsExplicitConst>{*this, sub_view, ptr_stride};
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
            constexpr array_obj_type iter_type_v =
                identify_view_iter_type_v<this->_non_scalar_indexers_table[Level], _indexers_t>;
            static_assert(iter_type_v == array_obj_type::regular || iter_type_v == array_obj_type::irregular);
            if constexpr (iter_type_v == array_obj_type::regular)
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
            constexpr array_obj_type iter_type_v =
                identify_view_iter_type_v<this->_non_scalar_indexers_table[Level], _indexers_t>;
            if constexpr (iter_type_v == array_obj_type::regular)
                iter += this->size<Level>();                  // add size if regular
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
        if constexpr (this->_non_scalar_indexers_table[LC] > BC) // encounter scalar_indexer
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


//
// regular_view_iter and irregular_view_iter is the generalized version
// of iterators for multi-dimensional arrays. They allow iterations over 
// various levels. 
//
// View iterators stores an sub view of the base view that can be copied, 
// or returned as const reference. Regular iterators only describe iterators 
// with fixed stride, and use the base pointer in the sub view as the 
// indicator of position. Irregular iterators stores extra indices as the 
// position, and update the base pointer when necessary.
//
// Arithmetic operations on irregular_elem_iter have the time complexity 
// O(_iter_depth_v) at most. 
//

template<typename SubView, bool IsExplicitConst>
class regular_view_iter
{
public:
    using _my_type      = regular_view_iter;
    using _sub_view_t   = SubView;
    using _elem_t       = typename _sub_view_t::_elem_t;
    static constexpr bool _is_const_v = IsExplicitConst || std::is_const_v<_elem_t>;
    using _ret_view_t   = std::conditional_t<_is_const_v, typename _sub_view_t::_my_const_t, _sub_view_t>;
    using _base_ptr_t   = typename _ret_view_t::_base_ptr_t;
    using _ptr_stride_t = size_t;

public:
    _ret_view_t   ret_view_;
    _ptr_stride_t ptr_stride_;  // always positive

public:
    regular_view_iter(_sub_view_t sub_view, _ptr_stride_t ptr_stride) :
        ret_view_{sub_view}, ptr_stride_{ptr_stride} {}

    _base_ptr_t& my_base_ptr_ref()
    {
        return ret_view_._base_ptr_ref();
    }
    const _base_ptr_t& my_base_ptr_ref() const
    {
        return ret_view_._base_ptr_ref();
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
class irregular_view_iter
{
public:
    using _my_type           = irregular_view_iter;
    using _sub_view_t        = SubView;
    using _base_view_t       = BaseView;
    using _base_view_cref_t  = const _base_view_t&;
    using _elem_t            = typename _sub_view_t::_elem_t;
    static constexpr bool _is_const_v = IsExplicitConst || std::is_const_v<_elem_t>;
    using _ret_view_t        = std::conditional_t<_is_const_v, typename _sub_view_t::_my_const_t, _sub_view_t>;
    using _base_ptr_t        = typename _ret_view_t::_base_ptr_t;
    using _ptr_stride_t      = size_t;
    static constexpr size_t _iter_depth_v = _base_view_t::_depth_v - _sub_view_t::_depth_v;
    using _indices_t         = std::array<size_t, _iter_depth_v>;

public:
    _indices_t          indices_{};    // zeros by default internally
    _base_view_cref_t   base_view_cref_;
    mutable _ret_view_t ret_view_;
    _ptr_stride_t       ptr_stride_;   // always positive

public:
    irregular_view_iter(_base_view_cref_t base_view_cref, _sub_view_t sub_view, _ptr_stride_t ptr_stride) :
        base_view_cref_{base_view_cref}, ret_view_{sub_view}, ptr_stride_{ptr_stride} {}

    // get reference to the indices
    _indices_t& _get_indices_ref()
    {
        return indices_;
    }

    //template<size_t IterLevel>
    //size_t _index_dimension() const
    //{
    //    static_assert(IterLevel < _iter_depth_v);
    //    return base_view_cref_.dimension<IterLevel>();
    //}

    void _update_sub_view_base_ptr() const
    {
        size_t pos = base_view_cref_._get_position(indices_);
        ret_view_._base_ptr_ref() = base_view_cref_.base_ptr() + pos * ptr_stride_;
    }

    template<typename Diff>
    _my_type& operator+=(Diff diff)
    {
        this->inc(diff);
        return *this;
    }
    template<typename Diff>
    _my_type& operator-=(Diff diff)
    {
        this->dec(diff);
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
        this->explicit_inc();
        return *this;
    }
    _my_type& operator--()
    {
        this->explicit_dec();
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
        this->_update_sub_view_base_ptr();
        return ret_view_;
    }
    _ret_view_t operator[](ptrdiff_t diff) const
    {
        return *((*this) + diff);
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return this->difference(other);
    }

    bool operator==(const _my_type& other) const
    {
        return this->cmp_equal(other);
    }
    bool operator<(const _my_type& other) const
    {
        return this->cmp_less(other);
    }
    bool operator>(const _my_type& other) const
    {
        return this->cmp_greater(other);
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
    template<typename Diff>
    void inc(Diff diff)
    {
        if constexpr (std::is_unsigned_v<Diff>)
            explicit_inc(diff);
        else if (diff >= 0)
            explicit_inc(diff);
        else
            explicit_dec(-diff);
    }

    template<typename Diff>
    void dec(Diff diff)
    {
        if constexpr (std::is_unsigned_v<Diff>)
            explicit_dec(diff);
        else if (diff >= 0)
            explicit_dec(diff);
        else
            explicit_inc(-diff);
    }

    // increment by 1 on Level
    template<size_t Level = _iter_depth_v - 1>
    void explicit_inc()
    {
        const size_t dim = base_view_cref_.dimension<Level>();
        indices_[Level]++;
        if constexpr (Level > 0)
        {
            if (indices_[Level] < dim) // if did not overflow
            { // do nothing
            }
            else // if did overflow
            {
                indices_[Level] = 0;
                explicit_inc<Level - 1>(); // carry
            }
        }
    }

    // increment by diff (diff >= 0) on Level
    template<size_t Level = _iter_depth_v - 1>
    void explicit_inc(size_t diff)
    {
        const size_t dim = base_view_cref_.dimension<Level>();
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
    template<size_t Level = _iter_depth_v - 1>
    void explicit_dec()
    {
        const size_t dim = base_view_cref_.dimension<Level>();
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
    template<size_t Level = _iter_depth_v - 1>
    void explicit_dec(size_t diff)
    {
        const size_t dim = base_view_cref_.dimension<Level>();
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
    template<size_t Level = _iter_depth_v - 1>
    ptrdiff_t difference(const _my_type& other) const
    {
        ptrdiff_t    diff = this->indices_[Level] - other.indices_[Level];
        const size_t dim  = base_view_cref_.dimension<Level>();
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
// Array view element iterator, iterates all elements in accessing order.
//
// The constness of the elements accessed through element iterators are 
// controlled by both their source array views and IsExplicitConst argument.
// _is_const_v indicates this constness. 
//
// simple_elem_iter and regular_elem_iter store the pointer to the 
// elements and access them by patterns. 
//
// Arithmetic operations on irregular_elem_iter have time complexity O(1) 
// on average, O(_depth_v) at maximum.
//

template<typename T, bool IsExplicitConst>
class simple_elem_iter
{
public:
    using _my_type    = simple_elem_iter;
    static constexpr bool _is_const_v = IsExplicitConst || std::is_const_v<T>;
    using _elem_t     = std::conditional_t<_is_const_v, const T, T>;
    using _elem_ptr_t = _elem_t*;

protected:
    _elem_ptr_t ptr_;

public:
    simple_elem_iter(_elem_ptr_t ptr) :
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
        ++ptr_;
        return *this;
    }
    _my_type& operator--()
    {
        --ptr_;
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

    _elem_t& operator*() const
    {
        return *ptr_;
    }
    _elem_t& operator[](ptrdiff_t diff) const
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
class regular_elem_iter
{
public:
    using _my_type    = regular_elem_iter;
    using _stride_t   = ptrdiff_t;
    static constexpr bool _is_const_v = IsExplicitConst || std::is_const_v<T>;
    using _elem_t     = std::conditional_t<_is_const_v, const T, T>;
    using _elem_ptr_t = _elem_t*;

protected:
    _elem_ptr_t ptr_;
    _stride_t   stride_;

public:
    regular_elem_iter(_elem_ptr_t ptr, _stride_t stride) :
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
        ptr_ += stride_;
        return *this;
    }
    _my_type& operator--()
    {
        ptr_ -= stride_;
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

    _elem_t& operator*() const
    {
        return *ptr_;
    }
    _elem_t& operator[](ptrdiff_t diff) const
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
class irregular_elem_iter
{
public:
    using _my_type     = irregular_elem_iter;
    using _view_t      = View;
    using _view_cref_t = const _view_t&;
    using _view_elem_t = typename _view_t::_elem_t;
    static constexpr size_t _depth_v    = _view_t::_depth_v;
    static constexpr bool   _is_const_v = IsExplicitConst || std::is_const_v<_view_elem_t>;
    using _elem_t      = std::conditional_t<_is_const_v, const _view_elem_t, _view_elem_t>;
    using _indices_t   = std::array<size_t, _depth_v>;

protected:
    _view_cref_t view_cref_;
    _indices_t   indices_;

public:
    irregular_elem_iter(_view_cref_t view_cref, _indices_t indices) :
        view_cref_{view_cref}, indices_{indices} {}

    _my_type& operator+=(ptrdiff_t diff)
    {
        this->inc(diff);
        return *this;
    }
    _my_type& operator-=(ptrdiff_t diff)
    {
        this->dec(diff);
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
        this->explicit_inc();
        return *this;
    }
    _my_type& operator--()
    {
        this->explicit_dec();
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

    _elem_t& operator*() const
    {
        return view_cref_.tuple_at(indices_);
    }
    _elem_t& operator[](ptrdiff_t diff) const
    {
        return *((*this) + diff);
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return this->difference(other);
    }

    bool operator==(const _my_type& other) const
    {
        return this->cmp_equal(other);
    }
    bool operator<(const _my_type& other) const
    {
        return this->cmp_less(other);
    }
    bool operator>(const _my_type& other) const
    {
        return this->cmp_greater(other);
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
    template<typename Diff>
    void inc(Diff diff)
    {
        if constexpr (std::is_unsigned_v<Diff>)
            explicit_inc(diff);
        else if (diff >= 0)
            explicit_inc(diff);
        else
            explicit_dec(-diff);
    }

    template<typename Diff>
    void dec(Diff diff)
    {
        if constexpr (std::is_unsigned_v<Diff>)
            explicit_dec(diff);
        else if (diff >= 0)
            explicit_dec(diff);
        else
            explicit_inc(-diff);
    }

    // increment by 1 on Level
    template<size_t Level = _depth_v - 1>
    void explicit_inc()
    {
        const size_t dim = view_cref_.dimension<Level>();
        indices_[Level]++;
        if constexpr (Level > 0)
        {
            if (indices_[Level] < dim) // if did not overflow
            {
            }
            else // if did overflow
            {
                indices_[Level] = 0;
                explicit_inc<Level - 1>(); // carry
            }
        }
    }

    // increment by diff (diff >= 0) on Level
    template<size_t Level = _depth_v - 1>
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
    template<size_t Level = _depth_v - 1>
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
    template<size_t Level = _depth_v - 1>
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
    template<size_t Level = _depth_v - 1>
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


// create array from array view
template<typename T, typename IndexerTuple>
auto make_array(const simple_view<T, IndexerTuple>& view)
{
    using view_t = simple_view<T, IndexerTuple>;
    return array<typename view_t::_no_const_elem_t, view_t::_depth_v>(view);
}

// create array from array view
template<typename T, typename IndexerTuple>
auto make_array(const regular_view<T, IndexerTuple>& view)
{
    using view_t = regular_view<T, IndexerTuple>;
    return array<typename view_t::_no_const_elem_t, view_t::_depth_v>(view);
}

// create array from array view
template<typename T, typename IndexerTuple>
auto make_array(const irregular_view<T, IndexerTuple>& view)
{
    using view_t = irregular_view<T, IndexerTuple>;
    return array<typename view_t::_no_const_elem_t, view_t::_depth_v>(view);
}

}
