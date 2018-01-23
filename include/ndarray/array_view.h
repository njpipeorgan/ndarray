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
    array_view_base(_base_ptr_t base_ptr, _base_dims_t base_dims, _indexers_t indexers, size_t base_stride ={}) :
        base_ptr_{base_ptr}, base_dims_{base_dims}, indexers_{std::move(indexers)}, base_stride_{base_stride} {}

    _base_ptr_t& _get_base_ptr_ref()
    {
        return base_ptr_;
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
    auto tuple_part_view(SpanTuple&& spans) const
    {
        return get_collapsed_view(
            base_ptr_, base_dims_, indexers_, std::forward<decltype(spans)>(spans));
    }
    template<typename... Spans>
    auto part_view(Spans&&... spans) const
    {
        return tuple_part_view(std::make_tuple(std::forward<decltype(spans)>(spans)...));
    }

public:
    template<size_t LastLevel = _depth_v, size_t FirstLevel = 0>
    size_t _total_size() const
    {
        static_assert(FirstLevel <= LastLevel && LastLevel <= _depth_v);
        if constexpr (FirstLevel == LastLevel)
            return size_t(1);
        else
            return dimension<LastLevel - 1>() * _total_size<LastLevel - 1, FirstLevel>();
    }

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

};

template<typename T, typename IndexerTuple>
class simple_view : public array_view_base<T, IndexerTuple>
{
public:
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
    simple_view(_base_ptr_t base_ptr, _base_dims_t base_dims, _indexers_t indexers, size_t) :
        _my_base{base_ptr, base_dims, std::move(indexers)} {}

    // convert from lvalue non-const view to this type
    simple_view(const simple_view<_no_const_elem_t, _indexers_t>& other) :
        _my_base{other.base_ptr_, other.base_dims_, other.indexers_} {}

    // convert from rvalue non-const view to this type
    simple_view(simple_view<_no_const_elem_t, _indexers_t>&& other) :
        _my_base{other.base_ptr_, other.base_dims_, std::move(other.indexers_)} {}

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
        return {this->base_ptr_ + this->_total_size()};
    }
    _simple_elem_const_iter<_elem_t> element_cbegin() const
    {
        return {this->base_ptr_};
    }
    _simple_elem_const_iter<_elem_t> element_cend() const
    {
        return {this->base_ptr_ + this->_total_size()};
    }

    template<size_t Level = 1>
    auto begin() const
    {
        static_assert(0 < Level && Level < _depth_v);
        size_t ptr_stride = this->_total_base_size<
            _base_depth_v, this->_non_scalar_indexers_table[Level - 1] + 1>();
        auto   sub_view   = this->tuple_part_view(_repeat_tuple_t<Level, size_t>{});
        using iter_type = _regular_view_iter<decltype(sub_view), false>;
        return iter_type{sub_view, ptr_stride};
    }
    template<size_t Level = 1>
    auto end() const
    {
        auto iter = this->begin<Level>();
        iter += this->_total_size<Level>();
        return iter;
    }
    template<size_t Level = 1>
    auto cbegin() const
    {
        using iter_type = typename decltype(this->begin<Level>())::_my_const_t;
        return iter_type{this->begin<Level>()};
    }
    template<size_t Level = 1>
    auto cend() const
    {
        using iter_type = typename decltype(this->end<Level>())::_my_const_t;
        return iter_type{this->end<Level>()};
    }

    // for each element in the view, call fn(element) in order
    template<typename Function>
    void traverse(Function fn) const
    {
        const size_t size = this->_total_size();
        for (size_t i = 0; i < size; ++i)
            fn(*(this->base_ptr_ + i));
    }

};

template<typename T, typename IndexerTuple>
class regular_view : public array_view_base<T, IndexerTuple>
{
public:
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
    regular_view(_base_ptr_t base_ptr, _base_dims_t base_dims, _indexers_t indexers, size_t base_stride) :
        _my_base{base_ptr, base_dims, std::move(indexers), base_stride} {}

    // convert from lvalue non-const view to this type
    regular_view(const regular_view<_no_const_elem_t, _indexers_t>& other) :
        _my_base{other.base_ptr_, other.base_dims_, other.indexers_, other.base_stride_} {}

    // convert from rvalue non-const view to this type
    regular_view(regular_view<_no_const_elem_t, _indexers_t>&& other) :
        _my_base{other.base_ptr_, other.base_dims_, std::move(other.indexers_), other.base_stride_} {}

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
        return {this->base_ptr_ + this->_total_size() * stride(), stride()};
    }
    _regular_elem_const_iter<_elem_t> element_cbegin() const
    {
        return {this->base_ptr_, stride()};
    }
    _regular_elem_const_iter<_elem_t> element_cend() const
    {
        return {this->base_ptr_ + this->_total_size() * stride(), stride()};
    }

    template<size_t Level = 1>
    auto begin() const
    {
        static_assert(0 < Level && Level < _depth_v);
        size_t ptr_stride = this->_total_base_size<
            _base_depth_v, this->_non_scalar_indexers_table[Level - 1] + 1>();
        auto   sub_view   = this->tuple_part_view(_repeat_tuple_t<Level, size_t>{});
        using iter_type = _regular_view_iter<decltype(sub_view), false>;
        return iter_type{sub_view, ptr_stride};
    }
    template<size_t Level = 1>
    auto end() const
    {
        auto iter = this->begin<Level>();
        iter += this->_total_size<Level>();
        return iter;
    }
    template<size_t Level = 1>
    auto cbegin() const
    {
        using iter_type = typename decltype(this->begin<Level>())::_my_const_t;
        return iter_type{this->begin<Level>()};
    }
    template<size_t Level = 1>
    auto cend() const
    {
        using iter_type = typename decltype(this->end<Level>())::_my_const_t;
        return iter_type{this->end<Level>()};
    }

    // for each element in the view, call fn(element) in order
    template<typename Function>
    void traverse(Function fn) const
    {
        const size_t    size   = this->_total_size();
        const ptrdiff_t stride = this->stride();
        for (size_t i = 0; i < size; ++i)
            fn(*(this->base_ptr_ + i * stride));
    }

};

template<typename T, typename IndexerTuple>
class irregular_view : public regular_view<T, IndexerTuple>
{
public:
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
    irregular_view(_base_ptr_t base_ptr, _base_dims_t base_dims, _indexers_t indexers, size_t base_stride) :
        _my_base{base_ptr, base_dims, std::move(indexers), base_stride} {}

    // convert from lvalue non-const view to this type
    irregular_view(const irregular_view<_no_const_elem_t, _indexers_t>& other) :
        _my_base{other.base_ptr_, other.base_dims_, other.indexers_, other.base_stride_} {}

    // convert from rvalue non-const view to this type
    irregular_view(irregular_view<_no_const_elem_t, _indexers_t>&& other) :
        _my_base{other.base_ptr_, other.base_dims_, std::move(other.indexers_), other.base_stride_} {}

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

    template<size_t Level>
    auto _regular_iter_begin() const
    { // is used if the iterator on this level is regular
        size_t ptr_stride = this->_total_base_size<
            _base_depth_v, this->_non_scalar_indexers_table[Level - 1] + 1>();
        auto   sub_view   = this->tuple_part_view(_repeat_tuple_t<Level, size_t>{});
        using iter_type = _regular_view_iter<decltype(sub_view), false>;
        return iter_type{sub_view, ptr_stride};
    }
    template<size_t Level>
    auto _irregular_iter_begin() const
    { // is used if the iterator on this level is irregular
        auto sub_view = this->tuple_part_view(_repeat_tuple_t<Level, size_t>{});
        size_t ptr_stride = this->_total_base_size<
            _base_depth_v, this->_non_scalar_indexers_table[Level - 1] + 1>();
        using iter_type = _derive_view_iter_type_t<
            this->_non_scalar_indexers_table[Level], _indexers_t, irregular_view, decltype(sub_view), false>;
        //using _ = typename iter_type::_;
        return iter_type{*this, sub_view, ptr_stride};
    }

    template<size_t Level = 1>
    auto begin() const
    {
        static_assert(0 < Level && Level < _depth_v);
        constexpr _view_type iter_type_v =
            _identify_view_iter_type_v<this->_non_scalar_indexers_table[Level], _indexers_t>;
        static_assert(iter_type_v == _view_type::regular || iter_type_v == _view_type::irregular);
        if constexpr (iter_type_v == _view_type::regular)
            return this->_regular_iter_begin<Level>();
        else
            return this->_irregular_iter_begin<Level>();
    }
    template<size_t Level = 1>
    auto end() const
    {
        constexpr _view_type iter_type_v =
            _identify_view_iter_type_v<this->_non_scalar_indexers_table[Level], _indexers_t>;
        if constexpr (iter_type_v == _view_type::regular)
        {
            auto iter = this->begin<Level>();
            iter += this->_total_size<Level>();
            return iter;
        }
        else
        {
            auto iter = this->begin<Level>();
            iter._get_indices_ref()[0] = this->dimension<0>();
            return iter;
        }
    }
    template<size_t Level = 1>
    auto cbegin() const
    {
        using iter_type = typename decltype(this->begin<Level>())::_my_const_t;
        return iter_type{this->begin<Level>()};
    }
    template<size_t Level = 1>
    auto cend() const
    {
        using iter_type = typename decltype(this->end<Level>())::_my_const_t;
        return iter_type{this->end<Level>()};
    }

    // for each element in the view, call fn(element) in order
    template<typename Function>
    void traverse(Function fn) const
    {
        traverse_impl(fn);
    }

protected:
    template<size_t BC = 0, size_t LC = 0, typename Function>
    void traverse_impl(Function fn, size_t offset = 0) const
    {
        const size_t new_offset = offset * this->_base_dimension<BC>();
        if constexpr (this->_non_scalar_indexers_table[LC] > BC) // encounter _scalar_indexer
        {
            traverse_impl<BC + 1, LC>(fn, new_offset);
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
                traverse_impl<BC + 1, LC + 1>(fn, new_offset + (this->_base_indexer<BC>())[i]);
        }
    }

};

}
