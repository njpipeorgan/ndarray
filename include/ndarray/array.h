#pragma once

#include <array>
#include <numeric>
#include <vector>

#include "decls.h"
#include "traits.h"
#include "array_view.h"

namespace ndarray
{

template<typename T, size_t Depth>
class array
{
public:
    using _my_type    = array;
    using _elem_t     = T;
    static constexpr size_t _depth_v = Depth;
    using _data_t     = std::vector<T>;
    using _dims_t     = std::array<size_t, _depth_v>;
    using _indexers_t = _n_all_indexer_tuple_t<_depth_v>;
    static constexpr _view_type _my_view_type_v = _view_type::array;
    static_assert(_depth_v > 0);

public:
    _data_t data_{};
    _dims_t dims_{};

public:
    explicit array(_dims_t dims) : 
        dims_{dims}
    {
        resize();
    }

    template<typename T>
    explicit array(std::initializer_list<T> dims)
    {
        resize(dims);
    }

    template<typename View, _view_type = View::_my_view_type_v>
    explicit array(const View& other) : 
        dims_{other.dimensions()}
    {
        assert(this->_identifier_ptr() != other._identifier_ptr());
        resize();
        other.copy_to(this->data());
    }
    
    // copy data from another view, assuming identical dimensions
    template<typename View, _view_type = View::_my_view_type_v>
    _my_type& operator=(const View& other)
    {
        data_copy(other, *this);
        return *this;
    }

    // resize by existing dimensions
    void resize()
    {
        data_.resize(_total_size_from_dims());
    }

    // resize by a container as new dimensions
    template<typename Dims>
    void resize(const _dims_t& dims)
    {
        dims_ = dims;
        resize();
    }

    // resize by an initializer list as new dimensions
    template<typename T>
    void resize(std::initializer_list<T> dims)
    {
        NDARRAY_ASSERT(dims.size() == _depth_v);
        NDARRAY_ASSERT(std::all_of(dims.begin(), dims.end(), [](auto d) { return d >= 0; }));
        std::copy_n(dims.begin(), _depth_v, dims_.begin());
        resize();
    }

    // total size of the array
    size_t total_size() const
    {
        return data_.size();
    }

    // dimension of the array on the i-th level
    template<size_t I>
    size_t dimension() const
    {
        static_assert(I < _depth_v);
        return dims_[I];
    }

    const size_t* _identifier_ptr() const
    {
        return dims_.data();
    }

    template<typename... Ints, typename = std::enable_if_t<sizeof...(Ints) == _depth_v && _is_all_ints_v<Ints...>>>
    _elem_t& operator()(Ints&&... ints)
    {
        return this->at(std::forward<decltype(ints)>(ints)...);
    }

    template<typename... Ints, typename = std::enable_if_t<sizeof...(Ints) == _depth_v && _is_all_ints_v<Ints...>>>
    _elem_t& operator()(Ints&&... ints) const
    {
        return this->at(std::forward<decltype(ints)>(ints)...);
    }

    template<typename... Spans, typename = std::enable_if_t<sizeof...(Spans) != _depth_v || !_is_all_ints_v<Spans...>>>
    _derive_view_type_t<_elem_t, _indexers_t, std::tuple<Spans...>>
        operator()(Spans&&... spans)
    {
        return this->part_view(std::forward<decltype(spans)>(spans)...);
    }

    template<typename... Spans, typename = std::enable_if_t<sizeof...(Spans) != _depth_v || !_is_all_ints_v<Spans...>>>
    _derive_view_type_t<_elem_t, _indexers_t, std::tuple<Spans...>>
        operator()(Spans&&... spans) const
    {
        return this->part_view(std::forward<decltype(spans)>(spans)...);
    }

    // indexing with a tuple/array of integers
    template<typename Tuple>
    _elem_t& tuple_at(const Tuple& indices)
    {
        static_assert(std::tuple_size_v<Tuple> == _depth_v, "incorrect number of indices");
        return _linear_at(_get_position(indices));
    }

    // indexing with a tuple/array of integers
    template<typename Tuple>
    const _elem_t& tuple_at(const Tuple& indices) const
    {
        static_assert(std::tuple_size_v<Tuple> == _depth_v, "incorrect number of indices");
        return _linear_at(_get_position(indices));
    }
   
    // indexing with multiple integers
    template<typename... Ints>
    _elem_t& at(Ints... ints)
    {
        return tuple_at(std::make_tuple(ints...));
    }

    // indexing with multiple integers
    template<typename... Ints>
    const _elem_t& at(Ints... ints) const
    {
        return tuple_at(std::make_tuple(ints...));
    }

    _elem_t* data()
    {
        return data_.data();
    }
    
    const _elem_t* data() const
    {
        return data_.data();
    }

    template<typename... Spans>
    _derive_view_type_t<_elem_t, _indexers_t, std::tuple<Spans...>> 
        part_view(Spans&&... spans)
    {
        return get_collapsed_view(
            data(), dims_.data(), _indexers_t{}, std::make_tuple(std::forward<decltype(spans)>(spans)...));
    }
    
    template<typename... Spans>
    _derive_view_type_t<const _elem_t, _indexers_t, std::tuple<Spans...>>
        part_view(Spans&&... spans) const
    {
        return get_collapsed_view(
            data(), dims_.data(), _indexers_t{}, std::make_tuple(std::forward<decltype(spans)>(spans)...));
    }

    template<typename Function>
    void traverse(Function fn)
    {
        std::for_each_n(data(), total_size(), fn);
    }

    template<typename Function>
    void traverse(Function fn) const
    {
        std::for_each_n(data(), total_size(), fn);
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

    // copy data to destination given size, assuming no aliasing
    template<typename Iter>
    void copy_to(Iter dst, size_t size) const
    {
        auto src = this->data();
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
    void copy_from(Iter src, size_t size)
    {
        auto dst = this->data();
        for (size_t i = 0; i < size; ++i)
        {
            *dst = *src;
            ++dst;
            ++src;
        }
    }

    // copy data from source, assuming no aliasing
    template<typename Iter>
    void copy_from(Iter src)
    {
        this->copy_from(src, this->total_size());
    }

public:
    size_t _total_size_from_dims() const
    {
        return std::accumulate(dims_.cbegin(), dims_.cend(), size_t(1), std::multiplies<>{});
    }

    _elem_t& _linear_at(size_t pos)
    {
        NDARRAY_ASSERT(pos < total_size());
        return data_[pos];
    }
    
    const _elem_t& _linear_at(size_t pos) const
    {
        NDARRAY_ASSERT(pos < total_size());
        return data_[pos];
    }

    template<size_t I = _depth_v - size_t(1), typename Tuple>
    size_t _get_position(const Tuple& tuple) const
    {
        size_t dim_i = dimension<I>();
        size_t pos_i = _add_if_negative<size_t>(std::get<I>(tuple), dim_i);
        NDARRAY_ASSERT(pos_i < dim_i);
        if constexpr (I == 0)
            return pos_i;
        else
            return pos_i + dim_i * _get_position<I - 1>(tuple);
    }

};


template<typename View>
auto make_array(View&& view)
{
    using view_t = remove_cvref_t<View>;
    return array<typename view_t::_no_const_elem_t, view_t::_depth_v>(std::forward<View>(view));
}

}
