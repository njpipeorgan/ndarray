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
    using _elem_t     = T;
    static constexpr size_t _depth_v = Depth;
    using _data_t     = std::vector<T>;
    using _dims_t     = std::array<size_t, _depth_v>;
    using _indexers_t = _n_all_indexer_tuple_t<_depth_v>;
    static_assert(_depth_v > 0);

public:
    _data_t data_{};
    _dims_t dims_{};

public:
    template<typename T>
    explicit array(std::initializer_list<T> dims)
    {
        resize(dims);
    }
    
    template<typename Dims>
    explicit array(const Dims& dims)
    {
        resize(dims);
    }

    // resize by existing dimensions
    void resize()
    {
        data_.resize(_total_size_from_dims());
    }

    // resize by a container as new dimensions
    template<typename Dims>
    void resize(const Dims& dims)
    {
        NDARRAY_ASSERT(dims.size() == _depth_v);
        NDARRAY_ASSERT(std::all_of(dims.begin(), dims.end(), [](auto d) { return d >= 0; }));
        std::copy_n(dims.begin(), _depth_v, dims_.begin());
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

}
