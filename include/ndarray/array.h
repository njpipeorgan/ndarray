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
    using _indexers_t = n_all_indexer_tuple_t<_depth_v>;
    static_assert(_depth_v > 0);

public:
    _data_t data_{};
    _dims_t dims_{};

public:
    array(_dims_t dims) :
        dims_{std::move(dims)}
    {
        resize();
    }

    template<typename View>
    array(const View& other) :
        dims_{other.dimensions()}
    {
        assert(this->_identifier_ptr() != other._identifier_ptr());
        resize();
        other.copy_to(this->data());
    }

    array(const std::vector<_elem_t>& data, _dims_t dims) :
        data_{data}, dims_{dims}
    {
        // caller should check the dimensions
        NDARRAY_ASSERT(_check_size());
    }

    array(std::vector<_elem_t>&& data, _dims_t dims) :
        data_{std::move(data)}, dims_{dims}
    {
        // caller should check the dimensions
        NDARRAY_ASSERT(_check_size());
    }

    // copy data from another view, assuming identical dimensions
    template<typename View>
    _my_type& operator=(const View& other)
    {
        data_copy(other, *this);
        return *this;
    }

    // resize by existing dimensions
    void resize()
    {
        data_.resize(_total_size_impl());
    }

    // resize by a container as new dimensions
    template<typename Dims>
    void resize(_dims_t dims)
    {
        dims_ = std::move(dims);
        resize();
    }

    // total size of the array
    size_t size() const
    {
        return data_.size();
    }

    // check whether the size of data_ is compatible with dims_
    bool _check_size() const
    {
        bool is_compatible = (size() == _total_size_impl());
        assert(is_compatible);
        return is_compatible;
    }

    // dimension of the array on the i-th level
    template<size_t I>
    size_t dimension() const
    {
        static_assert(I < _depth_v);
        return dims_[I];
    }

    // array of dimensions
    _dims_t dimensions() const
    {
        return dims_;
    }

    const size_t* _dims_data() const
    {
        return dims_.data();
    }

    const size_t* _identifier_ptr() const
    {
        return _dims_data();
    }

    // automatically calls at() or vpart(), depending on its arguments
    template<typename... Anys>
    deduce_part_or_elem_type_t<_elem_t, _elem_t, _depth_v, _indexers_t, std::tuple<Anys...>>
        operator()(Anys&&... anys) &&
    {
        constexpr bool is_complete_index = sizeof...(Anys) == _depth_v && is_all_ints_v<Anys...>;
        if constexpr (is_complete_index)
            return this->at(std::forward<decltype(anys)>(anys)...);
        else
            return this->part(std::forward<decltype(anys)>(anys)...);
    }

    // automatically calls at() or vpart(), depending on its arguments
    template<typename... Anys>
    deduce_array_view_or_elem_type_t<_elem_t&, _elem_t, _depth_v, _indexers_t, std::tuple<Anys...>>
        operator()(Anys&&... anys) &
    {
        constexpr bool is_complete_index = sizeof...(Anys) == _depth_v && is_all_ints_v<Anys...>;
        if constexpr (is_complete_index)
            return this->at(std::forward<decltype(anys)>(anys)...);
        else
            return this->vpart(std::forward<decltype(anys)>(anys)...);
    }

    // automatically calls at() or vpart(), depending on its arguments
    template<typename... Anys>
    deduce_array_view_or_elem_type_t<const _elem_t&, const _elem_t, _depth_v, _indexers_t, std::tuple<Anys...>>
        operator()(Anys&&... anys) const &
    {
        constexpr bool is_complete_index = sizeof...(Anys) == _depth_v && is_all_ints_v<Anys...>;
        if constexpr (is_complete_index)
            return this->at(std::forward<decltype(anys)>(anys)...);
        else
            return this->vpart(std::forward<decltype(anys)>(anys)...);
    }


    // indexing with a tuple/array of integers
    template<typename Tuple>
    _elem_t tuple_at(const Tuple& indices) &&
    {
        static_assert(std::tuple_size_v<Tuple> == _depth_v, "incorrect number of indices");
        return _linear_at(_get_position(indices));
    }

    // indexing with a tuple/array of integers
    template<typename Tuple>
    _elem_t& tuple_at(const Tuple& indices) &
    {
        static_assert(std::tuple_size_v<Tuple> == _depth_v, "incorrect number of indices");
        return _linear_at(_get_position(indices));
    }

    // indexing with a tuple/array of integers
    template<typename Tuple>
    const _elem_t& tuple_at(const Tuple& indices) const &
    {
        static_assert(std::tuple_size_v<Tuple> == _depth_v, "incorrect number of indices");
        return _linear_at(_get_position(indices));
    }

    // indexing with multiple integers
    template<typename... Ints>
    _elem_t at(Ints... ints) &&
    {
        return std::move(*this).tuple_at(std::make_tuple(ints...));
    }

    // indexing with multiple integers
    template<typename... Ints>
    _elem_t& at(Ints... ints) &
    {
        return this->tuple_at(std::make_tuple(ints...));
    }

    // indexing with multiple integers
    template<typename... Ints>
    const _elem_t& at(Ints... ints) const &
    {
        return this->tuple_at(std::make_tuple(ints...));
    }

    // linear accessing
    _elem_t operator[](size_t pos) &&
    {
        return data_[pos];
    }

    // linear accessing
    _elem_t& operator[](size_t pos) &
    {
        return data_[pos];
    }

    // linear accessing
    const _elem_t& operator[](size_t pos) const &
    {
        return data_[pos];
    }

    _elem_t* data()
    {
        return data_.data();
    }
    const _elem_t* data() const
    {
        return data_.data();
    }

    simple_elem_iter<_elem_t> element_begin()
    {
        return {data()};
    }
    simple_elem_iter<_elem_t> element_end()
    {
        return {data() + size()};
    }
    simple_elem_const_iter<_elem_t> element_cbegin() const
    {
        return {data()};
    }
    simple_elem_const_iter<_elem_t> element_cend() const
    {
        return {data() + size()};
    }
    simple_elem_const_iter<_elem_t> element_begin() const
    {
        return this->element_cbegin();
    }
    simple_elem_const_iter<_elem_t> element_end() const
    {
        return this->element_cend();
    }

    template<bool IsExplicitConst, size_t Level>
    auto _begin_impl()
    {
        static_assert(0 < Level && Level <= _depth_v);
        if constexpr (Level == _depth_v)
        {
            return this->element_begin();
        }
        else
        {
            size_t ptr_stride = this->_total_size_impl<_depth_v, Level>();
            auto   sub_view   = this->tuple_vpart(repeat_tuple_t<Level, size_t>{});
            return regular_view_iter<decltype(sub_view), IsExplicitConst>{std::move(sub_view), ptr_stride};
        }
    }
    template<bool IsExplicitConst, size_t Level>
    auto _end_impl()
    {
        static_assert(0 < Level && Level <= _depth_v);
        if constexpr (Level == _depth_v)
        {
            return this->element_end();
        }
        else
        {
            auto iter = this->_begin_impl<IsExplicitConst, Level>();
            iter.my_base_ptr_ref() += size();
            return iter;
        }
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
            size_t ptr_stride = this->_total_size_impl<_depth_v, Level>();
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
            iter.my_base_ptr_ref() += size();
            return iter;
        }
    }

    template<size_t Level = 1>
    auto begin()
    {
        return _begin_impl<false, Level>();
    }
    template<size_t Level = 1>
    auto end()
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
    template<size_t Level = 1>
    auto begin() const
    {
        return cbegin();
    }
    template<size_t Level = 1>
    auto end() const
    {
        return cend();
    }

    _data_t _get_vector() &&
    {
        return std::move(data_);
    }
    _data_t& _get_vector() &
    {
        return data_;
    }
    const _data_t& _get_vector() const &
    {
        return data_;
    }

    template<typename SpanTuple>
    deduce_array_view_type_t<_elem_t, _indexers_t, SpanTuple>
        tuple_vpart(SpanTuple&& spans) &&
    {
        static_assert(_always_false_v<SpanTuple>, "cannot call tuple_vpart() on an r-value array.");
    }

    template<typename SpanTuple>
    deduce_array_view_type_t<_elem_t, _indexers_t, SpanTuple>
        tuple_vpart(SpanTuple&& spans) &
    {
        return get_collapsed_view(
            data(), dims_.data(), _indexers_t{}, std::forward<decltype(spans)>(spans));
    }

    template<typename SpanTuple>
    deduce_array_view_type_t<const _elem_t, _indexers_t, SpanTuple>
        tuple_vpart(SpanTuple&& spans) const &
    {
        return get_collapsed_view(
            data(), dims_.data(), _indexers_t{}, std::forward<decltype(spans)>(spans));
    }

    template<typename... Spans>
    deduce_array_view_type_t<_elem_t, _indexers_t, std::tuple<Spans...>>
        vpart(Spans&&... spans) &&
    {
        static_assert(_always_false_v<Spans...>, "cannot call vpart() on an r-value array.");
    }

    template<typename... Spans>
    deduce_array_view_type_t<_elem_t, _indexers_t, std::tuple<Spans...>>
        vpart(Spans&&... spans) &
    {
        return this->tuple_vpart(std::forward_as_tuple(spans...));
    }

    template<typename... Spans>
    deduce_array_view_type_t<const _elem_t, _indexers_t, std::tuple<Spans...>>
        vpart(Spans&&... spans) const &
    {
        return this->tuple_vpart(std::forward_as_tuple(spans...));
    }

    template<typename SpanTuple>
    deduce_part_array_type_t<_elem_t, _indexers_t, SpanTuple>
        tuple_part(SpanTuple&& spans) const
    {
        if constexpr (is_all_a_type_tuple_v<all_span, remove_cvref_t<SpanTuple>>)
            return *this;
        else
        {
            auto view = this->tuple_vpart(std::forward<decltype(spans)>(spans));
            return make_array(view);
        }
    }

    template<typename... Spans>
    deduce_part_array_type_t<_elem_t, _indexers_t, std::tuple<Spans...>>
        part(Spans&&... spans) const
    {
        return this->tuple_part(std::forward_as_tuple(spans...));
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


    // copy data to destination given size as hint, assuming no aliasing
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
        this->copy_to(dst, this->size());
    }

    // copy data from source given size as hint, assuming no aliasing
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
        this->copy_from(src, this->size());
    }

public:

    template<size_t LastLevel = _depth_v, size_t FirstLevel = 0>
    size_t _total_size_impl() const
    {
        static_assert(FirstLevel <= LastLevel && LastLevel <= _depth_v);
        if constexpr (FirstLevel == LastLevel)
            return size_t(1);
        else
            return dimension<LastLevel - 1>() * _total_size_impl<LastLevel - 1, FirstLevel>();
    }

    _elem_t _linear_at(size_t pos) &&
    {
        NDARRAY_ASSERT(pos < size());
        return data_[pos];
    }

    _elem_t& _linear_at(size_t pos) &
    {
        NDARRAY_ASSERT(pos < size());
        return data_[pos];
    }

    const _elem_t& _linear_at(size_t pos) const &
    {
        NDARRAY_ASSERT(pos < size());
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


// create array from array
template<typename T, size_t Depth>
auto make_array(const array<T, Depth>& arr)
{
    return arr;
}

// create array from array
template<typename T, size_t Depth>
auto make_array(array<T, Depth>&& arr)
{
    return std::move(arr);
}

// create array from std::vector<T>
template<typename T>
auto make_array(const std::vector<T>& data)
{
    return array<T, 1>(data, {data.size()});
}

// create array from std::vector<T>
template<typename T>
auto make_array(std::vector<T>&& data)
{
    return array<T, 1>(std::move(data), {data.size()});
}



}
