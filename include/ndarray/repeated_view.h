#pragma once

#include "traits.h"
#include "array.h"

namespace ndarray
{

template<typename T>
class repeated_view_elem_iter
{
public:
    using _my_type = repeated_view_elem_iter;
    using _elem_t  = T;
    static constexpr bool _is_const_v = true;

protected:
    _elem_t val_;
    size_t  pos_;

public:
    repeated_view_elem_iter(_elem_t val, size_t pos) : 
        val_{val}, pos_{pos} {}

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
        return _my_type{val_, pos_ + diff};
    }
    _my_type operator-(ptrdiff_t diff) const
    {
        return _my_type{val_, pos_ - diff};
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
        return val_;
    }
    _elem_t operator[](ptrdiff_t diff) const
    {
        return val_;
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return other.pos_ - this->pos_;
    }

    bool operator==(const _my_type& other) const
    {
        return other.pos_ == this->pos_;
    }
    bool operator<(const _my_type& other) const
    {
        return other.pos_ < this->pos_;
    }
    bool operator>(const _my_type& other) const
    {
        return other.pos_ > this->pos_;
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

template<typename SubView>
class repeated_view_iter
{
public:
    using _my_type    = repeated_view_iter;
    using _sub_view_t = SubView;
    using _elem_t     = typename _sub_view_t::_elem_t;
    static constexpr bool _is_const_v = true;

protected:
    _sub_view_t sub_view_;
    size_t      pos_;

public:
    repeated_view_iter(_sub_view_t sub_view, size_t pos) : 
        sub_view_{sub_view}, pos_{pos} {}

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
        return _my_type{sub_view_, pos_ + diff};
    }
    _my_type operator-(ptrdiff_t diff) const
    {
        return _my_type{sub_view_, pos_ - diff};
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

    _sub_view_t operator*() const
    {
        return sub_view_;
    }
    _sub_view_t operator[](ptrdiff_t diff) const
    {
        return sub_view_;
    }
    ptrdiff_t operator-(const _my_type& other) const
    {
        return other.pos_ - this->pos_;
    }

    bool operator==(const _my_type& other) const
    {
        return other.pos_ == this->pos_;
    }
    bool operator<(const _my_type& other) const
    {
        return other.pos_ < this->pos_;
    }
    bool operator>(const _my_type& other) const
    {
        return other.pos_ > this->pos_;
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
// repeat_elem_view is equivalent to an array consists of only one value.
// It stores the dimensions, but does not check bounds when accessing. 
//

template<typename T, size_t Depth>
class repeated_view
{
public:
    using _my_type    = repeated_view;
    using _elem_t     = T;
    static constexpr size_t _depth_v    = Depth;
    static constexpr bool   _is_const_v = true;
    using _indexers_t = n_all_indexer_tuple_t<_depth_v>;
    using _dims_t     = std::array<size_t, _depth_v>;
    static_assert(_depth_v > 0);

protected:
    _elem_t val_;
    _dims_t dims_;

public:
    repeated_view(_elem_t val, _dims_t dims) :
        val_{val}, dims_{dims} {}

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

    repeated_view_elem_iter<_elem_t> element_cbegin() const
    {
        return {val_, size_t(0)};
    }
    repeated_view_elem_iter<_elem_t> element_cend() const
    {
        return {val_, size()};
    }
    repeated_view_elem_iter<_elem_t> element_begin() const
    {
        return this->element_cbegin();
    }
    repeated_view_elem_iter<_elem_t> element_end() const
    {
        return this->element_cend();
    }
    

    template<size_t Level = 1>
    auto cbegin() const
    {
        static_assert(0 < Level && Level <= _depth_v);
        if constexpr (Level == _depth_v)
        {
            return this->element_begin();
        }
        else
        {
            using sub_t    = repeated_view<_elem_t, _depth_v - Level>;
            sub_t sub_view = this->tuple_vpart(repeat_tuple_t<Level, size_t>{});
            using iter_t   = repeated_view_iter<sub_t>;
            return iter_t{sub_view, size_t(0)};
        }
    }
    template<size_t Level = 1>
    auto cend() const
    {
        static_assert(0 < Level && Level <= _depth_v);
        if constexpr (Level == _depth_v)
        {
            return this->element_end();
        }
        else
        {
            using  iter_t = repeated_view_iter<repeated_view<_elem_t, _depth_v - Level>>;
            iter_t iter   = this->begin<Level>();
            iter += this->size<Level, 0>();
            return iter;
        }
    }
    template<size_t Level = 1>
    auto begin() const
    {
        return this->cbegin<Level>();
    }
    template<size_t Level = 1>
    auto end() const
    {
        return this->cend<Level>();
    }

    // automatically calls at() or vpart(), depending on its arguments
    template<typename... Anys>
    deduce_repeated_view_or_elem_type_t<_elem_t, _elem_t, _depth_v, std::tuple<Anys...>>
        operator()(Anys&&... anys) const
    {
        constexpr bool is_complete_index = sizeof...(Anys) == _depth_v && is_all_ints_v<Anys...>;
        if constexpr (is_complete_index)
            return val_;
        else
            return this->tuple_vpart(std::forward_as_tuple(anys...));
    }

    // indexing with a tuple/array of integers
    template<typename Tuple>
    _elem_t tuple_at(const Tuple&) const
    {
        static_assert(std::tuple_size_v<Tuple> == _depth_v, "incorrect number of indices");
        return val_;
    }

    // indexing with multiple integers
    template<typename... Ints>
    _elem_t at(Ints... ints) const
    {
        return tuple_at(std::make_tuple(ints...));
    }

    // linear accessing
    _elem_t operator[](size_t pos) const
    {
        return val_;
    }

    template<typename SpanTuple>
    deduce_repeated_view_type_t<_elem_t, _depth_v, SpanTuple>
        tuple_vpart(SpanTuple&& spans) const
    {
        using result_t = deduce_repeated_view_type_t<_elem_t, _depth_v, SpanTuple>;
        constexpr size_t new_depth_v = result_t::_depth_v;
        std::array<size_t, new_depth_v> new_dims;
        _tuple_vpart_impl(new_dims.data(), spans);
        return result_t{val_, new_dims};
    }

    template<typename... Spans>
    deduce_repeated_view_type_t<_elem_t, _depth_v, std::tuple<Spans...>>
        vpart(Spans&&... spans) const
    {
        return this->tuple_vpart(std::forward_as_tuple(spans...));
    }

    // copy data to destination given size
    template<typename Iter>
    void copy_to(Iter dst, size_t size) const
    {
        for (size_t i = 0; i < size; ++i)
        {
            *dst = val_;
            ++dst;
        }
    }
    // copy data to destination given size
    template<typename Iter>
    void copy_to(Iter dst) const
    {
        size_t size = this->size();
        for (size_t i = 0; i < size; ++i, ++dst)
            *dst = val_;
    }

private:
    template<size_t NewLevel = 0, size_t Level = 0, typename SpanTuple>
    void _tuple_vpart_impl(size_t* new_dims, SpanTuple&& spans) const
    {
        constexpr size_t span_size_v = std::tuple_size_v<remove_cvref_t<SpanTuple>>;
        size_t dim_i = dimension<Level>();
        if constexpr (Level >= span_size_v)
        {
            new_dims[NewLevel] = dim_i; // must be all_indexer
            if constexpr (Level + 1 < _depth_v)
                _tuple_vpart_impl<NewLevel + 1, Level + 1>(new_dims, std::forward<decltype(spans)>(spans));
        }
        else
        {
            using indexer_t = indexer_collapsing_t<all_indexer,
                std::tuple_element_t<_mp_min_v<Level, span_size_v - 1>, remove_cvref_t<SpanTuple>>>;
            if constexpr (std::is_same_v<indexer_t, scalar_indexer>)
            { // skip this level
                if constexpr (Level + 1 < _depth_v)
                    _tuple_vpart_impl<NewLevel, Level + 1>(new_dims, std::forward<decltype(spans)>(spans));
            }
            else
            {
                indexer_t indexer = collapse_indexer(
                    dim_i, all_indexer{}, std::get<Level>(std::forward<decltype(spans)>(spans))).second;
                new_dims[NewLevel] = indexer.size(dim_i);
                if constexpr (Level + 1 < _depth_v)
                    _tuple_vpart_impl<NewLevel + 1, Level + 1>(new_dims, std::forward<decltype(spans)>(spans));
            }
        }
    }
};


// create array from repeated_view
template<typename T, size_t Depth>
inline auto make_array(const repeated_view<T, Depth>& view)
{
    return array<T, Depth>{std::vector<T>(view.size(), view[0]), view.dimensions()};
}

}

