#pragma once

#include "traits.h"
#include "array.h"
#include "array_view.h"
#include "range_view.h"

namespace ndarray
{

template<typename T, size_t Depth>
inline auto get_vector(const array<T, Depth>& src)
{
    return src._get_vector();
}
template<typename T, size_t Depth>
inline auto get_vector(array<T, Depth>&& src)
{
    return std::move(src._get_vector());
}
template<typename T>
inline auto get_vector(const std::vector<T>& vec)
{
    return vec;
}
template<typename T>
inline auto get_vector(std::vector<T>&& vec)
{
    return std::move(vec);
}

template<typename View>
inline auto begin(View& view)
{
    return view.begin();
}
template<typename View>
inline auto end(View& view)
{
    return view.end();
}
template<typename View>
inline auto cbegin(View& view)
{
    return view.cbegin();
}
template<typename View>
inline auto cend(View& view)
{
    return view.cend();
}

template<size_t Level, typename View>
inline auto begin(View& view)
{
    return view.begin<Level>();
}
template<size_t Level, typename View>
inline auto end(View& view)
{
    return view.end<Level>();
}
template<size_t Level, typename View>
inline auto cbegin(View& view)
{
    return view.cbegin<Level>();
}
template<size_t Level, typename View>
inline auto cend(View& view)
{
    return view.cend<Level>();
}

template<typename View>
inline auto element_begin(View& view)
{
    return view.element_begin();
}
template<typename View>
inline auto element_end(View& view)
{
    return view.element_end();
}
template<typename View>
inline auto element_cbegin(View& view)
{
    return view.element_cbegin();
}
template<typename View>
inline auto element_cend(View& view)
{
    return view.element_cend();
}
template<typename T>
inline auto element_begin(std::vector<int>& vec)
{
    return vec.begin();
}
template<typename T>
inline auto element_begin(const std::vector<int>& vec)
{
    return vec.begin();
}
template<typename T>
inline auto element_cbegin(std::vector<int>& vec)
{
    return vec.cbegin();
}
template<typename T>
inline auto element_cbegin(const std::vector<int>& vec)
{
    return vec.begin();
}

template<typename View>
inline auto dimensions(const View& view)
{
    return view.dimensions();
}
template<typename T>
inline auto dimensions(const std::vector<T>& vec)
{
    return std::array<size_t, 1>{vec.size()};
}

// handles data copy between array and array view
template<typename SrcArray, typename DstArray>
inline void data_copy(const SrcArray& src, DstArray& dst)
{
    using _type = array_obj_type;
    using src_t = remove_cvref_t<SrcArray>;
    using dst_t = remove_cvref_t<DstArray>;
    constexpr _type src_type_v = array_obj_type_of_v<src_t>;
    constexpr _type dst_type_v = array_obj_type_of_v<dst_t>;

    static_assert(dst_type_v == _type::array     ||
                  dst_type_v == _type::simple    ||
                  dst_type_v == _type::regular   ||
                  dst_type_v == _type::irregular);
    static_assert(src_type_v == _type::vector    ||
                  src_type_v == _type::array     ||
                  src_type_v == _type::simple    ||
                  src_type_v == _type::regular   ||
                  src_type_v == _type::irregular ||
                  src_type_v == _type::range);

    if constexpr (src_type_v == _type::vector || 
                  src_type_v == _type::range)
    {
        size_t src_size = src.size();
        assert(dst.size() == src_size);
        no_alias_data_copy(src, dst, src_size);
    }
    else
    {
        assert(dst.check_size_with(src));
        size_t size = src_type_v == _type::array ? src.size() : dst.size();

        if (src._identifier_ptr() != dst._identifier_ptr())
        {
            no_alias_data_copy(src, dst, size);
        }
        else if constexpr (src_type_v == _type::array &&
                           dst_type_v == _type::array)
        { // no copy will happen between two identical arrays
        }
        else if constexpr (src_type_v == _type::array     ||
                           dst_type_v == _type::array     ||
                           src_type_v == _type::irregular ||
                           dst_type_v == _type::irregular)
        { // must be aliased or be unable to distinguish
            aliased_data_copy(src, dst, size);
        }
        else // copy between two simple_view/regular_view
        {
            const auto src_ptr = src.base_ptr();
            const auto dst_ptr = dst.base_ptr();
            if constexpr (src_type_v == _type::simple &&
                          dst_type_v == _type::simple)
            {
                if (src_ptr + size <= dst_ptr ||
                    dst_ptr + size <= src_ptr)
                    no_alias_data_copy(src, dst, size); // not overlaped
                else
                    aliased_data_copy(src, dst, size);  // overlaped
            }
            else if (src_type_v == _type::regular &&
                     dst_type_v == _type::regular &&
                     src.stride() == dst.stride())
            { // two regular_view with the same stride 
                const auto stride = src.stride();
                if ((src_ptr - dst_ptr) % stride != 0  ||
                    src_ptr + size * stride <= dst_ptr ||
                    dst_ptr + size * stride <= src_ptr)
                    no_alias_data_copy(src, dst, size);
                else
                    aliased_data_copy(src, dst, size);
            }
            else
            { // two simple_view/regular_view with different stride
                auto src_stride = src.stride();
                auto dst_stride = dst.stride();
                if (src_ptr + size * src_stride <= dst_ptr ||
                    dst_ptr + size * dst_stride <= src_ptr)
                    no_alias_data_copy(src, dst, size);
                else
                    aliased_data_copy(src, dst, size);
            }
        }

    }
}

template<typename SrcArray, typename DstArray>
inline void aliased_data_copy(const SrcArray& src, DstArray& dst, size_t size)
{
    NDARRAY_DEBUG(puts("aliased_data_copy() called."));
    using src_t = remove_cvref_t<SrcArray>;
    using dst_t = remove_cvref_t<DstArray>;
    constexpr array_obj_type src_type_v = array_obj_type_of_v<src_t>;
    constexpr array_obj_type dst_type_v = array_obj_type_of_v<dst_t>;

    using temp_type = std::conditional_t<
        sizeof(typename src_t::_elem_t{}) < sizeof(typename dst_t::_elem_t{}),
        typename src_t::_elem_t, typename dst_t::_elem_t>;
    std::vector<temp_type> temp(src.size());

    src.copy_to(temp.begin(), size);    // copy src to temp
    dst.copy_from(temp.begin(), size);  // then copy temp to dst
}

template<typename SrcArray, typename DstArray>
inline void no_alias_data_copy(const SrcArray& src, DstArray& dst, size_t size)
{
    NDARRAY_DEBUG(puts("no_alias_data_copy() called."));
    using _type = array_obj_type;
    using src_t = remove_cvref_t<SrcArray>;
    using dst_t = remove_cvref_t<DstArray>;
    constexpr _type src_type_v = array_obj_type_of_v<src_t>;
    constexpr _type dst_type_v = array_obj_type_of_v<dst_t>;

    if constexpr (src_type_v == _type::vector ||
                  src_type_v == _type::array  ||
                  src_type_v == _type::range)
        dst.copy_from(element_cbegin(src), size);
    else if constexpr (dst_type_v == _type::array)
        src.copy_to(dst.data(), size);
    else if constexpr (src_type_v != _type::irregular)
        dst.copy_from(element_cbegin(src), size);
    else if constexpr (dst_type_v != _type::irregular)
        dst.copy_from(element_cbegin(src), size);
    else // both arrays are irregular_array_view
        aliased_data_copy(src, dst, size);
}



}

