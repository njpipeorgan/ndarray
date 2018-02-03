#pragma once

#include "traits.h"
#include "array.h"
#include "array_view.h"

namespace ndarray
{

template<typename SrcArray, typename DstArray>
inline void aliased_data_copy(const SrcArray& src, DstArray& dst, size_t size)
{
    using src_t = remove_cvref_t<SrcArray>;
    using dst_t = remove_cvref_t<DstArray>;
    constexpr view_type src_type_v = src_t::_my_view_type_v;
    constexpr view_type dst_type_v = dst_t::_my_view_type_v;

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
    using src_t = remove_cvref_t<SrcArray>;
    using dst_t = remove_cvref_t<DstArray>;
    constexpr view_type src_type_v = src_t::_my_view_type_v;
    constexpr view_type dst_type_v = dst_t::_my_view_type_v;

    if constexpr (src_type_v == view_type::array)
        dst.copy_from(src.data(), size);
    else if constexpr (dst_type_v == view_type::array)
        src.copy_to(dst.data(), size);
    else if constexpr (src_type_v != view_type::irregular)
        dst.copy_from(src.element_begin(), size);
    else if constexpr (dst_type_v != view_type::irregular)
        src.copy_to(dst.element_begin(), size);
    else // both arrays are irregular_array_view
        aliased_data_copy(src, dst, size);
}

template<typename SrcArray, typename DstArray>
inline void data_copy(const SrcArray& src, DstArray& dst)
{
    using src_t = remove_cvref_t<SrcArray>;
    using dst_t = remove_cvref_t<DstArray>;
    constexpr view_type src_type_v = src_t::_my_view_type_v;
    constexpr view_type dst_type_v = dst_t::_my_view_type_v;

    assert(src.has_same_dimensions(dst));
    size_t size = src_type_v == view_type::array ? src.size() : dst.size();

    if (src._identifier_ptr() != dst._identifier_ptr())
    {
        no_alias_data_copy(src, dst, size);
    }
    else if constexpr (src_type_v == view_type::array &&
                       dst_type_v == view_type::array) // no copy will happen
    {
    }
    else if constexpr (src_type_v == view_type::array     ||
                       dst_type_v == view_type::array     ||
                       src_type_v == view_type::irregular ||
                       dst_type_v == view_type::irregular)
    {
        aliased_data_copy(src, dst, size);
    }
    else
    {
        const auto src_ptr = src.base_ptr();
        const auto dst_ptr = dst.base_ptr();
        if constexpr (src_type_v == view_type::simple &&
                      dst_type_v == view_type::simple)
        {
            if (src_ptr + size <= dst_ptr ||
                dst_ptr + size <= src_ptr)
                no_alias_data_copy(src, dst, size);
            else
                aliased_data_copy(src, dst, size);
        }
        else if (src_type_v == view_type::regular &&
                 dst_type_v == view_type::regular &&
                 src.stride() == dst.stride()) // two regular_view with the same stride
        {
            const auto stride = src.stride();
            if ((src_ptr - dst_ptr) % stride != 0  ||
                src_ptr + size * stride <= dst_ptr ||
                dst_ptr + size *stride <= src_ptr)
                no_alias_data_copy(src, dst, size);
            else
                aliased_data_copy(src, dst, size);
        }
        else // two non-irregular views with different stride
        {
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
