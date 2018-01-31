#pragma once

#include "traits.h"
#include "array.h"
#include "array_view.h"

namespace ndarray
{

//
// three interfaces for accessing the data of an array / array view
//
// _access_type::vector :
//     get_vector(src)
//
// _access_type::iterator : 
//     begin(src),         end(src),
//     cbegin(src),        cend(src)
//     begin<Level>(src),  end<Level>(src),
//     cbegin<Level>(src), cend<Level>(src)
//
// _access_type::traverse : 
//     traverse(fn, src)
//

template<typename T, size_t Depth>
inline std::vector<T> get_vector(const array<T, Depth>& src)
{
    return src._get_vector();
}
template<typename T, size_t Depth>
inline std::vector<T>&& get_vector(array<T, Depth>&& src)
{
    return std::move(src._get_vector());
}
template<typename T>
inline std::vector<T> get_vector(const std::vector<T>& vec)
{
    return vec;
}
template<typename T>
inline std::vector<T>&& get_vector(std::vector<T>&& vec)
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

template<typename Function, typename View>
inline void traverse(Function fn, View& view)
{
    view.traverse<Function>(fn);
}



}

