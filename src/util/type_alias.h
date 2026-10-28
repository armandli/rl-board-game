#ifndef RLBG_TYPE_ALIAS_H
#define RLBG_TYPE_ALIAS_H

#include <cstdint>

#include <unordered_map>
#include <unordered_set>

using ubyte = uint8_t;
using byte  = int8_t;
using uint  = uint32_t;

template <typename K, typename V>
using umap = std::unordered_map<K, V>;

template <typename K>
using uset = std::unordered_set<K>;

#endif//RLBG_TYPE_ALIAS_H
