#pragma once
#include <array>
#include <cstddef>
#include <cstring>

namespace sprite_policy {
// Dedicated SOCK_SEQPACKET message, distinct from every impedance target.
inline constexpr std::array<char,8> operator_disable_packet{'S','P','R','T',1,0,3,0};
inline constexpr char operator_disable_reason[]="Xbox operator disable request";
inline bool is_operator_disable(const void* bytes,std::size_t count) {
    return count==operator_disable_packet.size() &&
        std::memcmp(bytes,operator_disable_packet.data(),count)==0;
}
}
