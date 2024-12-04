/* This file is auto-generated */
/* Command Options: player_size=2 table_size=64 */
#ifndef RLBG_ZOBRIST_HASH
#define RLBG_ZOBRIST_HASH
#include <cassert>

#include <type_alias.h>

namespace rlbg {

uint zobrist_hash(uint player, uint p_index){
  uint key = p_index;
  key |= player << 30U;
  switch (key){
    case 0x40000000U: return 0xf13a25a3U; break;
    case 0x40000001U: return 0xda94dae1U; break;
    case 0x40000002U: return 0xd759af2dU; break;
    case 0x40000003U: return 0xe479f520U; break;
    case 0x40000004U: return 0x3f1307e9U; break;
    case 0x40000005U: return 0xfcc5c41bU; break;
    case 0x40000006U: return 0x52107860U; break;
    case 0x40000007U: return 0xc4ce9660U; break;
    case 0x40000008U: return 0x967aa007U; break;
    case 0x40000009U: return 0x58333c4cU; break;
    case 0x4000000aU: return 0x718be273U; break;
    case 0x4000000bU: return 0x46762343U; break;
    case 0x4000000cU: return 0x853e4691U; break;
    case 0x4000000dU: return 0x307c1499U; break;
    case 0x4000000eU: return 0x2bc1abf3U; break;
    case 0x4000000fU: return 0x7d072021U; break;
    case 0x40000010U: return 0x4067add4U; break;
    case 0x40000011U: return 0x286f10ffU; break;
    case 0x40000012U: return 0xd2364709U; break;
    case 0x40000013U: return 0x26904574U; break;
    case 0x40000014U: return 0xd23b1a5fU; break;
    case 0x40000015U: return 0xe87f1292U; break;
    case 0x40000016U: return 0x667052b3U; break;
    case 0x40000017U: return 0x7b4fb710U; break;
    case 0x40000018U: return 0x871c8c29U; break;
    case 0x40000019U: return 0xcdd0f80fU; break;
    case 0x4000001aU: return 0x7c6ecc2fU; break;
    case 0x4000001bU: return 0xacce2b5dU; break;
    case 0x4000001cU: return 0xeb9c7fb2U; break;
    case 0x4000001dU: return 0x3ac04f69U; break;
    case 0x4000001eU: return 0xefcfe902U; break;
    case 0x4000001fU: return 0xbc5d54f9U; break;
    case 0x40000020U: return 0x37d791a7U; break;
    case 0x40000021U: return 0x75821d8aU; break;
    case 0x40000022U: return 0xd28ecd8aU; break;
    case 0x40000023U: return 0x3c08accdU; break;
    case 0x40000024U: return 0xf5fe637cU; break;
    case 0x40000025U: return 0x742b5690U; break;
    case 0x40000026U: return 0x4b15c84bU; break;
    case 0x40000027U: return 0x6c1122bcU; break;
    case 0x40000028U: return 0xf964adffU; break;
    case 0x40000029U: return 0x77c32015U; break;
    case 0x4000002aU: return 0x7c9ae41fU; break;
    case 0x4000002bU: return 0x229c3084U; break;
    case 0x4000002cU: return 0xe2e55ae5U; break;
    case 0x4000002dU: return 0x1ccf7665U; break;
    case 0x4000002eU: return 0x8ea68c9eU; break;
    case 0x4000002fU: return 0x74f1aadbU; break;
    case 0x40000030U: return 0xa1135136U; break;
    case 0x40000031U: return 0xea82bc52U; break;
    case 0x40000032U: return 0xbf6466faU; break;
    case 0x40000033U: return 0x66478b32U; break;
    case 0x40000034U: return 0x73c859e2U; break;
    case 0x40000035U: return 0x825d9a44U; break;
    case 0x40000036U: return 0x3ddfa340U; break;
    case 0x40000037U: return 0x6bd3665dU; break;
    case 0x40000038U: return 0x88f2228eU; break;
    case 0x40000039U: return 0xe4dba061U; break;
    case 0x4000003aU: return 0xe5ce18bdU; break;
    case 0x4000003bU: return 0xdda2470fU; break;
    case 0x4000003cU: return 0xd83e9900U; break;
    case 0x4000003dU: return 0x49294938U; break;
    case 0x4000003eU: return 0x18f1fcd3U; break;
    case 0x4000003fU: return 0x6ffc8b29U; break;
    case 0x80000000U: return 0x183c073eU; break;
    case 0x80000001U: return 0xbe0ef19fU; break;
    case 0x80000002U: return 0x8e085a09U; break;
    case 0x80000003U: return 0x343ae6c0U; break;
    case 0x80000004U: return 0xf19e4dc1U; break;
    case 0x80000005U: return 0x78b09eceU; break;
    case 0x80000006U: return 0xcc34b171U; break;
    case 0x80000007U: return 0x550608c7U; break;
    case 0x80000008U: return 0x577dde8fU; break;
    case 0x80000009U: return 0xbcd2d3f4U; break;
    case 0x8000000aU: return 0x919b0cb8U; break;
    case 0x8000000bU: return 0x359257beU; break;
    case 0x8000000cU: return 0xd77ce367U; break;
    case 0x8000000dU: return 0xd4a160f9U; break;
    case 0x8000000eU: return 0x5bc1e0b5U; break;
    case 0x8000000fU: return 0xdfb9129aU; break;
    case 0x80000010U: return 0xf79b8a01U; break;
    case 0x80000011U: return 0x4e2b5529U; break;
    case 0x80000012U: return 0x1ceb34edU; break;
    case 0x80000013U: return 0x9fa607e6U; break;
    case 0x80000014U: return 0x8a10e1cbU; break;
    case 0x80000015U: return 0xb699796fU; break;
    case 0x80000016U: return 0x921f8813U; break;
    case 0x80000017U: return 0x3beb4b60U; break;
    case 0x80000018U: return 0xcbc2d146U; break;
    case 0x80000019U: return 0xa3a63bc9U; break;
    case 0x8000001aU: return 0x2d1713f6U; break;
    case 0x8000001bU: return 0xcdb00c9bU; break;
    case 0x8000001cU: return 0xccf0a1d2U; break;
    case 0x8000001dU: return 0xc69be284U; break;
    case 0x8000001eU: return 0xe0a39492U; break;
    case 0x8000001fU: return 0x63a78df9U; break;
    case 0x80000020U: return 0x5db54726U; break;
    case 0x80000021U: return 0xb75fadb1U; break;
    case 0x80000022U: return 0x87f3561fU; break;
    case 0x80000023U: return 0x20c448f5U; break;
    case 0x80000024U: return 0x9101ab6eU; break;
    case 0x80000025U: return 0x84d4525cU; break;
    case 0x80000026U: return 0xf48237daU; break;
    case 0x80000027U: return 0x6e07f029U; break;
    case 0x80000028U: return 0x90dfaaf3U; break;
    case 0x80000029U: return 0x3380d6d0U; break;
    case 0x8000002aU: return 0x1b159c36U; break;
    case 0x8000002bU: return 0x578a0db5U; break;
    case 0x8000002cU: return 0xce4c1563U; break;
    case 0x8000002dU: return 0x743b32b8U; break;
    case 0x8000002eU: return 0xcb5c413dU; break;
    case 0x8000002fU: return 0x7f3eaab4U; break;
    case 0x80000030U: return 0x9bee5805U; break;
    case 0x80000031U: return 0xd671c793U; break;
    case 0x80000032U: return 0x7979cd3dU; break;
    case 0x80000033U: return 0x8d63e0eaU; break;
    case 0x80000034U: return 0xeba147bcU; break;
    case 0x80000035U: return 0xfbbb0e10U; break;
    case 0x80000036U: return 0xad522b7eU; break;
    case 0x80000037U: return 0xf9aa7c05U; break;
    case 0x80000038U: return 0x540577d1U; break;
    case 0x80000039U: return 0x89f88ec2U; break;
    case 0x8000003aU: return 0x34316587U; break;
    case 0x8000003bU: return 0xd41ae14aU; break;
    case 0x8000003cU: return 0x8ac1c26fU; break;
    case 0x8000003dU: return 0x6c4d6a0dU; break;
    case 0x8000003eU: return 0xacbaef00U; break;
    case 0x8000003fU: return 0xfdb0168aU; break;
    default: assert(false);
  }
}
uint zobrist_hash(uint player){
  uint key = player << 30U;
  switch (key){
    case 0x40000000U: return 0x4c5549b3U; break;
    case 0x80000000U: return 0x1f71a1c0U; break;
    default: assert(false);
  }
}

constexpr uint EMPTY_BOARD = 0U;

} // rlbg
#endif//RLBG_ZOBRIST_HASH

