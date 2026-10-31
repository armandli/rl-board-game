#include <mlx_util/io.h>

#include <cstdint>
#include <cstring>

#include <algorithm>
#include <array>
#include <fstream>
#include <ios>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <mlx/io/load.h>

#if defined(RLBG_HAS_ZLIB)
#include <zlib.h>
#endif

namespace rlbg::nn {

namespace {

constexpr uint32_t LOCAL_SIG = 0x04034b50;
constexpr uint32_t CENTRAL_SIG = 0x02014b50;
constexpr uint32_t EOCD_SIG = 0x06054b50;
constexpr uint32_t EOCD64_SIG = 0x06064b50;
constexpr uint32_t EOCD64_LOCATOR_SIG = 0x07064b50;
constexpr uint16_t ZIP64_EXTRA_ID = 0x0001;
constexpr uint16_t METHOD_STORED = 0;
constexpr uint16_t METHOD_DEFLATE = 8;
constexpr uint64_t U32_MAX = 0xFFFFFFFFULL;
constexpr uint64_t U16_MAX = 0xFFFFULL;
// 1980-01-01 00:00, the earliest DOS timestamp
constexpr uint16_t DOS_DATE = 0x0021;
constexpr uint16_t DOS_TIME = 0;
constexpr s::string_view NPY_SUFFIX = ".npy";

bool ends_with(s::string_view str, s::string_view suffix) {
  return str.size() >= suffix.size() and
         str.substr(str.size() - suffix.size()) == suffix;
}

uint32_t crc32_of(const s::string& data) {
  static const s::array<uint32_t, 256> table = []() {
    s::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k)
        c = (c & 1) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
      t[i] = c;
    }
    return t;
  }();
  uint32_t c = 0xFFFFFFFFU;
  for (unsigned char ch : data)
    c = table[(c ^ ch) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFU;
}

// reads an in-memory buffer; mx::load keeps the reader alive for lazy reads
struct MemReader : mx::io::Reader {
  MemReader(s::shared_ptr<const s::string> buffer, s::string label)
    : mBuffer(s::move(buffer)), mLabel(s::move(label)) {}

  bool is_open() const override { return true; }
  bool good() const override { return mPos <= mBuffer->size(); }
  size_t tell() override { return mPos; }
  void seek(int64_t off, s::ios_base::seekdir way) override {
    int64_t base = way == s::ios_base::beg ? 0
                 : way == s::ios_base::cur ? (int64_t)mPos
                 : (int64_t)mBuffer->size();
    mPos = (size_t)(base + off);
  }
  void read(char* data, size_t n) override {
    read(data, n, mPos);
    mPos += n;
  }
  void read(char* data, size_t n, size_t offset) override {
    if (offset + n > mBuffer->size())
      throw s::runtime_error("[load_npz] truncated array data in " + mLabel);
    s::memcpy(data, mBuffer->data() + offset, n);
  }
  s::string label() const override { return mLabel; }

private:
  s::shared_ptr<const s::string> mBuffer;
  s::string mLabel;
  size_t mPos = 0;
};

struct MemWriter : mx::io::Writer {
  bool is_open() const override { return true; }
  bool good() const override { return true; }
  size_t tell() override { return mPos; }
  void seek(int64_t off, s::ios_base::seekdir way) override {
    int64_t base = way == s::ios_base::beg ? 0
                 : way == s::ios_base::cur ? (int64_t)mPos
                 : (int64_t)buffer.size();
    mPos = (size_t)(base + off);
  }
  void write(const char* data, size_t n) override {
    if (mPos + n > buffer.size()) buffer.resize(mPos + n);
    s::memcpy(buffer.data() + mPos, data, n);
    mPos += n;
  }
  s::string label() const override { return "memory"; }

  s::string buffer;

private:
  size_t mPos = 0;
};

struct ByteWriter {
  void u16(uint16_t v) {
    out.push_back((char)(v & 0xFF));
    out.push_back((char)(v >> 8));
  }
  void u32(uint32_t v) {
    u16((uint16_t)(v & 0xFFFF));
    u16((uint16_t)(v >> 16));
  }
  void u64(uint64_t v) {
    u32((uint32_t)(v & U32_MAX));
    u32((uint32_t)(v >> 32));
  }
  void bytes(s::string_view b) { out.append(b); }

  s::string out;
};

uint16_t rd16(s::string_view b, size_t at) {
  if (at + 2 > b.size()) throw s::runtime_error("[load_npz] corrupt zip file");
  return (uint16_t)((unsigned char)b[at] | ((unsigned char)b[at + 1] << 8));
}

uint32_t rd32(s::string_view b, size_t at) {
  return (uint32_t)rd16(b, at) | ((uint32_t)rd16(b, at + 2) << 16);
}

uint64_t rd64(s::string_view b, size_t at) {
  return (uint64_t)rd32(b, at) | ((uint64_t)rd32(b, at + 4) << 32);
}

#if defined(RLBG_HAS_ZLIB)
s::string deflate_raw(const s::string& data) {
  z_stream zs{};
  // negative window bits: raw deflate stream as zip stores it
  if (deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8,
                   Z_DEFAULT_STRATEGY) != Z_OK)
    throw s::runtime_error("[save_npz] deflateInit2 failed");
  s::string out(deflateBound(&zs, data.size()), '\0');
  zs.next_in = (Bytef*)data.data();
  zs.avail_in = (uInt)data.size();
  zs.next_out = (Bytef*)out.data();
  zs.avail_out = (uInt)out.size();
  int rc = ::deflate(&zs, Z_FINISH);
  deflateEnd(&zs);
  if (rc != Z_STREAM_END) throw s::runtime_error("[save_npz] deflate failed");
  out.resize(zs.total_out);
  return out;
}

s::string inflate_raw(s::string_view data, uint64_t size) {
  z_stream zs{};
  if (inflateInit2(&zs, -15) != Z_OK)
    throw s::runtime_error("[load_npz] inflateInit2 failed");
  s::string out(size, '\0');
  zs.next_in = (Bytef*)data.data();
  zs.avail_in = (uInt)data.size();
  zs.next_out = (Bytef*)out.data();
  zs.avail_out = (uInt)out.size();
  int rc = ::inflate(&zs, Z_FINISH);
  inflateEnd(&zs);
  if (rc != Z_STREAM_END) throw s::runtime_error("[load_npz] inflate failed");
  return out;
}
#endif

s::string to_npy(const mx::array& a) {
  auto w = s::make_shared<MemWriter>();
  mx::save(w, a);
  return s::move(w->buffer);
}

struct CentralEntry {
  s::string name;
  uint32_t crc;
  uint64_t csize;
  uint64_t usize;
  uint64_t offset;
  uint16_t method;
};

s::string local_header(const CentralEntry& e) {
  bool zip64 = e.csize >= U32_MAX or e.usize >= U32_MAX;
  ByteWriter w;
  w.u32(LOCAL_SIG);
  w.u16(zip64 ? 45 : 20);
  w.u16(0);
  w.u16(e.method);
  w.u16(DOS_TIME);
  w.u16(DOS_DATE);
  w.u32(e.crc);
  w.u32(zip64 ? (uint32_t)U32_MAX : (uint32_t)e.csize);
  w.u32(zip64 ? (uint32_t)U32_MAX : (uint32_t)e.usize);
  w.u16((uint16_t)e.name.size());
  w.u16(zip64 ? 20 : 0);
  w.bytes(e.name);
  if (zip64) {
    w.u16(ZIP64_EXTRA_ID);
    w.u16(16);
    w.u64(e.usize);
    w.u64(e.csize);
  }
  return w.out;
}

s::string central_header(const CentralEntry& e) {
  bool zip64 =
    e.csize >= U32_MAX or e.usize >= U32_MAX or e.offset >= U32_MAX;
  ByteWriter w;
  w.u32(CENTRAL_SIG);
  w.u16(zip64 ? 45 : 20);
  w.u16(zip64 ? 45 : 20);
  w.u16(0);
  w.u16(e.method);
  w.u16(DOS_TIME);
  w.u16(DOS_DATE);
  w.u32(e.crc);
  w.u32(zip64 ? (uint32_t)U32_MAX : (uint32_t)e.csize);
  w.u32(zip64 ? (uint32_t)U32_MAX : (uint32_t)e.usize);
  w.u16((uint16_t)e.name.size());
  w.u16(zip64 ? 28 : 0);
  w.u16(0);
  w.u16(0);
  w.u16(0);
  w.u32(0);
  w.u32(zip64 ? (uint32_t)U32_MAX : (uint32_t)e.offset);
  w.bytes(e.name);
  if (zip64) {
    w.u16(ZIP64_EXTRA_ID);
    w.u16(24);
    w.u64(e.usize);
    w.u64(e.csize);
    w.u64(e.offset);
  }
  return w.out;
}

s::string end_records(uint64_t count, uint64_t cd_offset, uint64_t cd_size) {
  bool zip64 = count >= U16_MAX or cd_offset >= U32_MAX or cd_size >= U32_MAX;
  ByteWriter w;
  if (zip64) {
    uint64_t eocd64_offset = cd_offset + cd_size;
    w.u32(EOCD64_SIG);
    w.u64(44);
    w.u16(45);
    w.u16(45);
    w.u32(0);
    w.u32(0);
    w.u64(count);
    w.u64(count);
    w.u64(cd_size);
    w.u64(cd_offset);
    w.u32(EOCD64_LOCATOR_SIG);
    w.u32(0);
    w.u64(eocd64_offset);
    w.u32(1);
  }
  w.u32(EOCD_SIG);
  w.u16(0);
  w.u16(0);
  w.u16(zip64 ? (uint16_t)U16_MAX : (uint16_t)count);
  w.u16(zip64 ? (uint16_t)U16_MAX : (uint16_t)count);
  w.u32(zip64 ? (uint32_t)U32_MAX : (uint32_t)cd_size);
  w.u32(zip64 ? (uint32_t)U32_MAX : (uint32_t)cd_offset);
  w.u16(0);
  return w.out;
}

s::string read_file(const s::string& file) {
  s::ifstream in(file, s::ios::binary);
  if (not in) throw s::runtime_error("[load_arrays] cannot open " + file);
  return s::string(s::istreambuf_iterator<char>(in), {});
}

// applies the zip64 extended information extra field to the entry fields
// that overflowed; they appear in the order usize, csize, offset
void apply_zip64_extra(s::string_view extra, CentralEntry& e) {
  size_t at = 0;
  while (at + 4 <= extra.size()) {
    uint16_t id = rd16(extra, at);
    uint16_t len = rd16(extra, at + 2);
    if (id == ZIP64_EXTRA_ID) {
      size_t p = at + 4;
      if (e.usize == U32_MAX) { e.usize = rd64(extra, p); p += 8; }
      if (e.csize == U32_MAX) { e.csize = rd64(extra, p); p += 8; }
      if (e.offset == U32_MAX) e.offset = rd64(extra, p);
      return;
    }
    at += 4 + len;
  }
}

} // namespace

bool has_npz_compression() {
#if defined(RLBG_HAS_ZLIB)
  return true;
#else
  return false;
#endif
}

void save_npz(const s::string& file, const Params& arrays, bool compressed) {
#if not defined(RLBG_HAS_ZLIB)
  if (compressed)
    throw s::runtime_error("[save_npz] built without zlib, cannot compress");
#endif
  s::ofstream out(file, s::ios::binary | s::ios::trunc);
  if (not out) throw s::runtime_error("[save_npz] cannot open " + file);

  s::vector<CentralEntry> entries;
  uint64_t offset = 0;
  for (const auto& [key, value] : arrays) {
    s::string npy = to_npy(value);
    CentralEntry e{
      .name = key + s::string(NPY_SUFFIX),
      .crc = crc32_of(npy),
      .csize = npy.size(),
      .usize = npy.size(),
      .offset = offset,
      .method = METHOD_STORED,
    };
#if defined(RLBG_HAS_ZLIB)
    if (compressed) {
      npy = deflate_raw(npy);
      e.csize = npy.size();
      e.method = METHOD_DEFLATE;
    }
#endif
    s::string header = local_header(e);
    out.write(header.data(), (s::streamsize)header.size());
    out.write(npy.data(), (s::streamsize)npy.size());
    offset += header.size() + npy.size();
    entries.push_back(s::move(e));
  }

  uint64_t cd_offset = offset;
  uint64_t cd_size = 0;
  for (const CentralEntry& e : entries) {
    s::string header = central_header(e);
    out.write(header.data(), (s::streamsize)header.size());
    cd_size += header.size();
  }
  s::string tail = end_records(entries.size(), cd_offset, cd_size);
  out.write(tail.data(), (s::streamsize)tail.size());
  if (not out) throw s::runtime_error("[save_npz] failed writing " + file);
}

Params load_npz(const s::string& file) {
  const s::string content = read_file(file);
  s::string_view b(content);

  // the end of central directory record is followed by a comment of up to
  // 64KiB, so scan backwards for its signature
  if (b.size() < 22) throw s::runtime_error("[load_npz] not a zip file: " + file);
  size_t eocd = s::string_view::npos;
  size_t lowest = b.size() > 22 + U16_MAX ? b.size() - 22 - U16_MAX : 0;
  for (size_t at = b.size() - 22; ; --at) {
    if (rd32(b, at) == EOCD_SIG) { eocd = at; break; }
    if (at == lowest) break;
  }
  if (eocd == s::string_view::npos)
    throw s::runtime_error("[load_npz] not a zip file: " + file);

  uint64_t count = rd16(b, eocd + 10);
  uint64_t cd_size = rd32(b, eocd + 12);
  uint64_t cd_offset = rd32(b, eocd + 16);
  if (eocd >= 20 and rd32(b, eocd - 20) == EOCD64_LOCATOR_SIG) {
    uint64_t eocd64 = rd64(b, eocd - 20 + 8);
    if (rd32(b, eocd64) != EOCD64_SIG)
      throw s::runtime_error("[load_npz] corrupt zip64 record in " + file);
    count = rd64(b, eocd64 + 32);
    cd_size = rd64(b, eocd64 + 40);
    cd_offset = rd64(b, eocd64 + 48);
  }
  if (cd_offset + cd_size > b.size())
    throw s::runtime_error("[load_npz] corrupt central directory in " + file);

  Params arrays;
  size_t at = cd_offset;
  for (uint64_t i = 0; i < count; ++i) {
    if (rd32(b, at) != CENTRAL_SIG)
      throw s::runtime_error("[load_npz] corrupt central directory in " + file);
    uint16_t name_len = rd16(b, at + 28);
    uint16_t extra_len = rd16(b, at + 30);
    uint16_t comment_len = rd16(b, at + 32);
    CentralEntry e{
      .name = s::string(b.substr(at + 46, name_len)),
      .crc = rd32(b, at + 16),
      .csize = rd32(b, at + 20),
      .usize = rd32(b, at + 24),
      .offset = rd32(b, at + 42),
      .method = rd16(b, at + 10),
    };
    apply_zip64_extra(b.substr(at + 46 + name_len, extra_len), e);
    at += 46 + name_len + extra_len + comment_len;

    if (rd32(b, e.offset) != LOCAL_SIG)
      throw s::runtime_error("[load_npz] corrupt local header in " + file);
    size_t data_at =
      e.offset + 30 + rd16(b, e.offset + 26) + rd16(b, e.offset + 28);
    if (data_at + e.csize > b.size())
      throw s::runtime_error("[load_npz] truncated entry in " + file);
    s::string_view raw = b.substr(data_at, e.csize);

    s::string data;
    if (e.method == METHOD_STORED) {
      data = s::string(raw);
    } else if (e.method == METHOD_DEFLATE) {
#if defined(RLBG_HAS_ZLIB)
      data = inflate_raw(raw, e.usize);
#else
      throw s::runtime_error(
        "[load_npz] built without zlib, cannot read compressed " + file
      );
#endif
    } else {
      throw s::runtime_error(
        "[load_npz] unsupported compression method in " + file
      );
    }
    if (crc32_of(data) != e.crc)
      throw s::runtime_error("[load_npz] CRC mismatch for " + e.name);

    s::string key = e.name;
    if (ends_with(key, NPY_SUFFIX)) key.resize(key.size() - NPY_SUFFIX.size());
    auto reader = s::make_shared<MemReader>(
      s::make_shared<const s::string>(s::move(data)),
      file + ":" + e.name
    );
    arrays.insert_or_assign(key, mx::load(reader));
  }
  return arrays;
}

Params load_safetensors(const s::string& file) {
  auto [arrays, metadata] = mx::load_safetensors(file);
  return Params(arrays.begin(), arrays.end());
}

void save_safetensors(
  const s::string& file,
  const Params& arrays,
  const s::unordered_map<s::string, s::string>& metadata)
{
  mx::save_safetensors(
    file,
    s::unordered_map<s::string, mx::array>(arrays.begin(), arrays.end()),
    metadata
  );
}

Params load_arrays(const s::string& file) {
  if (ends_with(file, ".npz")) return load_npz(file);
  if (ends_with(file, ".safetensors")) return load_safetensors(file);
  if (ends_with(file, ".gguf")) {
    auto [arrays, metadata] = mx::load_gguf(file);
    return Params(arrays.begin(), arrays.end());
  }
  throw s::invalid_argument(
    "[load_arrays] unsupported file extension for " + file +
    "; use .npz, .safetensors or .gguf"
  );
}

void save_arrays(const s::string& file, const Params& arrays) {
  if (ends_with(file, ".npz")) return save_npz(file, arrays);
  if (ends_with(file, ".safetensors")) return save_safetensors(file, arrays);
  throw s::invalid_argument(
    "[save_arrays] unsupported file extension for " + file +
    "; use .npz or .safetensors"
  );
}

} // namespace rlbg::nn
