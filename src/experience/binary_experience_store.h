#ifndef RLBG_EXPERIENCE_BINARY_EXPERIENCE_STORE_H
#define RLBG_EXPERIENCE_BINARY_EXPERIENCE_STORE_H

#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <experience/experience_store.h>
#include <experience/trajectory.h>

namespace rlbg {

namespace s = std;
namespace sf = std::filesystem;

// appends trajectories to rotating files dir/games_NNNNNN.bin, each a
// sequence of records:
//   u32 magic "RLTJ", u32 version, i32 length, i32 observation_size,
//   i32 num_tokens, i32 num_actions, i8 outcome, u32 model_version,
//   then features, tokens, to_play, actions, rewards, root_values,
//   policies, policy_valid, final_tokens, value_versions as raw little
//   endian arrays. Version 1 records have no value_versions; their root
//   values load as unversioned.
struct BinaryExperienceStore : ExperienceStore {
  static constexpr uint32_t MAGIC = 0x4a544c52; // "RLTJ"
  static constexpr uint32_t VERSION = 2;

  BinaryExperienceStore(const s::string& dir, int games_per_file = 1000)
  : mDir(dir), mGamesPerFile(games_per_file)
  {
    sf::create_directories(mDir);
    for (const sf::path& p : files())
      mNextFile = s::max(mNextFile, file_index(p) + 1);
  }
  ~BinaryExperienceStore() override {
    flush();
  }

  void write(const Trajectory& t) override {
    s::lock_guard<s::mutex> lock(mMutex);
    if (not mOut.is_open() or mGamesInFile >= mGamesPerFile){
      if (mOut.is_open()) mOut.close();
      mOut.open(mDir / s::format("games_{:06d}.bin", mNextFile++), s::ios::binary | s::ios::app);
      mGamesInFile = 0;
    }
    put<uint32_t>(MAGIC);
    put<uint32_t>(VERSION);
    put<int32_t>(t.length);
    put<int32_t>(t.observation_size);
    put<int32_t>(t.num_tokens);
    put<int32_t>(t.num_actions);
    put<int8_t>(t.outcome);
    put<uint32_t>(t.model_version);
    put_vec(t.features);
    put_vec(t.tokens);
    put_vec(t.to_play);
    put_vec(t.actions);
    put_vec(t.rewards);
    {
      s::lock_guard<s::mutex> plock(t.policy_mutex);
      put_vec(t.root_values);
      put_vec(t.policies);
      put_vec(t.policy_valid);
      put_vec(t.final_tokens);
      put_vec(t.value_versions);
    }
    ++mGamesInFile;
  }

  void flush() override {
    s::lock_guard<s::mutex> lock(mMutex);
    if (mOut.is_open()) mOut.flush();
  }

  s::vector<s::shared_ptr<Trajectory>> load_recent(size_t max_positions) override {
    flush();
    s::vector<sf::path> fs = files();
    s::vector<s::vector<s::shared_ptr<Trajectory>>> per_file;
    size_t positions = 0;
    for (auto it = fs.rbegin(); it != fs.rend() and positions < max_positions; ++it){
      per_file.push_back(read_file(*it));
      for (const auto& t : per_file.back()) positions += t->length;
    }
    s::vector<s::shared_ptr<Trajectory>> out;
    for (auto it = per_file.rbegin(); it != per_file.rend(); ++it)
      for (auto& t : *it) out.push_back(s::move(t));
    // drop the oldest beyond max_positions
    size_t drop = 0;
    while (drop < out.size() and positions - out[drop]->length >= max_positions){
      positions -= out[drop]->length;
      ++drop;
    }
    out.erase(out.begin(), out.begin() + drop);
    return out;
  }

protected:
  s::vector<sf::path> files() const {
    s::vector<sf::path> out;
    for (const auto& e : sf::directory_iterator(mDir)){
      s::string name = e.path().filename().string();
      if (name.starts_with("games_") and name.ends_with(".bin")) out.push_back(e.path());
    }
    s::sort(out.begin(), out.end());
    return out;
  }
  static int file_index(const sf::path& p){
    s::string stem = p.stem().string();
    return s::stoi(stem.substr(6));
  }

  template <typename T>
  void put(T v){
    mOut.write(reinterpret_cast<const char*>(&v), sizeof(T));
  }
  template <typename T>
  void put_vec(const s::vector<T>& v){
    mOut.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
  }
  template <typename T>
  static T get(s::ifstream& in){
    T v{};
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
  }
  template <typename T>
  static void get_vec(s::ifstream& in, s::vector<T>& v, size_t n){
    v.resize(n);
    in.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
  }

  static s::vector<s::shared_ptr<Trajectory>> read_file(const sf::path& path){
    s::vector<s::shared_ptr<Trajectory>> out;
    s::ifstream in(path, s::ios::binary);
    while (in.peek() != EOF){
      uint32_t magic = get<uint32_t>(in);
      uint32_t version = get<uint32_t>(in);
      if (magic != MAGIC or version < 1 or version > VERSION)
        throw s::runtime_error("corrupt experience file " + path.string());
      auto t = s::make_shared<Trajectory>();
      t->length = get<int32_t>(in);
      t->observation_size = get<int32_t>(in);
      t->num_tokens = get<int32_t>(in);
      t->num_actions = get<int32_t>(in);
      t->outcome = get<int8_t>(in);
      t->model_version = get<uint32_t>(in);
      size_t n = t->length;
      get_vec(in, t->features, (n + 1) * t->observation_size);
      get_vec(in, t->tokens, (n + 1) * t->num_tokens);
      get_vec(in, t->to_play, n + 1);
      get_vec(in, t->actions, n);
      get_vec(in, t->rewards, n);
      get_vec(in, t->root_values, n);
      get_vec(in, t->policies, n * t->num_actions);
      get_vec(in, t->policy_valid, n);
      get_vec(in, t->final_tokens, t->num_tokens);
      if (version >= 2) get_vec(in, t->value_versions, n);
      else              t->value_versions.assign(n, Trajectory::NO_VALUE);
      // a run killed mid-write leaves a truncated last record
      if (not in) break;
      out.push_back(s::move(t));
    }
    return out;
  }

  sf::path       mDir;
  int            mGamesPerFile;
  int            mNextFile = 0;
  int            mGamesInFile = 0;
  s::ofstream    mOut;
  s::mutex       mMutex;
};

}  // namespace rlbg

#endif//RLBG_EXPERIENCE_BINARY_EXPERIENCE_STORE_H
