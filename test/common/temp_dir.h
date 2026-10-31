#ifndef RLBG_TEST_TEMP_DIR_H
#define RLBG_TEST_TEMP_DIR_H

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

namespace rlbg::test {

namespace s = std;
namespace sf = std::filesystem;

// an empty directory named after the running test, removed when it goes
// out of scope
struct TempDir {
  TempDir(){
    const testing::TestInfo* info = testing::UnitTest::GetInstance()->current_test_info();
    s::string name = s::string("rlbg_") + info->test_suite_name() + "_" + info->name();
    path = sf::temp_directory_path() / name;
    sf::remove_all(path);
    sf::create_directories(path);
  }
  ~TempDir(){
    s::error_code ec;
    sf::remove_all(path, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;

  s::string str() const { return path.string(); }
  sf::path operator/(const s::string& p) const { return path / p; }

  sf::path path;
};

}  // namespace rlbg::test

#endif//RLBG_TEST_TEMP_DIR_H
