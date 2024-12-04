#include <ctime>
#include <random>
#include <string>
#include <unordered_set>
#include <iostream>
#include <fstream>

#include <type_alias.h>

namespace s = std;

namespace {
s::default_random_engine& get_default_random_engine(){
  static s::default_random_engine engine(time(0));
  return engine;
}
} // namespace

void generate_file_includes(s::fstream& out, size_t player_size, size_t table_size){
  out << "/* This file is auto-generated */\n";
  out << "/* Command Options: player_size=" << player_size << " table_size=" << table_size << " */\n";
  out << "#ifndef RLBG_ZOBRIST_HASH\n";
  out << "#define RLBG_ZOBRIST_HASH\n";
  out << "#include <cassert>\n";
  out << "\n";
  out << "#include <type_alias.h>\n";
  out << "\n";
  out << "namespace rlbg {\n";
  out << "\n";
}

uint get_next_value(s::uniform_int_distribution<uint>& dist, s::unordered_set<uint>& used){
  unsigned value = dist(get_default_random_engine());
  while (used.find(value) != used.end()){
    value = dist(get_default_random_engine());
  }
  return value;
}

void generate_board_hash_table(s::fstream& out, size_t player_size, size_t table_size, s::unordered_set<uint>& used){
  s::uniform_int_distribution<uint> dist(0x10000000U, 0xFFFFFFFFU);

  out << "uint zobrist_hash(uint player, uint p_index){\n";
  out << "  uint key = p_index;\n";
  out << "  key |= player << 30U;\n";
  out << "  switch (key){\n";

  out << s::hex;
  for (uint pi = 1; pi <= player_size; ++pi){
    for (uint i = 0; i < table_size; ++i){
      uint value = pi << 30U;
      value |= i;
      uint hash_value = get_next_value(dist, used);
      used.insert(hash_value);
      out << "    case 0x" << value << "U: return 0x" << hash_value << "U; break;\n";
    }
  }
  out << "    default: assert(false);\n";
  
  out << "  }\n";
  out << "}\n";
}

void generate_player_hash_table(s::fstream& out, size_t player_size, s::unordered_set<uint>& used){
  s::uniform_int_distribution<uint> dist(0x10000000U, 0xFFFFFFFFU);

  out << "uint zobrist_hash(uint player){\n";
  out << "  uint key = player << 30U;\n";
  out << "  switch (key){\n";

  out << s::hex;
  for (uint pi = 1; pi <= player_size; ++pi){
    uint value = pi << 30U;
    uint hash_value = get_next_value(dist, used);
    out << "    case 0x" << value << "U: return 0x" << hash_value << "U; break;\n";
  }
  out << "    default: assert(false);\n";

  out << "  }\n";
  out << "}\n";
}

void generate_file_suffix(s::fstream& out){
  out << "\n";
  out << "constexpr uint EMPTY_BOARD = 0U;\n";
  out << "\n";
  out << "} // rlbg\n";
  out << "#endif//RLBG_ZOBRIST_HASH\n";
}

int main(int argc, char* argv[]){
  if (argc != 4){
    s::cout << "Usage: " << argv[0] << " <filename> <player_size> <index_size>" << s::endl;
    exit(1);
  }

  const char* filename = argv[1];
  s::fstream output(filename, s::ios_base::out);

  int player_size = atoi(argv[2]);
  int table_size = atoi(argv[3]);

  if (player_size > 4){
    s::cout << "cannot handle player size > 4" << s::endl;
    exit(1);
  }

  s::unordered_set<uint> used;

  generate_file_includes(output, player_size, table_size);
  generate_board_hash_table(output, player_size, table_size, used);
  generate_player_hash_table(output, player_size, used);
  generate_file_suffix(output);
  output << s::endl;

  output.close();
}
