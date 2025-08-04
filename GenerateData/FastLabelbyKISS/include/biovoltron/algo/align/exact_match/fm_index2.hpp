#include <spdlog/spdlog.h>
#include <iostream>
#include <biovoltron/algo/sort/sorter.hpp>
#include <biovoltron/algo/sort/kiss1_sorter.hpp>
#include <biovoltron/container/xbit_vector.hpp>
#include <biovoltron/utility/archive/serializer.hpp>
#include <biovoltron/utility/istring.hpp>
#include <chrono>
#include <execution>
#include <queue>
#include <span>
#include <thread>
#include <utility>
#include <unordered_set>

namespace biovoltron {
using namespace std::chrono;


template<int SA_INTV = 1, typename size_type = std::uint32_t,
         SASorter Sorter = KISS1Sorter<size_type>>
class FMIndex {
 public:
  /**
   * Occ sampling interval, default value is 16.
   */
  const int OCC_INTV = 16;
  std::unordered_set<int> end_positions;
  /**
   * The length of fixed suffix for lookup, default value is 14.
   */
  const int LOOKUP_LEN = 0;

  const int OCC1_INTV = 256;
  const int OCC2_INTV = OCC_INTV;

  using char_type = std::int8_t;

  /**
   * A compression vector which store the bwt.
   */
  DibitVector<std::uint8_t> bwt_;

  /**
   * A hierarchical sampled occurrence table.
   */
  std::pair<std::vector<std::array<size_type, 4>>,
            std::vector<std::array<std::uint8_t, 4>>>
    occ_;

  /**
   * A sampled suffix array.
   */
  Sorter::SA_t sa_;

  /**
   * A bit vector recording sampled suffix array.
   */
  detail::XbitVector<1, std::uint64_t, std::allocator<std::uint64_t>> b_;

  /**
   * prefix sum of b_ sampling interval, default value is 64.
   */
  const int B_OCC_INTV = 64;

  /**
   * A sampled prefix sum for b_.
   */
  std::vector<size_type> b_occ_;

  /**
   * A lookup table for fixed suffix query.
   */
  std::array<size_type, 4> cnt_{};
  size_type pri_{};
  std::vector<size_type> lookup_;
  mutable int seed_len = 5;

  FMIndex(std::unordered_set<int> end_positions_init = {})
      : end_positions(std::move(end_positions_init)) {
    // 其他初始化操作可以在这里完成
    size_t new_size = end_positions.size() / 0.5;
    end_positions.reserve(new_size);
    // for (const auto& end_position : end_positions) {
    //   std::cout<<end_position<<" ";
    // }

  }

 public:
  constexpr static auto cnt_table = [] {
    std::array<std::array<std::uint8_t, 4>, 256> cnt_table{};
    for (auto i = size_type{}; i < cnt_table.size(); i++)
      for (auto shift = size_type{}; shift < 8; shift += 2)
        cnt_table[i][i >> shift & 3u]++;
    return cnt_table;
  }();

  auto
  create_pseudogene(const std::vector<std::string>& sequences, const std::string& output_file) {
    std::string pseudogene;  // 用于存储最终的伪基因序列
    
    // 遍历输入的字符串数组
    for (const auto& seq : sequences) {
        pseudogene += seq;  // 拼接字符串
        end_positions.insert(pseudogene.size()-1);
    }
    
    // 将伪基因序列写入文件
    std::ofstream file(output_file);
    if (file.is_open()) {
        file << ">seq\n";         // 写入第一行
        file << pseudogene << "\n"; // 写入第二行
        file.close();
    } else {
        std::cerr << "Error opening file: " << output_file << std::endl;
    }
  }

  auto
  compute_occ(char_type c, size_type i) const {
    const auto occ1_beg = i / OCC1_INTV;
    const auto occ2_beg = i / OCC2_INTV;
    auto beg = occ2_beg * OCC2_INTV;
    auto cnt = size_type{};
    const auto pass_pri = c == 0 && beg <= pri_ && pri_ < i;
    const auto run = (i - beg) / 4;
    for (auto j = size_type{}; j < run; j++) {
      cnt += cnt_table[bwt_.data()[beg / 4]][c];
      beg += 4;
    }
    for (; beg < i; beg++)
      if (bwt_[beg] == c)
        cnt++;
    return occ_.first[occ1_beg][c] + occ_.second[occ2_beg][c] + cnt - pass_pri;
  }

  auto
  lf(char_type c, size_type i) const {
    return cnt_[c] + compute_occ(c, i);
  };

  auto
  compute_b_occ(size_type i) const {
    if constexpr (SA_INTV == 1)
      return i;
    else {
      const auto b_occ_beg = i / B_OCC_INTV;
      auto beg = b_occ_beg * B_OCC_INTV;
      auto cnt = size_type{};

      const auto run = (i - beg) / 64;
      for (auto j = size_type{}; j < run; j++) {
        cnt += std::popcount(b_.data()[beg / 64]);
        beg += 64;
      }

      auto mask = (1ull << i - beg) - 1;
      cnt += std::popcount(b_.data()[beg / 64] & mask);
      return b_occ_[b_occ_beg] + cnt;
    }
  }

  auto
  compute_sa(size_type i) const {
    if constexpr (SA_INTV == 1)
      return sa_[i];
    else {
      auto cnt = size_type{};
      while (not b_[i]) {
        i = lf(bwt_[i], i);
        cnt++;
      }
      return (sa_[compute_b_occ(i)] >> 2) * SA_INTV + cnt;
    }
  }

  auto
  compute_range(istring_view seed, size_type beg, size_type end,
                size_type stop_upper) const {
    while (!seed.empty()) {
      if (end - beg < stop_upper)
        break;
      beg = lf(seed.back(), beg);
      end = lf(seed.back(), end);
      seed.remove_suffix(1);
    }
    return std::array{beg, end, static_cast<size_type>(seed.size())};
  }

  auto
  get_offsets_traditional(size_type beg, size_type end) const {
    if constexpr (SA_INTV == 1)
      return std::span{&sa_[beg], end - beg};
    else {
      auto offsets = std::vector<size_type>{};
      offsets.reserve(end - beg);
      for (auto i = beg; i < end; i++) {
        offsets.push_back(compute_sa(i));
      }
      return offsets;
    }
  }

  /**
   * Compute the suffix array value according to the begin and end.
   * If sa_intv is 1, this can be done at O(1).
   */
  auto
  get_offsets(size_type beg, size_type end) const {
    // std::cout << "Input beg: " << beg << ", end: " << end << std::endl;
    if constexpr (SA_INTV == 1)
      return std::span{&sa_[beg], end - beg};
    else {
      auto offsets = std::vector<size_type>{};
      offsets.reserve(end - beg);
      auto new_offsets = std::vector<size_type>{};
      new_offsets.reserve(end - beg);
      constexpr auto MAX_DEPTH = SA_INTV;

      auto q = std::queue<std::tuple<size_type, size_type, int>>{};
      q.emplace(beg, end, 0);

      auto enqueue = [&q](auto beg, auto end, auto dep) {
        if (beg != end)
          q.emplace(beg, end, dep);
      };
      auto del_pos = 0;
      while (q.size() and offsets.size() < end - beg) {
        auto [cur_beg, cur_end, cur_dep] = q.front();
        q.pop();

        // add offset from sampled sa value
        const auto b_occ_cur_beg = compute_b_occ(cur_beg);
        const auto b_occ_cur_end = compute_b_occ(cur_end);
        for (auto i = b_occ_cur_beg; i < b_occ_cur_end; i++) {
          size_type candidate_offset = sa_[i] + cur_dep;
          // 检查位置是否符合条件
          auto is_valid_position = 0;

          for (size_t j = 0; j < seed_len-1; ++j) {
          //   // 如果当前的pos加上i在end_positions中存在，则不加入offsets

            if (end_positions.find(candidate_offset + j) != end_positions.end()) {
              is_valid_position = 1;
              break;
            }
          }
          if(is_valid_position == 1){
            continue;
          }
          offsets.push_back(candidate_offset);
        }

        const auto nxt_dep = cur_dep + 1;
        if (nxt_dep == MAX_DEPTH)
          continue;

        if (cur_beg + 1 == cur_end) {
          const auto nxt_beg = lf(bwt_[cur_beg], cur_beg);
          const auto nxt_end = nxt_beg + 1;
          q.emplace(nxt_beg, nxt_end, nxt_dep);
        } else {
          [&]<auto... Idx>(std::index_sequence<Idx...>) {
            auto bg = std::array{lf(Idx, cur_beg)...};
            auto ed = std::array{lf(Idx, cur_end)...};
            (enqueue(bg[Idx], ed[Idx], nxt_dep), ...);
          }
          (std::make_index_sequence<4>{});
        }
      }
      return offsets;
    }
  }

  size_type is_valid_position1(size_type pos) const {
    for (size_t i = 0; i < seed_len; ++i) {
      auto pp = end_positions.find(pos + i);
      auto pd = end_positions.end();
        if (pp != pd) {
          // if (pos % 2 == 0){
            // return 0;
            return 0;
        }
    }
    return pos;
  }


  auto
  fmtree(istring_view seed) {
    auto [pre_beg, pre_end, pre_offs] = get_range(seed.substr(1), 0);
    auto [beg, end, offs] = get_range(seed.substr(0, 1), pre_beg, pre_end, 0);

    auto offsets = std::vector<size_type>{};
    offsets.reserve(end - beg);

    constexpr auto MAX_DEPTH = SA_INTV;

    auto q = std::queue<std::tuple<size_type, size_type, int>>{};
    q.emplace(beg, end, 0);

    auto enqueue = [&q](auto beg, auto end, auto dep) {
      if (beg != end)
        q.emplace(beg, end, dep);
    };

    while (q.size() and offsets.size() < end - beg) {
      auto [cur_beg, cur_end, cur_dep] = q.front();
      q.pop();

      // add offset from sampled sa value
      const auto b_occ_cur_beg = compute_b_occ(cur_beg);
      const auto b_occ_cur_end = compute_b_occ(cur_end);
      for (auto i = b_occ_cur_beg; i < b_occ_cur_end; i++) {
        offsets.push_back(sa_[i] + cur_dep);
      }

      const auto nxt_dep = cur_dep + 1;
      if (nxt_dep == MAX_DEPTH)
        continue;

      if (cur_beg + 1 == cur_end) {
        const auto nxt_beg = lf(bwt_[cur_beg], cur_beg);
        const auto nxt_end = nxt_beg + 1;
        q.emplace(nxt_beg, nxt_end, nxt_dep);
      } else {
        [&]<auto... Idx>(std::index_sequence<Idx...>) {
          auto bg = std::array{lf(Idx, cur_beg)...};
          auto ed = std::array{lf(Idx, cur_end)...};
          (enqueue(bg[Idx], ed[Idx], nxt_dep), ...);
        }
        (std::make_index_sequence<4>{});
      }
    }

    return offsets;
  }

  auto
  get_range1(istring_view seed, size_type beg, size_type end,
            size_type stop_cnt = 0) const {
    if (end == beg || seed.empty())
      return std::array{beg, end, size_type{}};
    return compute_range(seed, beg, end, stop_cnt + 1);
  }

  /**
   * Get begin and end index of the uncompressed suffix array for the input
   * seed. Difference between begin and end is the occurrence count in
   * reference.
   *
   * @param seed Seed to search.
   * @param stop_cnt The remaining prefix length for the seed. Notice that when
   * the stop_cnt is -1, this value always be 0. The stop_cnt can be using to
   * early stop when occurrence count is not greater than the value.
   * Set to 0 to forbid early stop.
   * @return An array of begin, end index and the match stop position.
   */
  auto
  get_range(istring_view seed, size_type stop_cnt = 0) const {
    auto beg = size_type{};
    auto end = bwt_.size();
    seed_len = seed.size();
    if (seed.size() >= LOOKUP_LEN) {
      // std::cout <<LOOKUP_LEN<< std::endl;
      const auto key = Codec::hash(seed.substr(seed.size() - LOOKUP_LEN));
      beg = lookup_[key];
      end = lookup_[key + 1];
      seed.remove_suffix(LOOKUP_LEN);
    }
    return get_range1(seed, beg, end, stop_cnt);
  }


  /**
   * Load index, utility for serialization.
   */
  auto
  load(std::ifstream& fin) {
    const auto start = high_resolution_clock::now();

    // 打印文件流的初始状态和文件位置
    // std::cout << "Initial fin state: " << fin.good() << ", position: " << fin.tellg() << std::endl;
    // SPDLOG_DEBUG("Initial fin state: {}, position: {}", fin.good(), fin.tellg());

    fin.read(reinterpret_cast<char*>(&cnt_), sizeof(cnt_));
    fin.read(reinterpret_cast<char*>(&pri_), sizeof(pri_));
    SPDLOG_DEBUG("load bwt...");
    Serializer::load(fin, bwt_);
    
    // 打印文件流在加载 bwt_ 之后的状态和文件位置
    // std::cout << "After loading bwt_, fin state: " << fin.good() << ", position: " << fin.tellg() << std::endl;
    // SPDLOG_DEBUG("After loading bwt_, fin state: {}, position: {}", fin.good(), fin.tellg());

    auto size = bwt_.size();
    // SPDLOG_DEBUG("load occ...");
    Serializer::load(fin, occ_.first);
    Serializer::load(fin, occ_.second);

    // 打印文件流在加载 occ_ 之后的状态和文件位置
    // std::cout << "After loading occ_, fin state: " << fin.good() << ", position: " << fin.tellg() << std::endl;
    // SPDLOG_DEBUG("After loading occ_, fin state: {}, position: {}", fin.good(), fin.tellg());

    // SPDLOG_DEBUG("load sa...");
    Serializer::load(fin, sa_);
    // std::cout << "After loading sa, fin state: " << fin.good() << ", position: " << fin.tellg() << std::endl;
    // SPDLOG_DEBUG("load lookup...");
    Serializer::load(fin, lookup_);

    // std::cout << "After loading lookup, fin state: " << fin.good() << ", position: " << fin.tellg() << std::endl;

    if constexpr (SA_INTV != 1) {
        // SPDLOG_DEBUG("load b_...");
        Serializer::load(fin, b_);
        // SPDLOG_DEBUG("load b_occ_...");
        Serializer::load(fin, b_occ_);
    }

    // 打印文件流在加载完所有数据后的状态和文件位置
    // std::cout << "Final fin state: " << fin.good() << ", position: " << fin.tellg() << std::endl;
    // SPDLOG_DEBUG("Final fin state: {}, position: {}", fin.good(), fin.tellg());

    assert(fin.peek() == EOF);
    const auto end = high_resolution_clock::now();
    const auto dur = duration_cast<seconds>(end - start);
    // SPDLOG_DEBUG("elapsed time: {} s.", dur.count());
  }

  auto load1(const std::string &file_path) {
        std::ifstream in(file_path);
        if (!in) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }
        load(in);  // 调用已有的 ifstream 版本
  }

  bool
  operator==(const FMIndex& other) const = default;
};

  

}  // namespace biovoltron

