/**
 * 3次元Isingシミュレーション
 * 乱数生成: Mersenne Twister for Graphic Processors (MTGP32)
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <sstream>
#include <string>
#include <sycl/sycl.hpp>
#include <sys/random.h>
#include <unistd.h>

using namespace sycl;

// ==========================================
// 1. MTGP32 定数・構造体・パラメータ
// ==========================================
#define MTGPDC_N 351
#define MTGPDC_FLOOR_2P 256
#define MTGP_TN MTGPDC_FLOOR_2P
#define MTGP_LS (MTGP_TN * 3)

struct mtgp32_params_fast_t {
  int mexp;
  int pos;
  int sh1;
  int sh2;
  uint32_t tbl[16];
  uint32_t tmp_tbl[16];
  uint32_t flt_tmp_tbl[16];
  uint32_t mask;
  unsigned char poly_sha1[21];
};

static const mtgp32_params_fast_t mtgp32_params[] =
#include "mtgp32dc_params_fast_11213.h"
    ;

// ==========================================
// 2. MTGP デバイス関数
// ==========================================
inline uint32_t para_rec(uint32_t X1, uint32_t X2, uint32_t Y, int bid,
                         const uint32_t *sh1_tbl, const uint32_t *sh2_tbl,
                         const uint32_t *param_tbl, uint32_t mask) {
  uint32_t X = (X1 & mask) ^ X2;
  uint32_t MAT;
  X ^= X << sh1_tbl[bid];
  Y = X ^ (Y >> sh2_tbl[bid]);
  MAT = param_tbl[bid * 16 + (Y & 0x0f)];
  return Y ^ MAT;
}

inline uint32_t temper(uint32_t V, uint32_t T, int bid,
                       const uint32_t *temper_tbl) {
  uint32_t MAT;
  T ^= T >> 16;
  T ^= T >> 8;
  MAT = temper_tbl[bid * 16 + (T & 0x0f)];
  return V ^ MAT;
}

inline void status_read(sycl::local_accessor<uint32_t, 1> status,
                        const uint32_t *d_status_ptr, int bid, int tid,
                        sycl::nd_item<1> &item) {
  const uint32_t *block_status = &d_status_ptr[bid * MTGPDC_N];
  status[MTGP_LS - MTGPDC_N + tid] = block_status[tid];
  if (tid < MTGPDC_N - MTGP_TN) {
    status[MTGP_LS - MTGPDC_N + MTGP_TN + tid] = block_status[MTGP_TN + tid];
  }
  item.barrier(sycl::access::fence_space::local_space);
}

inline void status_write(uint32_t *d_status_ptr,
                         sycl::local_accessor<uint32_t, 1> status, int bid,
                         int tid, sycl::nd_item<1> &item) {
  uint32_t *block_status = &d_status_ptr[bid * MTGPDC_N];
  block_status[tid] = status[MTGP_LS - MTGPDC_N + tid];
  if (tid < MTGPDC_N - MTGP_TN) {
    block_status[MTGP_TN + tid] = status[4 * MTGP_TN - MTGPDC_N + tid];
  }
  item.barrier(sycl::access::fence_space::local_space);
}

// ==========================================
// 3. MTGP ホスト関数 & エンジンクラス
// ==========================================
void mtgp32_init_state(uint32_t *array, const mtgp32_params_fast_t *para,
                       uint32_t seed) {
  int size = para->mexp / 32 + 1;
  uint32_t hidden_seed;
  uint32_t tmp;
  hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
  tmp = hidden_seed;
  tmp += tmp >> 16;
  tmp += tmp >> 8;
  std::memset(array, tmp & 0xff, sizeof(uint32_t) * size);
  array[0] = seed;
  array[1] = hidden_seed;
  for (int i = 1; i < size; i++) {
    array[i] ^= 1812433253U * (array[i - 1] ^ (array[i - 1] >> 30)) + i;
  }
}

class MtgpEngine {
public:
  const int num_blocks = 64;
  uint32_t mask;

  // デバイスメモリ
  uint32_t *d_pos, *d_sh1, *d_sh2, *d_param, *d_temper, *d_status;
  uint32_t *d_rand_pool = nullptr; // 生成された乱数を格納するプール

  queue &q;
  sycl::event last_event; // 最後の乱数生成イベントを記録

  MtgpEngine(queue &sycl_queue, int max_count, int base_seed = 1234)
      : q(sycl_queue) {
    // メモリ確保
    d_pos = malloc_device<uint32_t>(num_blocks, q);
    d_sh1 = malloc_device<uint32_t>(num_blocks, q);
    d_sh2 = malloc_device<uint32_t>(num_blocks, q);
    d_param = malloc_device<uint32_t>(num_blocks * 16, q);
    d_temper = malloc_device<uint32_t>(num_blocks * 16, q);
    d_status = malloc_device<uint32_t>(num_blocks * MTGPDC_N, q);

    if (max_count) {
      realloc(max_count);
    }

    initHostData(base_seed);
  }

  ~MtgpEngine() {
    free(d_pos, q);
    free(d_sh1, q);
    free(d_sh2, q);
    free(d_param, q);
    free(d_temper, q);
    free(d_status, q);
    free(d_rand_pool, q);
  }

  void realloc(int max_count) {
    q.wait();

    if (d_rand_pool) {
      free(d_rand_pool, q);
    }

    int min_per_block = (max_count + num_blocks - 1) / num_blocks;
    int loop_steps = (min_per_block + MTGP_LS - 1) / MTGP_LS;
    int total_data =
        loop_steps * MTGP_LS * num_blocks; // アラインメント済みサイズ
    d_rand_pool = malloc_device<uint32_t>(total_data, q);
  }

  void initHostData(int base_seed) {
    std::vector<uint32_t> h_pos(num_blocks);
    std::vector<uint32_t> h_sh1(num_blocks);
    std::vector<uint32_t> h_sh2(num_blocks);
    std::vector<uint32_t> h_param(num_blocks * 16);
    std::vector<uint32_t> h_temper(num_blocks * 16);
    std::vector<uint32_t> h_status(num_blocks * MTGPDC_N);

    int param_count = sizeof(mtgp32_params) / sizeof(mtgp32_params[0]);
    mask = mtgp32_params[0].mask;

    for (int i = 0; i < num_blocks; i++) {
      // パラメータを循環利用
      const auto *para = &mtgp32_params[i % param_count];
      h_pos[i] = para->pos;
      h_sh1[i] = para->sh1;
      h_sh2[i] = para->sh2;
      for (int j = 0; j < 16; j++) {
        h_param[i * 16 + j] = para->tbl[j];
        h_temper[i * 16 + j] = para->tmp_tbl[j];
      }
      // シードを変えてステータス初期化
      mtgp32_init_state(&h_status[i * MTGPDC_N], para, base_seed + i);
    }

    q.copy(h_pos.data(), d_pos, num_blocks);
    q.copy(h_sh1.data(), d_sh1, num_blocks);
    q.copy(h_sh2.data(), d_sh2, num_blocks);
    q.copy(h_param.data(), d_param, num_blocks * 16);
    q.copy(h_temper.data(), d_temper, num_blocks * 16);
    q.copy(h_status.data(), d_status, num_blocks * MTGPDC_N);
    q.wait();
  }

  // 乱数生成カーネル
  sycl::event generate(int request_count) {
    // 各ブロックが処理すべきデータ数を計算
    // 最低でも MTGP_LS (768個) 単位で処理しないと状態遷移が狂うためアラインメントする
    int min_per_block = (request_count + num_blocks - 1) / num_blocks;
    int loop_steps = (min_per_block + MTGP_LS - 1) / MTGP_LS;
    int data_per_block = loop_steps * MTGP_LS;

    auto p_pos = d_pos;
    auto p_sh1 = d_sh1;
    auto p_sh2 = d_sh2;
    auto p_param = d_param;
    auto p_temper = d_temper;
    auto p_status = d_status;
    auto p_out = d_rand_pool;
    uint32_t m = mask;
    int max_idx = request_count; // 書き込み上限

    last_event = q.submit([&](handler &cgh) {
      // 前回の乱数生成が終わるのを待つ
      cgh.depends_on(last_event);

      sycl::local_accessor<uint32_t, 1> local_status(sycl::range<1>(MTGP_LS), cgh);

      cgh.parallel_for(
          nd_range<1>(range<1>(num_blocks * MTGP_TN), range<1>(MTGP_TN)),
          [=](nd_item<1> item) {
            int bid = item.get_group(0);
            int tid = item.get_local_id(0);
            int pos = p_pos[bid];
            uint32_t r, o;

            status_read(local_status, p_status, bid, tid, item);

            // data_per_block 分だけループ (不必要な巨大ループを回避)
            for (int i = 0; i < data_per_block; i += MTGP_LS) {
              // Step 1
              r = para_rec(local_status[MTGP_LS - MTGPDC_N + tid],
                           local_status[MTGP_LS - MTGPDC_N + tid + 1],
                           local_status[MTGP_LS - MTGPDC_N + tid + pos], bid,
                           p_sh1, p_sh2, p_param, m);
              local_status[tid] = r;
              o = temper(r, local_status[MTGP_LS - MTGPDC_N + tid + pos - 1],
                         bid, p_temper);

              // 必要な個数を超えたら書き込まない
              if (data_per_block * bid + i + tid < max_idx)
                p_out[data_per_block * bid + i + tid] = o;

              item.barrier(sycl::access::fence_space::local_space);

              // Step 2
              r = para_rec(
                  local_status[(4 * MTGP_TN - MTGPDC_N + tid) % MTGP_LS],
                  local_status[(4 * MTGP_TN - MTGPDC_N + tid + 1) % MTGP_LS],
                  local_status[(4 * MTGP_TN - MTGPDC_N + tid + pos) % MTGP_LS],
                  bid, p_sh1, p_sh2, p_param, m);
              local_status[tid + MTGP_TN] = r;
              o = temper(r,
                         local_status[(4 * MTGP_TN - MTGPDC_N + tid + pos - 1) %
                                      MTGP_LS],
                         bid, p_temper);

              if (data_per_block * bid + MTGP_TN + i + tid < max_idx)
                p_out[data_per_block * bid + MTGP_TN + i + tid] = o;

              item.barrier(sycl::access::fence_space::local_space);

              // Step 3
              r = para_rec(local_status[2 * MTGP_TN - MTGPDC_N + tid],
                           local_status[2 * MTGP_TN - MTGPDC_N + tid + 1],
                           local_status[2 * MTGP_TN - MTGPDC_N + tid + pos],
                           bid, p_sh1, p_sh2, p_param, m);
              local_status[tid + 2 * MTGP_TN] = r;
              o = temper(r,
                         local_status[tid + pos - 1 + 2 * MTGP_TN - MTGPDC_N],
                         bid, p_temper);

              if (data_per_block * bid + 2 * MTGP_TN + i + tid < max_idx)
                p_out[data_per_block * bid + 2 * MTGP_TN + i + tid] = o;

              item.barrier(sycl::access::fence_space::local_space);
            }
            status_write(p_status, local_status, bid, tid, item);
          });
    });

    return last_event;
  }
};

// ==========================================
// 4. シミュレーション本編
// ==========================================

// ビット展開ヘルパー
inline int64_t spread_bits(uint32_t x) {
  int64_t n = static_cast<int64_t>(x);
  n = (n | (n << 16)) & 0x0000FFFF0000FFFFL;
  n = (n | (n << 8)) & 0x00FF00FF00FF00FFL;
  n = (n | (n << 4)) & 0x0F0F0F0F0F0F0F0FL;
  n = (n | (n << 2)) & 0x3333333333333333L;
  n = (n | (n << 1)) & 0x5555555555555555L;
  return n;
}

bool no_calc_en = false;

class SpinGlassSimulator {
public:
  int L, N;
  queue q;
  uint32_t counter = 0;
  static constexpr int NRMAX = 1024 * 1024 * 32;
  int MULTI;

  bool made_kexp = false;
  static constexpr sycl::specialization_id<bool> spec_made_kexp;
  bool disable_kexp = false;

  bool same_bond = false;

  // デバイスメモリ
  std::vector<int32_t *> d_spins, d_kexp, d_bonds;
  std::vector<uint32_t *> d_thresholds;
  std::vector<int64_t *> d_sum_overlap, d_sum_energy, d_sum_mag, d_sum_mag_stag;
  int32_t *d_shuffle;

  // ホストメモリ
  std::vector<double> temperatures;
  std::vector<uint32_t> h_thr;

  // MTGP エンジン
  MtgpEngine *mtgp;

  // 各レプリカ i が、次に待つべきイベントのリスト
  std::vector<std::vector<sycl::event>> replica_deps;

  uint64_t last_log_time = 0;

  SpinGlassSimulator(int latticeSize, queue &sycl_queue, int MULTI)
      : q(sycl_queue), MULTI(MULTI) {

    int base_seed;
    ssize_t result = getrandom(&base_seed, sizeof(base_seed), 0);
    mtgp = new MtgpEngine(q, 0, base_seed);

    d_spins.resize(MULTI, nullptr);
#ifdef SIMULATION_MODE_SG
    d_bonds.resize(MULTI, nullptr);
    d_sum_overlap.resize(MULTI);
    for (int i = 0; i < MULTI; ++i)
      d_sum_overlap[i] = malloc_device<int64_t>(16, q);
#elif defined(SIMULATION_MODE_FM)
    d_sum_mag.resize(MULTI);
    for (int i = 0; i < MULTI; ++i)
      d_sum_mag[i] = malloc_device<int64_t>(32, q);
#elif defined(SIMULATION_MODE_AFM)
    d_sum_mag_stag.resize(MULTI);
    for (int i = 0; i < MULTI; ++i)
      d_sum_mag_stag[i] = malloc_device<int64_t>(32, q);
#endif
    d_kexp.resize(MULTI);
    for (int i = 0; i < MULTI; ++i)
      d_kexp[i] = malloc_device<int32_t>(NRMAX * 3, q);
    d_shuffle = malloc_device<int32_t>(NRMAX, q);
    d_sum_energy.resize(MULTI);
    for (int i = 0; i < MULTI; ++i)
      d_sum_energy[i] = malloc_device<int64_t>(32, q);
    d_thresholds.resize(MULTI);
    for (int i = 0; i < MULTI; ++i)
      d_thresholds[i] = malloc_device<uint32_t>(32 * 3, q);

    temperatures.resize(32 * MULTI, 1.0);
    h_thr.resize(32 * 3 * MULTI);

    replica_deps.resize(MULTI);

    realloc(latticeSize);

#ifdef SIMULATION_MODE_SG
    resetBonds(false);
#endif
    resetSpins();
  }

  ~SpinGlassSimulator() {
    for (int i = 0; i < MULTI; ++i)
      free(d_spins[i], q);
#ifdef SIMULATION_MODE_SG
    for (int i = 0; i < MULTI; ++i)
      free(d_bonds[i], q);
    for (int i = 0; i < MULTI; ++i)
      free(d_sum_overlap[i], q);
#elif defined(SIMULATION_MODE_FM)
    for (int i = 0; i < MULTI; ++i)
      free(d_sum_mag[i], q);
#elif defined(SIMULATION_MODE_AFM)
    for (int i = 0; i < MULTI; ++i)
      free(d_sum_mag_stag[i], q);
#endif
    for (int i = 0; i < MULTI; ++i)
      free(d_kexp[i], q);
    free(d_shuffle, q);
    for (int i = 0; i < MULTI; ++i)
      free(d_sum_energy[i], q);
    for (int i = 0; i < MULTI; ++i)
      free(d_thresholds[i], q);
    delete mtgp;
  }

  void realloc(int latticeSize) {
    q.wait();

    L = latticeSize;
    N = L * L * L;

    int max_count = std::max({
        NRMAX,
        N * MULTI * 32,
#ifdef SIMULATION_MODE_SG
        N * 3,
#endif
    });
    mtgp->realloc(max_count);

    for (int i = 0; i < MULTI; ++i)
      if (d_spins[i])
        free(d_spins[i], q);
    for (int i = 0; i < MULTI; ++i)
      d_spins[i] = malloc_device<int32_t>(L * L * L, q);
#ifdef SIMULATION_MODE_SG
    for (int i = 0; i < MULTI; ++i)
      if (d_bonds[i])
        free(d_bonds[i], q);
    for (int i = 0; i < MULTI; ++i)
      d_bonds[i] = malloc_device<int32_t>(L * L * L * 3, q);
#endif
  }

#ifdef SIMULATION_MODE_SG
  // ボンド初期化カーネル
  void resetBonds(bool same) {
    if (!same) {
      for (int i = 0; i < MULTI; ++i) {
        // 1. 乱数を生成
        sycl::event rng_evt = mtgp->generate(N * 3);

        auto b_ptr = d_bonds[i];
        auto rnd_ptr = mtgp->d_rand_pool;
        int local_L = L;
        int local_N = N;

        // 2. バルク部分を乱数で埋める (-1 or 0)
        sycl::event evt = q.submit([&](handler &h) {
          h.depends_on(replica_deps[i]);
          h.depends_on(rng_evt);

          h.parallel_for(range<1>(N * 3), [=](id<1> idx) {
            int i = idx[0];

            // MTGPの出力はuint32なので、最下位ビット等で判定
            uint32_t r = rnd_ptr[i];

            // 奇数ビットを隣の偶数ビットの値で上書き
            uint32_t mask_odd = 0x55555555;
            uint32_t even_vals = r & mask_odd;
            r = even_vals | (even_vals << 1);
            b_ptr[i] = r;
          });
        });

        replica_deps[i] = {evt};

        mtgp->last_event = evt;
      }
    } else {
      // 1. 乱数を生成
      sycl::event rng_evt = mtgp->generate(N * 3);

      for (int i = 0; i < MULTI; ++i) {
        auto b_ptr = d_bonds[i];
        auto rnd_ptr = mtgp->d_rand_pool;
        int local_L = L;
        int local_N = N;

        // 2. バルク部分を乱数で埋める (-1 or 0)
        q.wait();
        sycl::event evt = q.submit([&](handler &h) {
          h.depends_on(replica_deps[i]);
          h.depends_on(rng_evt);

          h.parallel_for(range<1>(N * 3), [=](id<1> idx) {
            int i = idx[0];
            uint32_t r = rnd_ptr[i];
            b_ptr[i] = (r & 1) ? 0xFFFFFFFF : 0;
          });
        });

        replica_deps[i] = {evt};
      }
    }

    q.wait();
    same_bond = same;
  }
#endif

  // スピン初期化 (0クリア)
  void resetSpins() {
    for (int i = 0; i < MULTI; ++i) {
      sycl::event evt =
          q.memset(d_spins[i], 0, L * L * L * sizeof(int32_t), replica_deps[i]);
      replica_deps[i] = {evt};
    }

    q.wait();
  }

  // スピンランダム化カーネル
  void randomizeSpins() {
    sycl::event rng_evt = mtgp->generate(N * MULTI);

    for (int i = 0; i < MULTI; ++i) {
      auto rnd_ptr = mtgp->d_rand_pool + N * i;
      auto s_ptr = d_spins[i];

      sycl::event evt =
          q.copy(reinterpret_cast<int32_t *>(rnd_ptr), s_ptr, N, rng_evt);

      replica_deps[i] = {evt};
    }

    q.wait();
  }

  void setTemperatures(const std::vector<double> &temps) {
    temperatures = temps;
    made_kexp = false;
    for (int i = 0; i < MULTI; ++i) {
      for (int j = 0; j < 32; ++j) {
        double beta = 1.0 / temperatures[i * 32 + j];
        auto to_uint32 = [](double p) {
          return std::min(UINT32_MAX, static_cast<uint32_t>(p * UINT32_MAX));
        };
        h_thr[i * 32 * 3 + j * 3 + 0] = to_uint32(std::exp(-8.0 * beta));
        h_thr[i * 32 * 3 + j * 3 + 1] = to_uint32(std::exp(-4.0 * beta));
        h_thr[i * 32 * 3 + j * 3 + 2] = to_uint32(std::exp(-12.0 * beta));
      }

      auto t_ptr = d_thresholds[i];
      q.copy(h_thr.data() + i * 32 * 3, t_ptr, 32 * 3, replica_deps[i]);
    }
    q.wait();
  }

  void updateKexp(void) {
    if (disable_kexp || made_kexp) {
      return;
    }
    for (int i = 0; i < MULTI; ++i) {
      auto k_ptr = d_kexp[i];
      auto s_ptr = d_shuffle;
      auto t_ptr = d_thresholds[i];

      q.memset(k_ptr, 0, NRMAX * 3 * sizeof(int32_t), replica_deps[i]);
      q.submit([&](handler &h) {
        h.depends_on(replica_deps[i]);

        h.parallel_for(range<1>(NRMAX), [=](id<1> r_id) {
          int r = r_id[0];
          s_ptr[r] = r;
        });
      });
      q.wait();

      for (int j = 0; j < 32; ++j) {
        // 1. 乱数を更新
        mtgp->generate(NRMAX).wait();
        auto rnd_ptr = mtgp->d_rand_pool;

        int needed_count = h_thr[i * 32 * 3 + j * 3 + 1];
        needed_count = (static_cast<int64_t>(needed_count) * NRMAX) >> 32;

        // 2. oneDPLでソート (USMポインタを直接渡す)
        auto policy = oneapi::dpl::execution::make_device_policy(q);

        // キーとデータを「ジッパー」のように結合するイテレータ
        // 比較時は「第一要素（キー）」が優先される仕様を利用
        auto zipped_begin = oneapi::dpl::make_zip_iterator(rnd_ptr, s_ptr);
        auto zipped_end =
            oneapi::dpl::make_zip_iterator(rnd_ptr + NRMAX, s_ptr + NRMAX);

        // partial_sort(begin, middle, end)
        oneapi::dpl::partial_sort(
            policy,
            zipped_begin,                // 先頭
            zipped_begin + needed_count, // ここまでがソートされる (M個)
            zipped_end                   // 末尾
        );

        q.submit([&](handler &h) {
           h.parallel_for(range<1>(needed_count), [=](id<1> r_id) {
             int r = r_id[0];
             int index = s_ptr[r];
             k_ptr[index * 3 + 1] |= (1u << j);
             r = (static_cast<int64_t>(r) << 32) / NRMAX;
             if (r < t_ptr[j * 3 + 0])
               k_ptr[index * 3 + 0] |= (1u << j);
             if (r < t_ptr[j * 3 + 2])
               k_ptr[index * 3 + 2] |= (1u << j);
           });
         }).wait(); // 安全のためここで完了を待機
      }
    }

    q.wait();
    made_kexp = true;
  }

  void step() {
    // 1ステップ分の乱数を一括生成
    // step内でCheckerboard更新(flag=0, flag=1)を行うため、全サイト分の乱数が必要
    sycl::event rng_evt;
    if (made_kexp) {
#ifdef SIMULATION_MODE_SG
      rng_evt = mtgp->generate(N * MULTI * 2);
#else
      rng_evt = mtgp->generate(N * MULTI);
#endif
    } else {
      rng_evt = mtgp->generate(N * MULTI * 32);
    }

    for (int i = 0; i < MULTI; ++i) {
      auto s_ptr = d_spins[i];
#ifdef SIMULATION_MODE_SG
      auto b_ptr = d_bonds[i];
#endif
      auto k_ptr = d_kexp[i];
#ifdef SIMULATION_MODE_SG
      auto rnd_ptr = mtgp->d_rand_pool + N * 2 * i;
#else
      auto rnd_ptr = mtgp->d_rand_pool + N * i; // 生成済み乱数プール
#endif
      if (!made_kexp) {
        rnd_ptr = mtgp->d_rand_pool + N * 32 * i;
      }
      auto t_ptr = d_thresholds[i];

      const int local_L = L;
      const int L2 = L * L;

      sycl::event step_evt;

      // Metropolis 更新
      for (int flag = 0; flag <= 1; ++flag) {
        const int half_N = N / 2 + ((N & 1) & flag);
        const size_t local_size = 256;
        const size_t global_size =
            ((half_N + local_size - 1) / local_size) * local_size;

        if (global_size == 0)
          continue; // special case: L=1

        step_evt = q.submit([&](handler &h) {
          h.set_specialization_constant<spec_made_kexp>(made_kexp);

          // 乱数生成およびこのレプリカの前回の処理が終わるのを待つ
          h.depends_on({rng_evt, step_evt});

          h.parallel_for(
              nd_range<1>(range<1>(global_size), range<1>(local_size)),
              [=](nd_item<1> item,
                  kernel_handler kh) [[sycl::reqd_sub_group_size(32)]] {
                bool use_kexp =
                    kh.get_specialization_constant<spec_made_kexp>();

                int idx = item.get_global_id(0);
                if (idx >= half_N)
                  return;

                // インデックス計算
                int index = idx * 2;
                int zk = index / L2;
                int yj = (index % L2) / local_L;
                int xi = index % local_L;
                if (local_L % 2 == 0 && (xi + yj + zk) % 2 != flag) {
                  index++;
                  xi++;
                }

                auto get_spin = [&](int i, int j, int k) {
                  return s_ptr[((k + local_L) % local_L) * L2 +
                               ((j + local_L) % local_L) * local_L +
                               ((i + local_L) % local_L)];
                };

#ifdef SIMULATION_MODE_SG
                auto get_bond = [&](int i, int j, int k, int c) {
                  return b_ptr[(((k + local_L) % local_L) * L2 +
                                ((j + local_L) % local_L) * local_L +
                                ((i + local_L) % local_L)) *
                                   3 +
                               c];
                };
#endif

                int32_t j0 = s_ptr[index];

                // 6方向の隣接スピンと結合
#ifdef SIMULATION_MODE_SG
                int32_t j1 =
                    j0 ^ get_spin(xi + 1, yj, zk) ^ b_ptr[index * 3 + 0];
                int32_t j2 =
                    j0 ^ get_spin(xi, yj + 1, zk) ^ b_ptr[index * 3 + 1];
                int32_t j3 =
                    j0 ^ get_spin(xi, yj, zk + 1) ^ b_ptr[index * 3 + 2];
                int32_t j4 =
                    j0 ^ get_spin(xi - 1, yj, zk) ^ get_bond(xi - 1, yj, zk, 0);
                int32_t j5 =
                    j0 ^ get_spin(xi, yj - 1, zk) ^ get_bond(xi, yj - 1, zk, 1);
                int32_t j6 =
                    j0 ^ get_spin(xi, yj, zk - 1) ^ get_bond(xi, yj, zk - 1, 2);
#elif defined(SIMULATION_MODE_FM)
                int32_t j1 = j0 ^ get_spin(xi + 1, yj, zk);
                int32_t j2 = j0 ^ get_spin(xi, yj + 1, zk);
                int32_t j3 = j0 ^ get_spin(xi, yj, zk + 1);
                int32_t j4 = j0 ^ get_spin(xi - 1, yj, zk);
                int32_t j5 = j0 ^ get_spin(xi, yj - 1, zk);
                int32_t j6 = j0 ^ get_spin(xi, yj, zk - 1);
#elif defined(SIMULATION_MODE_AFM)
                int32_t j1 = ~(j0 ^ get_spin(xi + 1, yj, zk));
                int32_t j2 = ~(j0 ^ get_spin(xi, yj + 1, zk));
                int32_t j3 = ~(j0 ^ get_spin(xi, yj, zk + 1));
                int32_t j4 = ~(j0 ^ get_spin(xi - 1, yj, zk));
                int32_t j5 = ~(j0 ^ get_spin(xi, yj - 1, zk));
                int32_t j6 = ~(j0 ^ get_spin(xi, yj, zk - 1));
#endif

                // メトロポリス判定ビット演算
                int32_t n1 = j1 ^ j2;
                int32_t n0 = j3 ^ n1;
                n1 = (j1 & j2) ^ (j3 & n1);
                int32_t m1 = j4 ^ j5;
                int32_t m0 = j6 ^ m1;
                m1 = (j4 & j5) ^ (j6 & m1);
                int32_t i0 = n0 & m0;
                int32_t i2 = n1 ^ m1;
                int32_t i1 = i0 ^ i2;
                i2 = (n1 & m1) ^ (i0 & i2);
                i0 = n0 ^ m0;

                if (use_kexp) {
                  // 本来独立であるべき2つのレプリカの動きがシンクロしてしまう事象への対処
#ifdef SIMULATION_MODE_SG
                  int32_t k0 = 0, k1v = 0, k2v = 0;
                  int32_t mask = 0x55555555;
                  for (int i = 0; i < 2; ++i) {
                    uint32_t rnd_raw = rnd_ptr[index * 2 + i];
                    uint32_t rand_val = rnd_raw % NRMAX;

                    k0 |= k_ptr[rand_val * 3 + 0] & (mask << i);
                    k1v |= k_ptr[rand_val * 3 + 1] & (mask << i);
                    k2v |= k_ptr[rand_val * 3 + 2] & (mask << i);
                  }
#else
                  // 乱数取得: index: 0 ~ N-1
                  uint32_t rnd_raw = rnd_ptr[index];
                  uint32_t rand_val = rnd_raw % NRMAX;

                  int32_t k0 = k_ptr[rand_val * 3 + 0],
                          k1v = k_ptr[rand_val * 3 + 1],
                          k2v = k_ptr[rand_val * 3 + 2];
#endif
                  int32_t ic = k2v | (i0 & k0) | (i1 & k1v) | (i2 | (i1 & i0));
                  s_ptr[index] = j0 ^ ic;
                } else {
                  for (int i = 0; i < 32; ++i) {
                    int b0 = (i0 >> i) & 1;
                    int b1 = (i1 >> i) & 1;
                    int b2 = (i2 >> i) & 1;
                    if (((b2 << 2) | (b1 << 1) | b0) <= 2) {
                      uint32_t rnd_int = rnd_ptr[index * 32 + i];
                      if (b0) {
                        // dE = +8
                        if (rnd_int < t_ptr[3 * i + 0])
                          j0 ^= (1 << i);
                      } else if (b1) {
                        // dE = +4
                        if (rnd_int < t_ptr[3 * i + 1])
                          j0 ^= (1 << i);
                      } else {
                        // dE = +12
                        if (rnd_int < t_ptr[3 * i + 2])
                          j0 ^= (1 << i);
                      }
                    } else {
                      j0 ^= (1 << i);
                    }
                  }
                  s_ptr[index] = j0;
                }
              });
        });
      }

      // 次の処理がこのカーネルを待てるようにイベントを記録
      replica_deps[i] = {step_evt};
    }

    q.wait();
  }

  // 統計量計算
  void calcStats(int64_t *overlap_h, int64_t *mag_h, int64_t *mst_h,
                 int64_t *energy_h) {
    for (int i = 0; i < MULTI; ++i) {
      const size_t local_sz = 256;
      size_t global_sz = ((N + local_sz - 1) / local_sz) * local_sz;
      auto s_ptr = d_spins[i];
#ifdef SIMULATION_MODE_SG
      auto b_ptr = d_bonds[i];
      auto res_ov = d_sum_overlap[i];
#elif defined(SIMULATION_MODE_FM)
      auto res_mag = d_sum_mag[i];
#elif defined(SIMULATION_MODE_AFM)
      auto res_mst = d_sum_mag_stag[i];
#endif
      auto res_en = d_sum_energy[i];
      const int local_L = L;
      const int L2 = L * L;
      const int local_N = N;

#ifdef SIMULATION_MODE_SG
      sycl::event evt_clear_ov =
          q.memset(res_ov, 0, 16 * sizeof(int64_t), replica_deps[i]);
#elif defined(SIMULATION_MODE_FM)
      sycl::event evt_clear_mag =
          q.memset(res_mag, 0, 32 * sizeof(int64_t), replica_deps[i]);
#elif defined(SIMULATION_MODE_AFM)
      sycl::event evt_clear_mst =
          q.memset(res_mst, 0, 32 * sizeof(int64_t), replica_deps[i]);
#endif
      sycl::event evt_clear_en =
          q.memset(res_en, 0, 32 * sizeof(int64_t), replica_deps[i]);

      replica_deps[i] = {evt_clear_en};
#ifdef SIMULATION_MODE_SG
      replica_deps[i].push_back(evt_clear_ov);
#elif defined(SIMULATION_MODE_FM)
      replica_deps[i].push_back(evt_clear_mag);
#elif defined(SIMULATION_MODE_AFM)
      replica_deps[i].push_back(evt_clear_mst);
#endif

#ifdef SIMULATION_MODE_SG
      sycl::event evt_ov = q.submit([&](handler &h) {
        h.depends_on(replica_deps[i]);
        h.depends_on(evt_clear_ov);

        h.parallel_for(
            nd_range<1>(range<1>(global_sz), range<1>(local_sz)),
            [=](nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
              int gid = item.get_global_id(0);
              auto g = item.get_group();

              int32_t val = 0;
              if (gid < local_N) {
                val = s_ptr[gid];
              }

              for (int k = 0; k < 16; ++k) {
                int diff = ((val >> (2 * k)) & 1) ^ ((val >> (2 * k + 1)) & 1);
                int sum = reduce_over_group(g, diff, plus<>());
                if (g.leader()) {
                  atomic_ref<int64_t, memory_order::relaxed,
                             memory_scope::device,
                             access::address_space::global_space>(res_ov[k])
                      .fetch_add(sum);
                }
              }
            });
      });
#elif defined(SIMULATION_MODE_FM)
      sycl::event evt_mst = q.submit([&](handler &h) {
        h.depends_on(replica_deps[i]);
        h.depends_on(evt_clear_mag);

        h.parallel_for(nd_range<1>(range<1>(global_sz), range<1>(local_sz)),
                       [=](nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                         int gid = item.get_global_id(0);
                         auto g = item.get_group();

                         // ローカルスピンの取得
                         int32_t val = (gid < local_N) ? s_ptr[gid] : 0;

                         // 座標計算
                         int zk = gid / L2;
                         int yj = (gid % L2) / local_L;
                         int xi = gid % local_L;

                         // 32システム分（ビットごと）の集計
                         for (int k = 0; k < 32; ++k) {
                           // ビットが0ならスピン+1, ビットが1ならスピン-1
                           int bit = (val >> k) & 1;

                           // グループ内縮約 (reduce)
                           int sum = reduce_over_group(g, bit, plus<>());

                           if (g.leader()) {
                             atomic_ref<int64_t, memory_order::relaxed,
                                        memory_scope::device,
                                        access::address_space::global_space>(
                                 res_mag[k])
                                 .fetch_add(sum);
                           }
                         }
                       });
      });
#elif defined(SIMULATION_MODE_AFM)
      sycl::event evt_mst = q.submit([&](handler &h) {
        h.depends_on(replica_deps[i]);
        h.depends_on(evt_clear_mst);

        h.parallel_for(nd_range<1>(range<1>(global_sz), range<1>(local_sz)),
                       [=](nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                         int gid = item.get_global_id(0);
                         auto g = item.get_group();

                         // ローカルスピンの取得
                         int32_t val = (gid < local_N) ? s_ptr[gid] : 0;

                         // 座標計算
                         int zk = gid / L2;
                         int yj = (gid % L2) / local_L;
                         int xi = gid % local_L;

                         // パリティチェック: (x+y+z) が偶数なら +1, 奇数なら -1
                         int parity = (xi + yj + zk) % 2;

                         // 32システム分（ビットごと）の集計
                         for (int k = 0; k < 32; ++k) {
                           int bit = (val >> k) & 1;

                           // パリティに基づくスタガード係数の適用
                           bit = (bit ^ parity) & 1;

                           // グループ内縮約 (reduce)
                           int sum = reduce_over_group(g, bit, plus<>());

                           if (g.leader()) {
                             atomic_ref<int64_t, memory_order::relaxed,
                                        memory_scope::device,
                                        access::address_space::global_space>(
                                 res_mst[k])
                                 .fetch_add(sum);
                           }
                         }
                       });
      });
#endif

      sycl::event evt_en;
      if (!no_calc_en)
        evt_en = q.submit([&](handler &h) {
          h.depends_on(replica_deps[i]);
          h.depends_on(evt_clear_en);

          h.parallel_for(
              nd_range<1>(range<1>(global_sz), range<1>(local_sz)),
              [=](nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                int gid = item.get_global_id(0);
                auto g = item.get_group();

                int64_t e_bits = 0;
                if (gid < local_N) {
                  // インデックス計算
                  int zk = gid / L2;
                  int yj = (gid % L2) / local_L;
                  int xi = gid % local_L;

                  auto get_spin = [&](int i, int j, int k) {
                    return s_ptr[((k + local_L) % local_L) * L2 +
                                 ((j + local_L) % local_L) * local_L +
                                 ((i + local_L) % local_L)];
                  };

                  e_bits =
#ifdef SIMULATION_MODE_SG
                      spread_bits(s_ptr[gid] ^ get_spin(xi + 1, yj, zk) ^
                                  b_ptr[gid * 3 + 0]) +
                      spread_bits(s_ptr[gid] ^ get_spin(xi, yj + 1, zk) ^
                                  b_ptr[gid * 3 + 1]) +
                      spread_bits(s_ptr[gid] ^ get_spin(xi, yj, zk + 1) ^
                                  b_ptr[gid * 3 + 2]);
#elif defined(SIMULATION_MODE_FM)
                      spread_bits(s_ptr[gid] ^ get_spin(xi + 1, yj, zk)) +
                      spread_bits(s_ptr[gid] ^ get_spin(xi, yj + 1, zk)) +
                      spread_bits(s_ptr[gid] ^ get_spin(xi, yj, zk + 1));
#elif defined(SIMULATION_MODE_AFM)
                      spread_bits(~(s_ptr[gid] ^ get_spin(xi + 1, yj, zk))) +
                      spread_bits(~(s_ptr[gid] ^ get_spin(xi, yj + 1, zk))) +
                      spread_bits(~(s_ptr[gid] ^ get_spin(xi, yj, zk + 1)));
#endif
                }

                for (int i = 0; i < 32; ++i) {
                  int count = (int)((e_bits >> (i * 2)) & 3);
                  int sum = reduce_over_group(g, count, plus<>());
                  if (g.leader()) {
                    atomic_ref<int64_t, memory_order::relaxed,
                               memory_scope::device,
                               access::address_space::global_space>(res_en[i])
                        .fetch_add(sum);
                  }
                }
              });
        });

#ifdef SIMULATION_MODE_SG
      sycl::event evt_read_ov =
          q.copy(d_sum_overlap[i], &overlap_h[i * 16], 16, evt_ov);
#elif defined(SIMULATION_MODE_FM)
      sycl::event evt_read_mag =
          q.copy(d_sum_mag[i], &mag_h[i * 32], 32, evt_mst);
#elif defined(SIMULATION_MODE_AFM)
      sycl::event evt_read_mst =
          q.copy(d_sum_mag_stag[i], &mst_h[i * 32], 32, evt_mst);
#endif
      if (!no_calc_en) {
        sycl::event evt_read_en =
            q.copy(d_sum_energy[i], &energy_h[i * 32], 32, evt_en);

        replica_deps[i] = {evt_read_en};
      }
#ifdef SIMULATION_MODE_SG
      replica_deps[i].push_back(evt_read_ov);
#elif defined(SIMULATION_MODE_FM)
      replica_deps[i].push_back(evt_read_mag);
#elif defined(SIMULATION_MODE_AFM)
      replica_deps[i].push_back(evt_read_mst);
#endif
    }
    q.wait();
  }

  void get_spin(int sys_i, int32_t *out) {
    int rep = sys_i / 32;
    int bit = sys_i % 32;
    q.copy(d_spins[rep], out, L * L * L, replica_deps[rep]).wait();
    for (int i = 0; i < L * L * L; ++i) {
      out[i] = (out[i] >> bit) & 1;
    }
  }

#ifdef SIMULATION_MODE_SG
  void get_bond(int sys_i, int32_t *out) {
    int rep = sys_i / 32;
    int bit = sys_i % 32;
    q.copy(d_bonds[rep], out, L * L * L * 3, replica_deps[rep]).wait();
    for (int i = 0; i < L * L * L * 3; ++i) {
      out[i] = (out[i] >> bit) & 1;
    }
  }
#endif

  // パラレルテンパリング用: 系 sys_i と sys_j のスピン・ボンドをGPUで交換
  // 系インデックス: 0 〜 32*MULTI-1
  // 系 i は d_spins[i/32] の (i%32) ビット目に対応
  void swap(int sys_i, int sys_j) {
    const int total_systems = 32 * MULTI;
    if (sys_i < 0 || sys_i >= total_systems || sys_j < 0 ||
        sys_j >= total_systems) {
      std::cerr << "swap: system indices must be in range [0, "
                << total_systems - 1 << "]" << std::endl;
      return;
    }
    if (sys_i == sys_j) {
      return; // 同じ系なら何もしない
    }

    // 系インデックスからレプリカ番号とビット位置を計算
    int rep_i = sys_i / 32;
    int bit_i = sys_i % 32;
    int rep_j = sys_j / 32;
    int bit_j = sys_j % 32;

    int local_N = N;

    // スピンの交換
    {
      auto s_ptr_i = d_spins[rep_i];
      auto s_ptr_j = d_spins[rep_j];

      std::vector<sycl::event> deps;
      deps.insert(deps.end(), replica_deps[rep_i].begin(),
                  replica_deps[rep_i].end());
      if (rep_i != rep_j) {
        deps.insert(deps.end(), replica_deps[rep_j].begin(),
                    replica_deps[rep_j].end());
      }

      sycl::event evt = q.submit([&](handler &h) {
        h.depends_on(deps);

        h.parallel_for(range<1>(local_N), [=](id<1> idx) {
          int site = idx[0];

          // 各レプリカから該当ビットを取り出す
          int32_t val_i = (s_ptr_i[site] >> bit_i) & 1;
          int32_t val_j = (s_ptr_j[site] >> bit_j) & 1;

          int32_t diff = val_i ^ val_j;
          s_ptr_i[site] ^= (diff << bit_i);
          s_ptr_j[site] ^= (diff << bit_j);
        });
      });

      replica_deps[rep_i] = {evt};
      if (rep_i != rep_j) {
        replica_deps[rep_j] = {evt};
      }
    }

#ifdef SIMULATION_MODE_SG
    // ボンドの交換 (スピングラスモードのみ)
    if (!same_bond) {
      auto b_ptr_i = d_bonds[rep_i];
      auto b_ptr_j = d_bonds[rep_j];

      std::vector<sycl::event> deps;
      deps.insert(deps.end(), replica_deps[rep_i].begin(),
                  replica_deps[rep_i].end());
      if (rep_i != rep_j) {
        deps.insert(deps.end(), replica_deps[rep_j].begin(),
                    replica_deps[rep_j].end());
      }

      sycl::event evt = q.submit([&](handler &h) {
        h.depends_on(deps);

        // ボンドは N * 3 要素
        h.parallel_for(range<1>(local_N * 3), [=](id<1> idx) {
          int bond_idx = idx[0];

          // 各レプリカから該当ビットを取り出す
          int32_t val_i = (b_ptr_i[bond_idx] >> bit_i) & 1;
          int32_t val_j = (b_ptr_j[bond_idx] >> bit_j) & 1;

          int32_t diff = val_i ^ val_j;
          b_ptr_i[bond_idx] ^= (diff << bit_i);
          b_ptr_j[bond_idx] ^= (diff << bit_j);
        });
      });

      replica_deps[rep_i] = {evt};
      if (rep_i != rep_j) {
        replica_deps[rep_j] = {evt};
      }
    }
#endif

    q.wait();
  }

  void run(int burn_in, int calc_steps) {
    // スイープ数が少ないとkexp更新がボトルネックになるための処理
    if (!made_kexp && !disable_kexp && (burn_in + calc_steps) * N / 10 > NRMAX) {
      updateKexp();
    }

    for (int i = 0; i < burn_in; ++i) {
      step();
    }

#ifdef SIMULATION_MODE_SG
    std::vector<double> sum_q(16 * MULTI, 0), sum_q2(16 * MULTI, 0),
        sum_q4(16 * MULTI, 0), sum_pair_e(16 * MULTI, 0),
        sum_pair_e2(16 * MULTI, 0), sum_q2e(16 * MULTI, 0),
        sum_q4e(16 * MULTI, 0);
    std::vector<double> sum_mst2(32 * MULTI, 0), sum_mst4(32 * MULTI, 0);
#elif defined(SIMULATION_MODE_FM)
    std::vector<double> sum_m(32 * MULTI, 0), sum_m2(32 * MULTI, 0),
        sum_m4(32 * MULTI, 0);
#elif defined(SIMULATION_MODE_AFM)
    std::vector<double> sum_mst(32 * MULTI, 0), sum_mst2(32 * MULTI, 0),
        sum_mst4(32 * MULTI, 0), sum_me_st(32 * MULTI, 0);
#endif
    std::vector<double> sum_e(32 * MULTI, 0), sum_e2(32 * MULTI, 0);
    for (int i = 0; i < calc_steps; ++i) {
      step();
      std::vector<int64_t> ov(16 * MULTI), mag(32 * MULTI), mst(32 * MULTI),
          en(32 * MULTI);
      calcStats(ov.data(), mag.data(), mst.data(), en.data());
#ifdef SIMULATION_MODE_SG
      for (int k = 0; k < 16 * MULTI; ++k) {
#else
      for (int k = 0; k < 32 * MULTI; ++k) {
#endif
#ifdef SIMULATION_MODE_SG
        double q_val = (double)N - 2.0 * ov[k];
        sum_q[k] += q_val;
        sum_q2[k] += q_val * q_val;
        sum_q4[k] += q_val * q_val * q_val * q_val;
        double en1 = (double)en[2 * k] * 2.0 - 3.0 * N;
        double en2 = (double)en[2 * k + 1] * 2.0 - 3.0 * N;
        double etot = en1 + en2;
        sum_pair_e[k] += etot;
        sum_pair_e2[k] += etot * etot;
        sum_q2e[k] += q_val * q_val * etot;
        sum_q4e[k] += q_val * q_val * q_val * q_val * etot;
        sum_e[2 * k] += en1;
        sum_e2[2 * k] += en1 * en1;
        sum_e[2 * k + 1] += en2;
        sum_e2[2 * k + 1] += en2 * en2;
#else
        double e1 = (double)en[k] * 2.0 - 3.0 * N;
        sum_e[k] += e1;
        sum_e2[k] += e1 * e1;
#if defined(SIMULATION_MODE_FM)
        double m_val = (double)N - 2.0 * mag[k];
        sum_m[k] += std::abs(m_val);
        sum_m2[k] += m_val * m_val;
        sum_m4[k] += m_val * m_val * m_val * m_val;
#elif defined(SIMULATION_MODE_AFM)
        double m_st = (double)N - 2.0 * mst[k]; // 計算したスタガード磁化
        sum_mst[k] += std::abs(m_st); // 絶対値をとる（対称性破れのため）
        sum_mst2[k] += m_st * m_st;
        sum_mst4[k] += m_st * m_st * m_st * m_st;
        sum_me_st[k] += std::abs(m_st) * e1;
#endif
#endif
      }
    }

    if (calc_steps == 0) {
      std::cout << 2 << "\n" << std::flush;
      std::cout << "{}\n" << std::flush;
      return;
    }

#ifdef SIMULATION_MODE_SG
    std::vector<double> q1(16 * MULTI, 0), q2(16 * MULTI, 0), q4(16 * MULTI, 0);
    std::vector<double> pair_e1(16 * MULTI, 0), pair_e2(16 * MULTI, 0);
    std::vector<double> q2e(16 * MULTI, 0), q4e(16 * MULTI, 0);
    std::vector<double> chi(16 * MULTI, 0), cov_term(16 * MULTI, 0),
        d_chi_dT(16 * MULTI, 0), d_binder_dT(16 * MULTI, 0);
    std::vector<double> C(16 * MULTI, 0), binder(16 * MULTI, 0);
    for (int k = 0; k < 16 * MULTI; ++k) {
      double T = temperatures[k * 2];
      q1[k] = sum_q[k] / calc_steps / N;
      q2[k] = sum_q2[k] / calc_steps / N / N;
      q4[k] = sum_q4[k] / calc_steps / N / N / N / N;
      pair_e1[k] = sum_pair_e[k] / calc_steps / N;
      pair_e2[k] = sum_pair_e2[k] / calc_steps / N / N;
      q2e[k] = sum_q2e[k] / calc_steps / N / N / N;
      q4e[k] = sum_q4e[k] / calc_steps / N / N / N / N / N;
      chi[k] = q2[k] * N;
      cov_term[k] = (q2e[k] - q2[k] * pair_e1[k]) * N;
      d_chi_dT[k] = (1.0 / T) * (-chi[k] + (2.0 * N / T / T) * cov_term[k]);
      C[k] = (pair_e2[k] - pair_e1[k] * pair_e1[k]) * N / T / T;
      binder[k] = 0.5 * (3.0 - q4[k] / (q2[k] * q2[k]));

      // 温度 T における微分項の計算 (1/T^2 を忘れずに！)
      double dq2_dT = (1.0 / (T * T)) * (q2e[k] - q2[k] * pair_e1[k]);
      double dq4_dT = (1.0 / (T * T)) * (q4e[k] - q4[k] * pair_e1[k]);
      // Binder比の傾き
      // g = 0.5 * (3 - q4 / (q2 * q2)) の微分
      d_binder_dT[k] = (1.0 / (2.0 * q2[k] * q2[k])) *
                       (2.0 * (q4[k] / q2[k]) * dq2_dT - dq4_dT);
    }
#elif defined(SIMULATION_MODE_FM)
    std::vector<double> res_m(32 * MULTI, 0), res_chi(32 * MULTI, 0),
        res_binder(32 * MULTI, 0);
    std::vector<double> res_C(32 * MULTI, 0);
#elif defined(SIMULATION_MODE_AFM)
  std::vector<double> chi_st(32 * MULTI, 0), binder_st(32 * MULTI, 0),
      m_st(32 * MULTI, 0), me_st(32 * MULTI, 0);
#endif
    std::vector<double> e1(32 * MULTI, 0), e2(32 * MULTI, 0);
    for (int k = 0; k < 32 * MULTI; ++k) {
      double T = temperatures[k];

      e1[k] = sum_e[k] / calc_steps / N;
      e2[k] = sum_e2[k] / calc_steps / N / N;

#if defined(SIMULATION_MODE_FM)
      double m1 = sum_m[k] / calc_steps / N;
      double m2 = sum_m2[k] / calc_steps / N / N;
      double m4 = sum_m4[k] / calc_steps / N / N / N / N;
      res_m[k] = std::sqrt(sum_m2[k] / calc_steps) / N;
      // Susceptibility: N * (<m^2> - <|m|>^2) / T
      res_chi[k] = N * (m2 - m1 * m1) / T;
      // Binder Cumulant: 1 - <M^4> / (3<M^2>^2)
      res_binder[k] = 1.0 - m4 / (3.0 * m2 * m2);
      res_C[k] = (e2[k] - e1[k] * e1[k]) * N / T / T;
#elif defined(SIMULATION_MODE_AFM)
      double mst2_avg = sum_mst2[k] / calc_steps / N / N;
      double mst4_avg = sum_mst4[k] / calc_steps / N / N / N / N;
      double me = sum_me_st[k] / calc_steps / N / N;

      // スタガード磁化率
      chi_st[k] = mst2_avg * N / T;

      // Binder比
      binder_st[k] = 1.0 - mst4_avg / (3.0 * mst2_avg * mst2_avg);

      m_st[k] = sum_mst[k] / (calc_steps * N);

      me_st[k] = me;
#endif
    }

    // ビットレベルでの判定（-ffast-math 環境下での回避策）
    auto formatDouble = [](double val) -> std::string {
      uint64_t bits;
      // doubleのビット表現をuint64_tにコピー
      std::memcpy(&bits, &val, sizeof(bits));

      // IEEE 754: 指数部(11bit)がすべて1
      const uint64_t EXPONENT_MASK = 0x7FF0000000000000ULL;
      const uint64_t FRACTION_MASK = 0x000FFFFFFFFFFFFFULL;

      // 指数部を取り出す
      if ((bits & EXPONENT_MASK) == EXPONENT_MASK) {
        // 仮数部が非ゼロならNaN
        if (bits & FRACTION_MASK) {
          return "NaN";
        }
        // 仮数部が0ならInfinity (符号ビットで判定)
        return (val < 0) ? "-Infinity" : "Infinity";
      }

      std::ostringstream oss;
      oss << std::scientific
          << std::setprecision(std::numeric_limits<double>::max_digits10 - 1)
          << val;
      return oss.str();
    };
    std::stringstream ss;
    ss << "{";
    auto print_vec = [&](const std::string &name, auto func, int count,
                         bool last = false) {
      ss << "\"" << name << "\":[";
      for (int k = 0; k < count * MULTI; ++k)
        ss << formatDouble(func(k)) << (k < count * MULTI - 1 ? "," : "");
      ss << "]" << (last ? "" : ",");
    };
#ifdef SIMULATION_MODE_SG
    print_vec("q1", [&](int k) { return q1[k]; }, 16);
    print_vec("q2", [&](int k) { return q2[k]; }, 16);
    print_vec("q4", [&](int k) { return q4[k]; }, 16);
    print_vec("chi", [&](int k) { return chi[k]; }, 16);
    print_vec("d_chi_dT", [&](int k) { return d_chi_dT[k]; }, 16);
    print_vec("C", [&](int k) { return C[k]; }, 16);
    print_vec("binder", [&](int k) { return binder[k]; }, 16);
    print_vec("d_binder_dT", [&](int k) { return d_binder_dT[k]; }, 16);
    print_vec("pair_e1", [&](int k) { return pair_e1[k]; }, 16);
    print_vec("pair_e2", [&](int k) { return pair_e2[k]; }, 16);
#elif defined(SIMULATION_MODE_FM)
    print_vec("m", [&](int k) { return res_m[k]; }, 32);
    print_vec("chi", [&](int k) { return res_chi[k]; }, 32);
    print_vec("binder", [&](int k) { return res_binder[k]; }, 32);
    print_vec("C", [&](int k) { return res_C[k]; }, 32);
#elif defined(SIMULATION_MODE_AFM)
    print_vec("chi_st", [&](int k) { return chi_st[k]; }, 32);
    print_vec("binder_st", [&](int k) { return binder_st, 32 [k]; }, 32);
    print_vec("m_st", [&](int k) { return m_st[k]; }, 32);
    print_vec("me_st", [&](int k) { return me_st[k]; }, 32);
#endif
    print_vec("e1", [&](int k) { return e1[k]; }, 32);
    print_vec("e2", [&](int k) { return e2[k]; }, 32, true);
    ss << "}\n";
    std::string out = ss.str();
    std::cout << out.length() << "\n" << std::flush;
    std::cout << out << "\n" << std::flush;
  }
};

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;

  int L = std::stoi(argv[1]);
  if (L <= 0) {
    std::cerr << "L must be positive" << std::endl;
    return 1;
  }
  std::cerr << "L: " << L << std::endl;

#ifdef SIMULATION_MODE_SG
  std::cerr << "Mode: SPIN-GLASS" << std::endl;
#elif defined(SIMULATION_MODE_FM)
  std::cerr << "Mode: FERROMAGNETIC" << std::endl;
#elif defined(SIMULATION_MODE_AFM)
  std::cerr << "Mode: ANTI-FERROMAGNETIC" << std::endl;
#endif

  int MULTI = 1;
  char *multi_env = std::getenv("MULTI");
  if (multi_env) {
    MULTI = std::stoi(multi_env);
    if (MULTI <= 0)
      MULTI = 1;
  }
  std::cerr << "MULTI: " << MULTI << std::endl;

  no_calc_en = !!std::getenv("NO_CALC_EN");
  std::cerr << "NO_CALC_EN: " << no_calc_en << std::endl;

  setenv("ONEAPI_DEVICE_SELECTOR", ONEAPI_DEVICE_SELECTOR, 0);

  queue q{default_selector_v};
  std::cerr << "Running on: " << q.get_device().get_info<info::device::name>()
            << std::endl;

  SpinGlassSimulator sim(L, q, MULTI);
  std::string line;
  int burn_in = 1000, calc_steps = 2000;
  while (std::getline(std::cin, line)) {
    if (line == "exit")
      break;
    if (line == "reset_spins")
      sim.resetSpins();
#ifdef SIMULATION_MODE_SG
    else if (line == "reset_bonds")
      sim.resetBonds(false);
    else if (line == "reset_bonds_same")
      sim.resetBonds(true);
    else if (line.rfind("get_bond:", 0) == 0) {
      std::istringstream iss(line.substr(9));
      int idx = 0;
      iss >> idx;
      std::vector<int32_t> data(L * L * L * 3);
      sim.get_bond(idx, data.data());
      std::fwrite(data.data(), sizeof(int32_t), data.size(), stdout);
      std::fflush(stdout);
    }
#endif
    else if (line == "randomize_spins")
      sim.randomizeSpins();
    else if (line.rfind("T:", 0) == 0) {
      std::vector<double> temps;
      std::istringstream iss(line.substr(2));
      double t;
      while (iss >> t)
        temps.push_back(t);
      if (temps.size() == 32 * MULTI)
        sim.setTemperatures(temps);
      else
        std::cerr << "expected " << 32 * MULTI << " temperatures, got "
                  << temps.size() << std::endl;
    } else if (line.rfind("resize:", 0) == 0) {
      int new_L = std::stoi(line.substr(7));
      if (new_L != L) {
        L = new_L;
        sim.realloc(L);
      }
    } else if (line.rfind("burn_in:", 0) == 0)
      burn_in = std::stoi(line.substr(8));
    else if (line.rfind("calc_steps:", 0) == 0)
      calc_steps = std::stoi(line.substr(11));
    else if (line == "run_simulation")
      sim.run(burn_in, calc_steps);
    else if (line.rfind("get_spin:", 0) == 0) {
      std::istringstream iss(line.substr(9));
      int idx = 0;
      iss >> idx;
      std::vector<int32_t> data(L * L * L);
      sim.get_spin(idx, data.data());
      std::fwrite(data.data(), sizeof(int32_t), data.size(), stdout);
      std::fflush(stdout);
    } else if (line.rfind("swap:", 0) == 0) {
      std::vector<int> indices;
      std::istringstream iss(line.substr(5));
      int idx;
      while (iss >> idx)
        indices.push_back(idx);
      if (indices.size() == 2)
        sim.swap(indices[0], indices[1]);
      else
        std::cerr << "expected 2 indices, got " << indices.size() << std::endl;
    } else if (line.rfind("make_kexp", 0) == 0) {
      if (!sim.made_kexp) {
        sim.updateKexp();
        std::cout << "done\n" << std::flush;
      } else {
        std::cerr << "already made kexp" << std::endl;
      }
    } else if (line.rfind("disable_kexp", 0) == 0) {
      sim.made_kexp = false;
      sim.disable_kexp = true;
    } else
      std::cerr << "unknown command: " << line << std::endl;
  }
  return 0;
}
