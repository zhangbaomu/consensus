#include "abpoa_runner.h"
#include "abpoa_metrics.h" 
#include <stdexcept>
#include <cctype>
#include <atomic>
#include <iostream>
#include <memory>
#include <algorithm>
#include <cstdio>  // 顶部确保有
#include "abpoa.h"  // 引入 abPOA 库的头文件
#include "simd_instruction.h"  // 引入 SIMD 指令头文件
#include <chrono>
#include <cstdint>

extern "C" {
#include <abpoa.h>

}

std::atomic<long long> g_abpoa_t0_ns{-1};
std::atomic<long long> g_abpoa_t1_ns{-1};
std::atomic<int>       g_abpoa_inflight{0};

static inline long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static inline uint8_t enc(char c){
    switch(std::toupper(c)){
        case 'A': return 0; case 'C': return 1; case 'G': return 2; case 'T': return 3; default: return 4;
    }
}
static inline char dec(uint8_t x){
    switch(x){
        case 0: return 'A';
        case 1: return 'C';
        case 2: return 'G';
        case 3: return 'T';
        default: return 'N';
    }
}

TwoCons AbpoaRunner::consensus2(const std::vector<std::string>& seqs) const {
    TwoCons out;
    if (seqs.empty()) return out;

    // 准备输入：转换为 uint8_t 类型的序列
    std::vector<std::vector<uint8_t>> u8(seqs.size());
    std::vector<uint8_t*> ptr(seqs.size());
    std::vector<int> lens(seqs.size());
    for (size_t i = 0; i < seqs.size(); ++i) {
        u8[i].reserve(seqs[i].size());
        for (char c : seqs[i]) u8[i].push_back(enc(c));
        ptr[i] = u8[i].data();
        lens[i] = (int)u8[i].size();
    }

    // --- abPOA 开始前：标记全局开始 + 在运行计数 ---
    {
        long long t = now_ns();
        long long expect = -1;
        g_abpoa_t0_ns.compare_exchange_strong(expect, t); // 只第一次成功
        g_abpoa_inflight.fetch_add(1, std::memory_order_relaxed);
    }

    abpoa_para_t *abpt = abpoa_init_para();
    abpt->align_mode = ABPOA_GLOBAL_MODE; // GLOBAL alignment
    abpt->out_cons = 1;   // 生成共识序列
    abpt->out_gfa = 0;    // 不生成 GFA
    abpt->max_n_cons = p_.max_n_cons;  // 最大共识条数
    abpt->min_freq = p_.min_freq;      // 最小频率（用于分簇）
    abpt->wb = 10;
    abpt->wf = 0.01;
    abpoa_post_set_para(abpt);

    abpoa_t *ab = abpoa_init();

    // 日志信息打印
    // std::fprintf(stderr, "[ABPOA] n=%zu lens=", seqs.size());
    // for (size_t i = 0; i < seqs.size(); ++i) {
    //     if (i) std::fputc(',', stderr);
    //     std::fprintf(stderr, "%zu", seqs[i].size());
    // }
    // std::fputc('\n', stderr);
    
    // 执行 MSA（多序列比对）

    auto __t0 = std::chrono::steady_clock::now();

    int ret = abpoa_msa(ab, abpt, (int)seqs.size(), nullptr, lens.data(), ptr.data(), nullptr, nullptr);
    if (ret != 0) {
        abpoa_free(ab);
        abpoa_free_para(abpt);
        throw std::runtime_error("abpoa_msa failed");
    }

    // 生成共识序列
    abpoa_generate_consensus(ab, abpt);

    // 获取生成的共识序列
    if (ab->abc && ab->abc->n_cons > 0) {
        int n = std::min(ab->abc->n_cons, p_.max_n_cons);
        
        // 处理第一条共识（hap1）
        if (n >= 1) {
            out.c1.resize(ab->abc->cons_len[0]);
            for (int i = 0; i < ab->abc->cons_len[0]; ++i)
                out.c1[i] = dec(ab->abc->cons_base[0][i]);
        }

        // 处理第二条共识（hap2）
        if (n >= 2) {
            out.c2.resize(ab->abc->cons_len[1]);
            for (int i = 0; i < ab->abc->cons_len[1]; ++i)
                out.c2[i] = dec(ab->abc->cons_base[1][i]);
        }
    }

    // auto __ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     std::chrono::steady_clock::now() - __t0).count();
    
    // int n_seq = seqs.size();
    // fprintf(stderr, "[ABPOA] n=%d took=%lld ms\n", n_seq, (long long)__ms);  
    // 清理资源
    {
        long long t = now_ns();
        int left = g_abpoa_inflight.fetch_sub(1, std::memory_order_relaxed) - 1;
        if (left == 0) {
            g_abpoa_t1_ns.store(t, std::memory_order_relaxed);
        }
    }
    abpoa_free(ab);
    abpoa_free_para(abpt);

    return out;
}
