#include <thread>
#include <vector>
#include <iostream>
#include <atomic>
#include "cli.h"
#include "region_queue.h"
#include "producer.h"
#include "abpoa_runner.h"
#include "assembler.h"
#include "windowing.h"
#include <memory>
#include "step_timer.h"
#include "chrom_filter.h"
#include <chrono>
#include <cstdio>

#include "abpoa_metrics.h"
int main(int argc, char** argv){
    ScopedTimer __all__("__TOTAL__");
    auto cli = parse_cli(argc, argv);

    BlockingQueue q(std::max(1, cli.threads) * 1000);

    ProducerConfig prodConfig;
    prodConfig.bam_path = cli.bam;
    prodConfig.win_len  = cli.win_len;
    prodConfig.overlap  = cli.overlap;
    Producer prod(prodConfig, q);

    AsmConfig asmCfg;
    asmCfg.out_fa  = cli.out_fa;
    asmCfg.overlap = cli.overlap;
    Assembler asmblr(asmCfg);

    AbpoaParams ap;
    ap.max_n_cons = 1;
    ap.min_freq   = 0.0;
    ap.align_mode = 0;
    AbpoaRunner runner(ap);

    std::atomic<bool> producer_ok{false};

    // 先预注册所有chr窗口（用于按序拼接）
    // 注意：这里简化为在Producer内部再次生成一遍；若要避免重复，可重构。
    samFile* fp = sam_open(cli.bam.c_str(), "r"); bam_hdr_t* hdr = sam_hdr_read(fp);
    
    auto __run_t0 = std::chrono::steady_clock::now();
    long long total_regions = 0;
    {
        ScopedTimer _t_("step:windowing+register");
        for (int tid = 0; tid < hdr->n_targets; ++tid) {
            const char* nm = hdr->target_name[tid];
            if (!is_canonical_chr(nm)) continue;                  // ← 只保留正常染色体

            const std::string chr(nm);
            std::vector<Window> windows =
                build_windows_for_chr(chr, hdr->target_len[tid], cli.win_len, cli.overlap);

            total_regions += static_cast<long long>(windows.size()); // 分母只统计正常染色体
            asmblr.register_chr(chr, windows);
        }
    }

    bam_hdr_destroy(hdr); sam_close(fp);

    std::atomic<long long> regions_done(0);  // ← 新增：已处理窗口计数
    std::atomic<long long> progress_last_ns(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

        // 启动消费者线程
    std::vector<std::thread> workers;
    for (int i=0;i<cli.threads;i++){
        workers.emplace_back([&]{
            while (true){
                auto item = q.pop(); if (!item) break;

                TwoCons tc = runner.consensus2(item->seqs);

                asmblr.submit(item->chr, item->index, std::move(tc.c1), std::move(tc.c2));
                // ---- 新增：进度统计（每 10000 个窗口打印一次）----
                long long done = regions_done.fetch_add(1, std::memory_order_relaxed) + 1;
                if (done % 10000 == 0) {
                    double pct = (total_regions > 0) ? (100.0 * (double)done / (double)total_regions) : 0.0;
                    long long now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    long long prev_ns = progress_last_ns.exchange(now_ns);
                    double batch_ms = (now_ns - prev_ns) / 1e6;
                    std::fprintf(stderr, "[PROGRESS] processed=%lld / %lld (%.2f%%) last_10k_wall=%.3f s\n",
                                (long long)done, (long long)total_regions, pct, batch_ms / 1000.0);
                    // 可选：std::fflush(stderr);
                }
                // ---------------------------------------------

            }
        });
    }

    // 生产者（单线程）
    std::thread prod_thr([&]{ producer_ok = prod.run(); });

    prod_thr.join();
    for (auto& th: workers) th.join();

    long long bad_overlap_breaks = 0;
    {
        ScopedTimer _t_("step:assembler.finalize");
        asmblr.finalize();
    }
    bad_overlap_breaks = asmblr.bad_overlap_breaks();
    std::fprintf(stderr, "[ASSEMBLER] bad_overlap_breaks=%lld\n", bad_overlap_breaks);

    if (!producer_ok){ std::cerr << "Producer failed.\n"; return 1; }
    auto __run_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - __run_t0).count();
    fprintf(stderr, "[RUN-TOTAL] wall=%.3f s\n", __run_ms / 1000.0);
    
    // （可选）同时打印“仅 abPOA 的总 realtime”
    long long t0 = g_abpoa_t0_ns.load();
    long long t1 = g_abpoa_t1_ns.load();
    if (t0 != -1 && t1 != -1 && t1 > t0) {
        double abpoa_ms = (t1 - t0) / 1e6;
        fprintf(stderr, "[ABPOA-TOTAL] wall=%.3f s\n", abpoa_ms / 1000.0);
    } else {
        fprintf(stderr, "[ABPOA-TOTAL] no-abpoa-or-not-finished\n");
    }

    return 0;
}
