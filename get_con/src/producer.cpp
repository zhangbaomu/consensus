#include "producer.h"
#include <htslib/faidx.h>
#include <iostream>
#include <unordered_set>
#include <cassert>
#include <algorithm>  // for std::reverse
#include "extract_adapter.h"
#include "step_timer.h"
#include <htslib/sam.h>
#include "chrom_filter.h"
#include <htslib/hts.h>
#include <htslib/bgzf.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cmath>

using std::string; using std::vector; using std::int64_t;

static inline char comp_base(char b){
    switch(b){case 'A':return 'T';case 'C':return 'G';case 'G':return 'C';case 'T':return 'A';default:return 'N';}
}
static std::string cigar_to_string(const bam1_t* b) {
    const uint32_t* cigar = bam_get_cigar(b);
    const int n = b->core.n_cigar;
    std::string s; s.reserve(n * 4);
    for (int i = 0; i < n; ++i) {
        int len = bam_cigar_oplen(cigar[i]);
        char op = bam_cigar_opchr(cigar[i]); // "MIDNSHP=XB"
        // append len
        char buf[32]; int m = snprintf(buf, sizeof(buf), "%d", len);
        s.append(buf, buf + m);
        s.push_back(op);
    }
    return s;
}


// CIGAR映射，抽取窗口内的query子序列；当ref位置在窗口内时追加对应query碱基；
// 对于插入（I），若当前ref位置在窗口内，则也追加插入的碱基。
std::string Producer::clip_read_to_window(const bam1_t* b, int64_t ws, int64_t we) {
    std::string out;
    if (!extract_adapter::extract_window_seq(b, ws, we, out)) return std::string();
    return out;
}


bool Producer::run(){
    ScopedTimer _t_("step:producer.run");
    samFile* fp = sam_open(cfg_.bam_path.c_str(), "r");
    if (!fp){ std::cerr << "Open BAM failed: " << cfg_.bam_path << "\n"; return false; }
    bam_hdr_t* hdr = sam_hdr_read(fp);
    hts_idx_t* idx = sam_index_load(fp, cfg_.bam_path.c_str());
    if (!hdr || !idx){ std::cerr << "BAM header or index missing (need coordinate-sorted & indexed).\n"; return false; }

    // 为输出顺序，先注册每个chr的窗口信息（供Assembler）——通过队列的一个特殊空包或外部调用注册更好；
    // 这里直接在生产阶段按chr推进。

    bam1_t* b = bam_init1();

    long long produced_windows = 0;
    auto batch_start = std::chrono::steady_clock::now();


    for (int tid = 0; tid < hdr->n_targets; ++tid){
        const char* chr = hdr->target_name[tid];
        int64_t chr_len = hdr->target_len[tid];
        auto windows = build_windows_for_chr(chr, chr_len, cfg_.win_len, cfg_.overlap);

        // 活动窗口的累积reads
        std::vector<std::vector<std::string>> win_reads(windows.size());
        size_t next_flush = 0; // 下一个可能被封口（push）的窗口索引

        // 迭代该chr的所有比对
        hts_itr_t* itr = sam_itr_queryi(idx, tid, 0, chr_len);
        if (!itr){ continue; }
        int r;
        while ((r = sam_itr_next(fp, itr, b)) >= 0){
            if (b->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FUNMAP)) continue;
            uint8_t mapq = b->core.qual;
            if(mapq < 20) continue; // 低质量比对过滤
            const int32_t tid1 = b->core.tid;
            if (tid1 < 0) continue; // unmapped（上面已挡）
            const char* nm = hdr->target_name[tid1];
            if (!is_canonical_chr(nm)) continue;


            int64_t rstart = b->core.pos + 1; // 1-based
            // 只要当前read的起点 > 某窗口end，即可封口该窗口（满足你的严格条件）
            while (next_flush < windows.size() && rstart > windows[next_flush].end){
                RegionReads pack{windows[next_flush].chr, windows[next_flush].index,
                                 windows[next_flush].start, windows[next_flush].end,
                                 std::move(win_reads[next_flush])};
                q_.push(std::move(pack));
                ++produced_windows;
                if (produced_windows % 1000 == 0) {
                    auto now = std::chrono::steady_clock::now();
                    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - batch_start).count();
                    std::fprintf(stderr,
                                "[PRODUCER] batch=1000 total=%lld stage=flush queue=%zu wall_ms=%lld\n",
                                produced_windows, q_.size(), (long long)ms);
                    batch_start = now;
                }

                ++next_flush;
            }
            // 将当前read分发到所有重叠窗口（按ref范围）
            // 计算read ref_end
            int64_t ref_pos = rstart; const uint32_t* cig = bam_get_cigar(b);
            const int n_cigar = (int)b->core.n_cigar;   // 显式转成有符号
            for (int i = 0; i < n_cigar; ++i) {
                const int op  = bam_cigar_op(cig[i]);
                const int len = bam_cigar_oplen(cig[i]);
                if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF
                    || op == BAM_CDEL   || op == BAM_CREF_SKIP) {
                    ref_pos += len;
                }
            }
            int64_t rend = ref_pos - 1;
            // 定位窗口范围（可线性scan，win数量通常不大；若特别大可二分）
            // 简易：遍历重叠窗口
            for (size_t wi = next_flush; wi < windows.size(); ++wi) {
                const int wstart = windows[wi].start;
                const int wend   = windows[wi].end;

                if (wstart > rend) break;      // 后面更靠右，无需再查
                if (wend   < rstart) continue; // 还没到重叠，跳过

                if (rstart > wstart || rend < wend) continue; // 需整段覆盖窗口

                std::string s = clip_read_to_window(b, wstart, wend);
                if (!s.empty()) {
                    // 用“当前窗口实际长度”做 0.9×过滤（末窗可能短于 win_len）
                    const int wlen = wend - wstart + 1;
                    const int min_keep = static_cast<int>(std::ceil(wlen * 0.9));
                    // 如果想“向上取整”，用： const int min_keep = (9 * wlen + 9) / 10;
                    // if ((int)s.size() > wlen * 5) { // 自定义阈值，先用 5× 窗口
                    //     std::string cig = cigar_to_string(b);
                    //     std::fprintf(stderr,
                    //         "[CLIP-LONG] chr1 win=%d-%d len=%zu wlen=%d CIGAR=%s flag=0x%x read_start=%d\n",
                    //         wstart, wend, s.size(), wlen, cig.c_str(),
                    //         b->core.flag, b->core.pos);
                    // }
                    // if ((int)s.size() >= min_keep && (int)s.size() <= max_keep){
                    if ((int)s.size() >= min_keep ){
                        win_reads[wi].push_back(std::move(s));
                    }
                    // else: 截断后太短，丢弃
                }
            }            
        }
        hts_itr_destroy(itr);
        // BAM读完：flush剩余窗口
        while (next_flush < windows.size()){
            RegionReads pack{windows[next_flush].chr, windows[next_flush].index,
                             windows[next_flush].start, windows[next_flush].end,
                             std::move(win_reads[next_flush])};
            q_.push(std::move(pack));
            ++produced_windows;
            if (produced_windows % 1000 == 0) {
                auto now = std::chrono::steady_clock::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - batch_start).count();
                std::fprintf(stderr,
                            "[PRODUCER] batch=1000 total=%lld stage=tail queue=%zu wall_ms=%lld\n",
                            produced_windows, q_.size(), (long long)ms);
                batch_start = now;
            }

            ++next_flush;
        }
    }

    bam_destroy1(b); bam_hdr_destroy(hdr); hts_idx_destroy(idx); sam_close(fp);
    q_.close();
    return true;
}
