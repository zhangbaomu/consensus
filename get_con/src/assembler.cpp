#include "assembler.h"
#include <algorithm>   // std::min
#include <stdexcept>   // std::runtime_error
#include <cassert>

Assembler::Assembler(AsmConfig cfg)
    : cfg_(cfg), fp_(NULL) {
    fp_ = fopen(cfg_.out_fa.c_str(), "w");
    if (!fp_) throw std::runtime_error("Cannot open output FASTA");
}

Assembler::~Assembler() {
    if (fp_) fclose(fp_);
}

void Assembler::register_chr(const std::string& chr, std::vector<Window> windows) {
    std::lock_guard<std::mutex> lk(m_);
    st_[chr].windows = windows;
}

int Assembler::edit_distance(const std::string& a, const std::string& b) {
    const int n = (int)a.size();
    const int m = (int)b.size();
    std::vector<int> dp(m + 1);
    for (int j = 0; j <= m; ++j) dp[j] = j;
    for (int i = 1; i <= n; ++i) {
        int prev = dp[0]++;
        for (int j = 1; j <= m; ++j) {
            int cur = dp[j];
            int cost = (a[i-1] == b[j-1]) ? 0 : 1;
            int del  = dp[j] + 1;
            int ins  = dp[j-1] + 1;
            int sub  = prev + cost;
            dp[j] = std::min(std::min(del, ins), sub);
            prev = cur;
        }
    }
    return dp[m];
}

// 写一段（seg_start_idx..seg_end_idx）的 FASTA
void Assembler::write_segment_(const std::string& chr, ChrState& cs, int32_t seg_start_idx, int32_t seg_end_idx) {
    if (seg_start_idx < 0 || seg_end_idx < seg_start_idx) return;
    if (cs.hap1.empty()) return;

    // 段的参考坐标范围（1-based，闭区间）
    assert(seg_start_idx >= 0 && seg_start_idx < (int)cs.windows.size());
    assert(seg_end_idx   >= 0 && seg_end_idx   < (int)cs.windows.size());
    const int64_t start_1 = cs.windows[seg_start_idx].start;
    const int64_t end_1   = cs.windows[seg_end_idx].end;

    // header 形如：>chr:START-END
    char head1[256];
    std::snprintf(head1, sizeof(head1), "%s:%lld-%lld", chr.c_str(),
                  (long long)start_1, (long long)end_1);

    write_fasta_record_(head1, cs.hap1);

    // 清空当前段缓存
    cs.hap1.clear();
    cs.in_segment = false;
    cs.seg_start_idx = -1;
}

// 写单条 FASTA 记录，60 列换行
void Assembler::write_fasta_record_(const std::string& header, const std::string& seq) {
    std::fprintf(fp_, ">%s\n", header.c_str());
    const size_t L = seq.size();
    for (size_t i = 0; i < L; i += 60) {
        const size_t len = std::min((size_t)60, L - i);
        std::fwrite(seq.data() + i, 1, len, fp_);
        std::fputc('\n', fp_);
    }
}

void Assembler::flush_ready_(ChrState& cs, const std::string& chr) {
    const size_t ov = (size_t)cfg_.overlap;

    while (true) {
        std::map<int32_t, std::pair<std::string,std::string> >::iterator it = cs.pending.find(cs.next);
        if (it == cs.pending.end()) break;

        // 取该窗口的两条共识
        std::string c1 = it->second.first;
        cs.pending.erase(it);

        // 空窗口：立刻断开（如当前段非空则写出），不追加任何碱基
        if (c1.empty()) {
            if (cs.in_segment && !cs.hap1.empty()) {
                write_segment_(chr, cs, cs.seg_start_idx, cs.next - 1);
            }
            cs.next++;
            continue;
        }

        // 有序列：若不在段内，开新段
        if (!cs.in_segment) {
            cs.in_segment = true;
            cs.seg_start_idx = cs.next;
            cs.hap1 = c1;
        } else {
            // 段内：按 50bp 重叠二分配 + 只追加非重叠
            if (c1.size() > ov) cs.hap1.append(c1.substr(ov));
        }

        cs.next++;
    }
}

void Assembler::submit(const std::string& chr, int32_t index, std::string c1, std::string c2) {
    std::lock_guard<std::mutex> lk(m_);
    ChrState& cs = st_[chr];
    cs.pending.insert(std::make_pair(index, std::make_pair(c1, c2)));
    flush_ready_(cs, chr);
}

void Assembler::finalize() {
    std::lock_guard<std::mutex> lk(m_);
    // 逐 chr 处理未提交/未写出的窗口
    for (std::unordered_map<std::string, ChrState>::iterator kv = st_.begin(); kv != st_.end(); ++kv) {
        const std::string& chr = kv->first;
        ChrState& cs = kv->second;

        const size_t ov = (size_t)cfg_.overlap;

        while (cs.next < (int)cs.windows.size()) {
            // 若有该 index 的结果，按 flush_ready_ 同样逻辑处理
            std::map<int32_t, std::pair<std::string,std::string> >::iterator it = cs.pending.find(cs.next);
            if (it != cs.pending.end()) {
                std::string c1 = it->second.first;
                cs.pending.erase(it);

                if (c1.empty()) {
                    if (cs.in_segment && !cs.hap1.empty()) {
                        write_segment_(chr, cs, cs.seg_start_idx, cs.next - 1);
                    }
                    cs.next++;
                    continue;
                }

        if (!cs.in_segment) {
            cs.in_segment = true;
            cs.seg_start_idx = cs.next;
            cs.hap1 = c1;
        } else {
            const std::string h1_tail = (cs.hap1.size() >= ov)
                ? cs.hap1.substr(cs.hap1.size() - ov) : cs.hap1;
            const std::string c1_head = c1.substr(0, std::min(ov, c1.size()));
            const int dist = edit_distance(h1_tail, c1_head);
            const int threshold = std::max(1, (int)(cfg_.overlap * 0.1));
            if (dist > threshold) {
                write_segment_(chr, cs, cs.seg_start_idx, cs.next - 1);
                ++bad_overlap_breaks_;
                cs.in_segment = true;
                cs.seg_start_idx = cs.next;
                cs.hap1 = c1;
            } else {
                if (c1.size() > ov) cs.hap1.append(c1.substr(ov));
            }
        }
                cs.next++;
                continue;
            }

            // 没有该 index 的结果：视为“缺失窗口”，也断开段
            if (cs.in_segment && !cs.hap1.empty()) {
                write_segment_(chr, cs, cs.seg_start_idx, cs.next - 1);
            }
            cs.next++;
        }

        // 最后若仍在段内，写出最后一段
        if (cs.in_segment && !cs.hap1.empty()) {
            write_segment_(chr, cs, cs.seg_start_idx, cs.next - 1);
        }
    }
}
