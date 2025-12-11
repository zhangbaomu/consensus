#include "extract_adapter.h"
#include "extract_utils.h"    // 使用你旧实现提供的 calculate_trim_positions
#include <algorithm>
#include <vector>

// 老工具的命名空间
using namespace extract_utils;

namespace extract_adapter {

static inline char dec_base(const uint8_t* s, int i) {
    // htslib: bam_seqi(s,i) -> 0..15; 用内置表 seq_nt16_str 解码
    static const char* LUT = seq_nt16_str; // "=ACMGRSVTWYHKDBN"
    return LUT[bam_seqi(s, i)];
}

bool extract_window_seq(const bam1_t* rec,
                        int64_t window_start_1,
                        int64_t window_end_1,
                        std::string& out) {
    out.clear();
    if (rec == NULL) return false;
    if (window_end_1 < window_start_1) return false;

    // ---- 1) 参考起点（旧实现里用的是 0-based ref_start）----
    // 在 extract_utils.cpp 里看到：read_start_pos = b->core.pos; 传给 calculate_trim_positions
    const int32_t ref_start0 = rec->core.pos; // 0-based

    // ---- 2) 目标窗口（与旧代码一致：start-1, end 传入）----
    const int32_t target_start0 = (int32_t)window_start_1 - 1; // 0-based
    const int32_t target_end1   = (int32_t)window_end_1;       // 1-based 终点（旧实现就是这么传）

    // ---- 3) 用旧实现计算 query 上的截断区间 [qs, qe) ----
    int32_t qs = 0, qe = 0;
    const uint32_t* cigar = bam_get_cigar(rec);
    const int n_cigar      = rec->core.n_cigar;
    calculate_trim_positions(cigar, n_cigar,
                             ref_start0, target_start0, target_end1,
                             qs, qe);

    // 旧实现中：trimmed_length = read_end - read_start（end 为开区间）
    if (qe <= qs) return false;

    // ---- 4) 从 BAM 里解码 query，按旧逻辑：不做反向互补 ----
    const int lq = rec->core.l_qseq;
    if (qs < 0) qs = 0;
    if (qe > lq) qe = lq;
    if (qe <= qs) return false;

    const uint8_t* s = bam_get_seq(rec);
    out.reserve((size_t)(qe - qs));
    for (int i = qs; i < qe; ++i) {
        out.push_back(dec_base(s, i)); // 直接用 BAM 的 SEQ，保持与旧逻辑一致（不反向、不互补）
    }
    return !out.empty();
}

} // namespace extract_adapter
