#include "truth_eval.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <htslib/faidx.h>
#include <htslib/hts.h>
#if defined(__has_include)
#  if __has_include(<htslib/version.h>)
#    include <htslib/version.h>
#    define TRUTH_EVAL_HAS_HTS_VERSION 1
#  endif
#endif

namespace {

inline char upper_base(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

std::string to_upper_copy(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), upper_base);
    return out;
}

} // namespace

#ifndef KS_SEP_LINE
#define KS_SEP_LINE '\n'
#endif

TruthEvaluator::TruthEvaluator(const TruthEvalConfig& cfg)
    : cfg_(cfg) {
    if (cfg_.ref_fasta.empty() || cfg_.vcf_path.empty()) {
        std::fprintf(stderr, "[EVAL] ref_fasta or vcf_path not provided, evaluation disabled.\n");
        return;
    }

    faidx_ = fai_load(cfg_.ref_fasta.c_str());
    if (!faidx_) {
        std::fprintf(stderr, "[EVAL] failed to open reference fasta: %s\n", cfg_.ref_fasta.c_str());
        return;
    }
    if (!load_variants()) {
        std::fprintf(stderr, "[EVAL] failed to load VCF variants: %s\n", cfg_.vcf_path.c_str());
        return;
    }
    ready_ = true;
}

TruthEvaluator::~TruthEvaluator() {
    if (faidx_) {
        fai_destroy(static_cast<faidx_t*>(faidx_));
        faidx_ = nullptr;
    }
}

bool TruthEvaluator::load_variants() {
    htsFile* fp = hts_open(cfg_.vcf_path.c_str(), "r");
    if (!fp) {
        return false;
    }
    kstring_t line{0, 0, nullptr};
    size_t kept = 0, skipped = 0;

    while (hts_getline(fp, KS_SEP_LINE, &line) >= 0) {
        if (!line.s || line.s[0] == '#') continue;
        std::string line_str(line.s, line.l);
        std::vector<std::string> cols;
        cols.reserve(12);
        size_t start = 0;
        while (start <= line_str.size()) {
            size_t tab = line_str.find('\t', start);
            if (tab == std::string::npos) tab = line_str.size();
            cols.emplace_back(line_str.substr(start, tab - start));
            start = tab + 1;
        }
        if (cols.size() < 10) {
            skipped++;
            continue;
        }
        const std::string& chrom = cols[0];
        const std::string& pos_s = cols[1];
        const std::string& ref = cols[3];
        const std::string& alt_s = cols[4];
        const std::string& format = cols[8];
        const std::string& sample = cols[9];

        int32_t pos1 = 0;
        try {
            pos1 = std::stoi(pos_s);
        } catch (...) {
            skipped++;
            continue;
        }
        if (pos1 <= 0) {
            skipped++;
            continue;
        }

        int gt_idx = -1;
        {
            size_t i = 0, j = 0;
            int idx = 0;
            while (j <= format.size()) {
                if (j == format.size() || format[j] == ':') {
                    if (format.compare(i, j - i, "GT") == 0) {
                        gt_idx = idx;
                        break;
                    }
                    idx++;
                    i = j + 1;
                }
                ++j;
            }
        }
        if (gt_idx < 0) {
            skipped++;
            continue;
        }

        std::string gt;
        {
            size_t i = 0, j = 0;
            int idx = 0;
            while (j <= sample.size()) {
                if (j == sample.size() || sample[j] == ':') {
                    if (idx == gt_idx) {
                        gt = sample.substr(i, j - i);
                        break;
                    }
                    idx++;
                    i = j + 1;
                }
                ++j;
            }
        }
        if (gt.empty()) {
            skipped++;
            continue;
        }

        int allele1 = 0, allele2 = 0;
        if (!parse_gt(gt, allele1, allele2)) {
            skipped++;
            continue;
        }

        std::vector<std::string> alts;
        {
            size_t i = 0, j = 0;
            while (j <= alt_s.size()) {
                if (j == alt_s.size() || alt_s[j] == ',') {
                    alts.emplace_back(alt_s.substr(i, j - i));
                    i = j + 1;
                }
                ++j;
            }
        }

        auto allele_to_seq = [&](int allele) -> std::string {
            if (allele <= 0) return ref;
            if (allele - 1 < (int)alts.size()) {
                std::string alt = alts[allele - 1];
                if (alt == "*") alt.clear();
                return alt;
            }
            return ref;
        };

        std::string alt1 = allele_to_seq(allele1);
        std::string alt2 = allele_to_seq(allele2);
        if (alt1 == ref && alt2 == ref) {
            // 同型参照，跳过
            skipped++;
            continue;
        }

        VariantEntry ve;
        ve.pos_1based = pos1;
        ve.ref = normalize_seq(ref);
        ve.alt[0] = normalize_seq(alt1);
        ve.alt[1] = normalize_seq(alt2);
        variants_[chrom].push_back(std::move(ve));
        kept++;
    }

    if (line.s) free(line.s);
    hts_close(fp);

    for (auto& kv : variants_) {
        auto& vec = kv.second;
        std::sort(vec.begin(), vec.end(), [](const VariantEntry& a, const VariantEntry& b) {
            return a.pos_1based < b.pos_1based;
        });
    }

    std::fprintf(stderr, "[EVAL] VCF loaded: kept=%zu skipped=%zu chroms=%zu\n",
                 kept, skipped, variants_.size());
    return kept > 0;
}

bool TruthEvaluator::parse_gt(const std::string& gt, int& allele1, int& allele2) {
    allele1 = allele2 = 0;
    if (gt.empty() || gt[0] == '.') return false;

    auto parse_int = [](const std::string& s) -> int {
        try {
            return std::stoi(s);
        } catch (...) {
            return -1;
        }
    };

    size_t bar = gt.find('|');
    if (bar != std::string::npos) {
        allele1 = parse_int(gt.substr(0, bar));
        allele2 = parse_int(gt.substr(bar + 1));
        if (allele1 < 0 || allele2 < 0) return false;
        return true;
    }

    size_t slash = gt.find('/');
    if (slash != std::string::npos) {
        allele1 = parse_int(gt.substr(0, slash));
        allele2 = parse_int(gt.substr(slash + 1));
        if (allele1 < 0 || allele2 < 0) return false;
        return true;
    }

    int val = parse_int(gt);
    if (val < 0) return false;
    allele1 = allele2 = val;
    return true;
}

std::string TruthEvaluator::normalize_seq(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), upper_base);
    return out;
}

bool TruthEvaluator::fetch_window(const std::string& chr, int64_t start_1based, int64_t end_1based,
                                  std::string& truth_out) const {
    if (!ready_ || !faidx_) return false;
    if (start_1based > end_1based) return false;

    int64_t len = 0;
    std::string ref_seq;
    {
        std::lock_guard<std::mutex> lk(fetch_mutex_);
        char* seq = nullptr;
#if defined(TRUTH_EVAL_HAS_HTS_VERSION)
#  if HTS_VERSION_GE(1,10)
        seq = faidx_fetch_seq64(static_cast<faidx_t*>(faidx_),
                                 chr.c_str(), start_1based - 1, end_1based - 1, &len);
#  else
        int len32 = 0;
        seq = faidx_fetch_seq(static_cast<faidx_t*>(faidx_),
                              chr.c_str(), start_1based - 1, end_1based - 1, &len32);
        len = len32;
#  endif
#else
        int len32 = 0;
        seq = faidx_fetch_seq(static_cast<faidx_t*>(faidx_),
                              chr.c_str(), start_1based - 1, end_1based - 1, &len32);
        len = len32;
#endif
        if (!seq || len <= 0) {
            if (seq) free(seq);
            return false;
        }
        ref_seq.assign(seq, seq + len);
        free(seq);
    }

    ref_seq = to_upper_copy(ref_seq);
    auto it = variants_.find(chr);
    if (it != variants_.end()) {
        apply_variants_to_hap(it->second, start_1based, end_1based, ref_seq, 1); // hap2 (spike-in)
    }
    truth_out.swap(ref_seq);
    return true;
}

void TruthEvaluator::apply_variants_to_hap(const std::vector<VariantEntry>& vars,
                                           int64_t win_start, int64_t win_end,
                                           std::string& seq, int hap_index) {
    if (vars.empty()) return;
    auto lower = std::lower_bound(vars.begin(), vars.end(), win_start,
                                  [](const VariantEntry& v, int64_t s) {
                                      return v.pos_1based < s;
                                  });

    long long shift = 0;
    long long last_l = -1, last_r = -1;

    for (auto it = lower; it != vars.end(); ++it) {
        const VariantEntry& var = *it;
        if (var.pos_1based > win_end) break;

        const std::string& alt_full = var.alt[hap_index];
        if (alt_full == var.ref) continue; // 等位基因与 REF 相同，不需要替换

        std::string ref_clip = var.ref;
        std::string alt_clip = alt_full;

        int64_t ref_l = var.pos_1based;
        int64_t ref_r = var.pos_1based + (int64_t)var.ref.size() - 1;

        int64_t ov_l = std::max<int64_t>(ref_l, win_start);
        int64_t ov_r = std::min<int64_t>(ref_r, win_end);
        if (ov_l > ov_r) {
            // 插入且窗口未覆盖到REF anchor，无法处理
            continue;
        }

        int left_clip = (int)(ov_l - ref_l);
        int right_clip = (int)(ref_r - ov_r);

        if (left_clip > 0) {
            if (left_clip < (int)ref_clip.size()) ref_clip.erase(0, left_clip);
            else ref_clip.clear();
            if (left_clip < (int)alt_clip.size()) alt_clip.erase(0, left_clip);
            else alt_clip.clear();
        }
        if (right_clip > 0) {
            if (right_clip < (int)ref_clip.size()) ref_clip.erase(ref_clip.size() - right_clip);
            else ref_clip.clear();
            if (right_clip < (int)alt_clip.size()) alt_clip.erase(alt_clip.size() - right_clip);
            else alt_clip.clear();
        }

        if (ref_clip.empty() && alt_clip.empty()) continue;

        long long idx = (long long)(ov_l - win_start) + shift;
        if (idx < 0 || idx > (long long)seq.size()) continue;
        if (!(last_r <= idx || idx <= last_l)) continue;
        if ((size_t)idx + ref_clip.size() > seq.size()) continue;

        std::string cur = seq.substr((size_t)idx, ref_clip.size());
        if (to_upper_copy(cur) != to_upper_copy(ref_clip)) {
            continue;
        }

        seq.replace((size_t)idx, ref_clip.size(), alt_clip);
        long long delta = (long long)alt_clip.size() - (long long)ref_clip.size();
        shift += delta;
        last_l = idx;
        last_r = idx + (long long)alt_clip.size();
    }
}

int TruthEvaluator::edit_distance(const std::string& truth, const std::string& consensus) const {
    if (!ready_) return -1;
    if (truth.empty()) return (int)consensus.size();
    if (consensus.empty()) return (int)truth.size();
    if ((size_t)truth.size() > cfg_.max_eval_bases || (size_t)consensus.size() > cfg_.max_eval_bases) {
        return -1;
    }
    return myers_edit_distance(truth, consensus);
}

int TruthEvaluator::myers_edit_distance(const std::string& a, const std::string& b) const {
    const int n = (int)a.size();
    const int m = (int)b.size();
    if (n == 0) return m;
    if (m == 0) return n;

    const int maxd = n + m;
    std::vector<int> v(2 * maxd + 3, -1);
    const int offset = maxd + 1;
    v[offset + 1] = 0;

    for (int d = 0; d <= maxd; ++d) {
        for (int k = -d; k <= d; k += 2) {
            int idx = offset + k;
            int x;
            if (k == -d || (k != d && v[idx - 1] < v[idx + 1])) {
                x = v[idx + 1];
            } else {
                x = v[idx - 1] + 1;
            }
            int y = x - k;
            while (x < n && y < m && upper_base(a[x]) == upper_base(b[y])) {
                ++x; ++y;
            }
            v[idx] = x;
            if (x >= n && y >= m) {
                return d;
            }
        }
    }
    return maxd;
}
