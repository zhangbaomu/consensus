#include "windowing.h"
#include <algorithm>

std::vector<Window> build_windows_for_chr(const std::string& chr, int64_t chr_len,
                                          int32_t win_len, int32_t overlap) {
    int32_t step = win_len - overlap;
    std::vector<Window> v; v.reserve((chr_len + step - 1) / step + 2);
    int64_t s = 1; int32_t idx = 0;
    while (s <= chr_len) {
        int64_t e = std::min<int64_t>(s + win_len - 1, chr_len);
        v.push_back(Window{chr, s, e, idx++});
        if (e == chr_len) break;
        s += step;
    }
    return v;
}