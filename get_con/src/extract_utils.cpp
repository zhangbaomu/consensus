#include "../include/extract_utils.h"
#include <htslib/sam.h>
#include <htslib/hts.h>
#include <htslib/faidx.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <set>
#include <sys/stat.h>
#include "../include/json.hpp"
#include "../include/poa_utils.h"
#include <chrono>           // 添加这行
#include <sstream>          // 添加这行
#include <cstdio>           // 添加这行（为了popen/pclose）
#include <cctype>           // 添加这行（为了toupper）
#include <ctime>            // 添加这行
#include <unordered_map>    // 添加这行
#include <unordered_set>    // 添加这行
#include <htslib/kstring.h> // 供 hts_getline 使用
#include <cmath>  // 为 std::ceil


#ifndef KS_SEP_LINE
#define KS_SEP_LINE '\n'      // 兼容老版本 htslib，按换行分隔读取
#endif

namespace json_lib = nlohmann;

namespace extract_utils
{
    // 将 BAM 里的 4-bit 编码序列转成 ACGTN 字符串（方向与 BAM 一致）
    static std::string bam_seq_to_string(const bam1_t *b)
    {
        const uint8_t *s = bam_get_seq(b);
        int len = b->core.l_qseq;
        std::string out;
        out.resize(len);
        // htslib 提供的表，1:A, 2:C, 4:G, 8:T, 15:N，其它按 N 兜底
        for (int i = 0; i < len; ++i)
        {
            uint8_t code = bam_seqi(s, i) & 0xF;
            char base;
            switch (code)
            {
            case 1:
                base = 'A';
                break;
            case 2:
                base = 'C';
                break;
            case 4:
                base = 'G';
                break;
            case 8:
                base = 'T';
                break;
            case 15:
                base = 'N';
                break;
            default:
                base = 'N';
                break;
            }
            out[i] = base;
        }
        return out;
    }

    // 将 BAM 的 Phred 质量转换为 FASTQ 字符串（方向与 BAM 一致；反链时已被反转）
    static std::string bam_qual_to_string(const bam1_t *b)
    {
        const uint8_t *q = bam_get_qual(b);
        int len = b->core.l_qseq;
        std::string out;
        out.resize(len);
        if (len == 0 || q == nullptr)
            return out;
        // 若无质量，htslib 可能给 0xFF；此时返回同长度的 'I'(40) 兜底或空串均可
        if (q[0] == 0xFF)
        {
            std::fill(out.begin(), out.end(), 'I');
            return out;
        }
        for (int i = 0; i < len; ++i)
            out[i] = static_cast<char>(q[i] + 33);
        return out;
    }

    // RegionInfo 构造函数
    RegionInfo::RegionInfo(const std::string &region_str, const std::string &base_output_dir)
    {
        this->region_str = region_str;

        // 解析区域
        if (!parse_region(region_str.c_str(), chrom, start, end))
        {
            throw std::runtime_error("Invalid region format: " + region_str);
        }

        // 创建安全的文件名
        safe_name = region_str;
        std::replace(safe_name.begin(), safe_name.end(), ':', '_');
        std::replace(safe_name.begin(), safe_name.end(), '-', '_');

        // 设置输出目录
        output_dir = base_output_dir + "/" + safe_name;
    }

    // BatchRegionProcessor 实现
    BatchRegionProcessor::BatchRegionProcessor(const std::string &base_output_dir, const std::string &ref_fasta)
        : base_output_dir(base_output_dir), reference_fasta_file(ref_fasta) {}

    // bool BatchRegionProcessor::load_regions_from_file(const std::string& regions_file) {
    //     std::ifstream file(regions_file);
    //     if (!file.is_open()) {
    //         std::cerr << "Error: Cannot open regions file: " << regions_file << std::endl;
    //         return false;
    //     }

    //     std::string line;
    //     int line_number = 0;

    //     while (std::getline(file, line)) {
    //         line_number++;

    //         // 移除前后空白字符
    //         line.erase(0, line.find_first_not_of(" \t\r\n"));
    //         line.erase(line.find_last_not_of(" \t\r\n") + 1);

    //         // 跳过空行和注释行
    //         if (line.empty() || line[0] == '#') {
    //             continue;
    //         }

    //         // 验证区域格式
    //         if (line.find(':') == std::string::npos || line.find('-') == std::string::npos) {
    //             std::cerr << "Warning: Invalid region format at line " << line_number
    //                       << ": " << line << std::endl;
    //             continue;
    //         }

    //         try {
    //             RegionInfo region(line, base_output_dir);
    //             regions.push_back(region);

    //             // 创建区域输出目录
    //             struct stat st = {0};
    //             if (stat(region.output_dir.c_str(), &st) == -1) {
    //                 if (mkdir(region.output_dir.c_str(), 0755) == -1) {
    //                     std::cerr << "Warning: Failed to create directory: " << region.output_dir << std::endl;
    //                 }
    //             }

    //         } catch (const std::exception& e) {
    //             std::cerr << "Error parsing region at line " << line_number << ": " << e.what() << std::endl;
    //             continue;
    //         }
    //     }

    //     file.close();

    //     std::cout << "Loaded " << regions.size() << " valid regions" << std::endl;
    //     for (const auto& region : regions) {
    //         std::cout << "  - " << region.region_str << " -> " << region.safe_name << std::endl;
    //     }

    //     return !regions.empty();
    // }

    bool BatchRegionProcessor::load_regions_from_file(const std::string &regions_file)
    {
        const size_t kMaxRegions = 7000; // 只读前个有效区域10000

        std::ifstream file(regions_file);
        if (!file.is_open())
        {
            std::cerr << "Error: Cannot open regions file: " << regions_file << std::endl;
            return false;
        }

        std::string line;
        size_t line_number = 0;
        size_t valid_count = 0;

        while (std::getline(file, line))
        {
            if (valid_count >= kMaxRegions)
            {
                // 已经读够 1 万个有效区域，提前结束
                break;
            }

            line_number++;
            // 去掉前后空白
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            if (!line.empty())
                line.erase(line.find_last_not_of(" \t\r\n") + 1);

            // 跳过空行和注释
            if (line.empty() || line[0] == '#')
                continue;

            // 先做个简单格式检查
            if (line.find(':') == std::string::npos || line.find('-') == std::string::npos)
            {
                std::cerr << "Warning: Invalid region format at line " << line_number
                          << ": " << line << std::endl;
                continue;
            }

            try
            {
                RegionInfo region(line, base_output_dir);

                region.use_ref_in_poa = (valid_count < 5000);
                const char* group = region.use_ref_in_poa ? "with_ref_poa" : "wo_ref_poa";
                region.output_dir = base_output_dir + "/" + std::string(group) + "/" + region.safe_name;
        
                regions.push_back(region);
                valid_count++;

                // 为该区域创建输出目录
                struct stat st = {0};
                if (stat((base_output_dir + "/" + group).c_str(), &st) == -1) mkdir((base_output_dir + "/" + group).c_str(), 0755);
                if (stat(region.output_dir.c_str(), &st) == -1) mkdir(region.output_dir.c_str(), 0755);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error parsing region at line " << line_number << ": " << e.what() << std::endl;
                continue;
            }
        }

        // 如果文件里还有剩余但我们只取了前 1 万个，提示一下
        if (file && !file.eof())
        {
            std::cerr << "Info: Reached the 10,000 valid region limit. Remaining regions in file are ignored." << std::endl;
        }

        file.close();

        std::cout << "Loaded " << regions.size() << " valid regions (capped at 10,000)" << std::endl;

        // 输出区域名字
        //  for (const auto& region : regions) {
        //      std::cout << "  - " << region.region_str << " -> " << region.safe_name << std::endl;
        //  }

        return !regions.empty();
    }

    bool BatchRegionProcessor::extract_all_reads_from_bam(const char *bam_file)
    {
        std::cout << "Opening BAM file: " << bam_file << std::endl;

        samFile *in = sam_open(bam_file, "r");
        if (!in)
        {
            std::cerr << "Failed to open BAM file: " << bam_file << std::endl;
            return false;
        }

        bam_hdr_t *header = sam_hdr_read(in);
        if (!header)
        {
            std::cerr << "Failed to read BAM header" << std::endl;
            sam_close(in);
            return false;
        }

        bam1_t *b = bam_init1();

        // 预处理区域信息，建立染色体索引和排序优化查找
        std::unordered_map<std::string, std::vector<RegionInfo *>> chrom_regions;
        for (auto &region : regions)
        {
            chrom_regions[region.chrom].push_back(&region);
        }

        // 对每个染色体的区域按起始位置排序，便于快速查找
        for (auto &chrom_pair : chrom_regions)
        {
            std::sort(chrom_pair.second.begin(), chrom_pair.second.end(),
                      [](const RegionInfo *a, const RegionInfo *b)
                      {
                          return a->start < b->start;
                      });
        }

        std::cout << "Starting optimized single-pass BAM traversal..." << std::endl;

        int total_reads_processed = 0;
        int mapped_reads = 0;
        int matched_reads = 0;
        int quality_filtered_reads = 0; // 质量过滤的read计数
        int error_read=0;

        // 优化：减少字符串构造
        std::string current_chrom;
        std::string qname;
        qname.reserve(256); // 预分配空间

        // 关键：只遍历BAM文件一次，不使用区域特定的迭代器
        while (sam_read1(in, header, b) >= 0)
        {
            total_reads_processed++;

            // 跳过未比对的reads
            if (b->core.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY))
                continue;
            //跳过反链
            // if (b->core.flag & BAM_FREVERSE) continue;
            bool is_rev = (b->core.flag & BAM_FREVERSE) != 0;
            mapped_reads++;

            // 9/13 只选主要比对

            // 质量过滤 - 跳过低质量的reads
            uint8_t mapq = b->core.qual;
            // 0先不跳过
            if (mapq < min_mapping_quality)
            {
                quality_filtered_reads++;
                continue;
            }

            // 优化：减少染色体名称的字符串构造
            // 优化：减少染色体名称的字符串构造
            const char *chrom_cstr = nullptr;

            // 兼容不同版本的htslib
            if (b->core.tid >= 0 && b->core.tid < header->n_targets)
            {
                chrom_cstr = header->target_name[b->core.tid];
            }

            if (!chrom_cstr)
                continue;

            // 跳过非 chr1 的 reads
            std::string target_chrom = "chr1";  // 只处理 chr1
            if (std::string(chrom_cstr) != "chr1" && std::string(chrom_cstr) != "chr2"){
                break;  // 只有 chr1 的 reads 才处理
            }

            // 只在染色体改变时更新字符串
            if (current_chrom != chrom_cstr)
            {
                current_chrom = chrom_cstr;
            }

            // 优化：快速检查该染色体是否有感兴趣的区域
            // 9.14注释，因为每条染色体必有区域
            auto chrom_it = chrom_regions.find(current_chrom);
            if (chrom_it == chrom_regions.end())
                continue;

            // 获取read的基本信息
            int32_t read_start_pos = b->core.pos;
            int32_t read_end_pos = bam_endpos(b);

            // 优化：使用二分查找找到可能重叠的区域范围
            const auto &regions_vec = chrom_it->second;

            // 找到第一个可能重叠的区域（end > read_start）
            auto lower = std::lower_bound(regions_vec.begin(), regions_vec.end(), read_start_pos,
                                          [](const RegionInfo *region, int32_t pos)
                                          {
                                              return region->end <= pos;
                                          });

            bool found_overlap = false;

            // 从第一个可能的区域开始检查
            for (auto it = lower; it != regions_vec.end(); ++it)
            {
                const RegionInfo *region = *it;

                // 如果区域起始位置已经超过read结束位置，后续区域也不会重叠
                if (region->start >= read_end_pos)
                    break;

                // 检查是否真正重叠
                if (read_end_pos > region->start && read_start_pos < region->end)
                {
                    // 🔥 新增：检查read是否需要截断（两端都在区域外）
                    bool need_left_trim = read_start_pos <= region->start;
                    bool need_right_trim = read_end_pos >= region->end;
                    // 只保留需要两端截断的reads
                    if (!(need_left_trim && need_right_trim))
                    {
                        continue; // 跳过不需要两端截断的reads
                    }
                    // 存在重叠
                    if (!found_overlap)
                    {
                        // 优化：只在确实有重叠时才构造qname字符串
                        // 多个重叠也只构造一次
                        qname.assign(bam_get_qname(b));
                        // double base_quality = calculate_base_quality_from_bam(b);
                        // all_read_base_qualities[qname] = base_quality;

                        found_overlap = true;
                    }

                    // 记录read信息
                    ReadInfo info;
                    info.qname = qname;
                    info.start_pos = read_start_pos;
                    info.end_pos = read_end_pos;
                    info.original_length = b->core.l_qseq; // 更直接可信
                    info.mapping_quality = mapq;           // 保存mapping quality

                    // 新增：从预存储的质量分数中获取
                    // auto qual_it = all_read_base_qualities.find(qname);
                    // if (qual_it != all_read_base_qualities.end()) {
                    //     info.base_quality_score = qual_it->second;
                    // }
                    // 计算截断位置

                    info.need_trim = true;
                    calculate_trim_positions(bam_get_cigar(b), b->core.n_cigar,
                                             read_start_pos, region->start - 1, region->end,
                                             info.read_start, info.read_end);

                    int trimmed_length = info.read_end - info.read_start;
                    // 记录链向：0 正链，1 反链
                    // 按当前区域长度动态计算最小保留阈值（90% × (region->end - region->start)）
                    int region_len = static_cast<int>(region->end - region->start);
                    int min_keep   = std::max(1, static_cast<int>(std::ceil(0.9 * region_len)));

                    if (trimmed_length < min_keep)
                    { // 跳过截断后太短的序列（随区域长度自动适配）
                        continue;
                    }
                    
                    // 在这里添加 count_errors_in_window 的检查
                    // if (count_errors_in_window(b, info.read_start, info.read_end) >= 50) {
                    //     error_read++;
                    //     continue;  // 如果前100bp的错误数超过5个，则丢弃该read
                    // }
                    
                    bool is_rev = (b->core.flag & BAM_FREVERSE) != 0;
                    info.flag = is_rev ? 1 : 0;

                    // 统一为“原始 read 正向坐标系”
                    // L 直接用 minimap BAM 的 read 长度（不是 Dorado mv）
                    if (is_rev) {
                        const int L = b->core.l_qseq;
                        const int s = info.read_start;
                        const int e = info.read_end;    // 右开
                        int raw_start = std::max(0, L - e);
                        int raw_end   = std::max(raw_start, L - s);
                        info.raw_read_start = raw_start;
                        info.raw_read_end   = raw_end;
                    }
                    else{
                        // ★ 正链也要写 raw_*（与 read_* 等值），统一“原始 read 正向坐标系”
                        const int L = b->core.l_qseq;
                        info.raw_read_start = std::max(0, std::min(info.read_start, L));
                        info.raw_read_end   = std::max(info.raw_read_start, std::min(info.read_end, L));
                    }
                    // {
                    //     std::string oriseq = bam_seq_to_string(b); // 临时取序列，不改任何状态
                    //     std::cout << "[DBG] read_id="   << qname
                    //               << " orilen="         << oriseq.size()
                    //               << " raw_start="      << info.raw_read_start
                    //               << " raw_end="        << info.raw_read_end
                    //               << " read_start="     << info.read_start
                    //               << " read_end="       << info.read_end << '\n'
                    //               << "[DBG] oriseq="    << oriseq << '\n';
                    // }
                    
                    // ✅ 新增：把该 read 的 SEQ / QUAL 存到 all_fastq_reads / all_fastq_quals
                    // 只在第一次遇到该 read 时填充，避免重复构造
                    if (all_fastq_reads.find(qname) == all_fastq_reads.end())
                    {
                        std::string seq = bam_seq_to_string(b);
                        std::string qual = bam_qual_to_string(b);
                        all_fastq_reads.emplace(qname, std::move(seq));
                        all_fastq_quals.emplace(qname, std::move(qual));
                    }

                    // 存储到对应区域（保持你原有的数据结构不变）
                    region_reads[region->safe_name][qname] = info;
                    all_needed_read_ids.insert(qname);
                }
            }

            if (found_overlap)
            {
                matched_reads++;
            }

            // 进度显示
            if (total_reads_processed % 500000 == 0)
            {
                std::cout << "  Processed " << total_reads_processed
                          << " reads (mapped: " << mapped_reads
                          << ", matched: " << matched_reads << ")" << std::endl;
            }
        }

        bam_destroy1(b);
        bam_hdr_destroy(header);
        sam_close(in);

        std::cout << "Finished scanning BAM:" << std::endl;
        std::cout << "  Total reads: " << total_reads_processed << std::endl;
        std::cout << "  Mapped reads: " << mapped_reads << std::endl;
        std::cout << "  Matched reads: " << matched_reads << std::endl;
        std::cout << "  Error reads: " << error_read << std::endl;
        std::cout << "  Unique reads needed: " << all_needed_read_ids.size() << std::endl;


        // // 对每个区域进行read数量限制
        // filter_reads_by_count_and_quality();

        // 输出每个区域的统计信息
        std::cout << "\nRegion-wise read counts:" << std::endl;
        int total_region_reads = 0;
        for (const auto &region : regions)
        {
            auto it = region_reads.find(region.safe_name);
            int read_count = (it != region_reads.end()) ? it->second.size() : 0;
            total_region_reads += read_count;
            // std::cout << "  " << region.region_str << ": " << read_count << " reads" << std::endl;
        }

        std::cout << "Total region-read pairs: " << total_region_reads << std::endl;

        return true;
    }

    void BatchRegionProcessor::filter_reads_by_count_and_quality()
    {
        std::cout << "\nApplying read count and quality filtering..." << std::endl;

        int total_regions_filtered = 0;
        int total_reads_removed = 0;

        for (auto &region_pair : region_reads)
        {
            const std::string &region_name = region_pair.first;
            auto &reads_map = region_pair.second;

            int original_count = reads_map.size();

            // 如果read数量超过阈值，进行过滤
            if (original_count > max_reads_per_region)
            {
                std::cout << "  Region " << region_name << ": " << original_count
                          << " reads -> filtering to " << max_reads_per_region << std::endl;

                // 将reads转换为vector以便排序
                std::vector<std::pair<std::string, ReadInfo>> reads_vec;
                reads_vec.reserve(original_count);

                for (const auto &read_pair : reads_map)
                {
                    reads_vec.push_back(read_pair);
                }

                // 按多个质量指标排序
                std::sort(reads_vec.begin(), reads_vec.end(),
                          [](const std::pair<std::string, ReadInfo> &a,
                             const std::pair<std::string, ReadInfo> &b)
                          {
                              // 1. 首先按碱基平均质量分数降序排序
                              if (std::abs(a.second.base_quality_score - b.second.base_quality_score) > 0.5)
                              {
                                  return a.second.base_quality_score > b.second.base_quality_score;
                              }
                              // 2. 碱基质量相近时，按mapping quality降序
                              if (a.second.mapping_quality != b.second.mapping_quality)
                              {
                                  return a.second.mapping_quality > b.second.mapping_quality;
                              }
                              // 3. 最后按read长度降序（更长的read更有价值）
                              return a.second.original_length > b.second.original_length;
                          });

                // 清空原来的map并重新填充（只保留前max_reads_per_region个）
                reads_map.clear();

                for (int i = 0; i < std::min(max_reads_per_region, (int)reads_vec.size()); ++i)
                {
                    reads_map[reads_vec[i].first] = reads_vec[i].second;
                }

                // 从all_needed_read_ids中移除被过滤掉的reads
                for (int i = max_reads_per_region; i < (int)reads_vec.size(); ++i)
                {
                    all_needed_read_ids.erase(reads_vec[i].first);
                }

                int filtered_count = reads_map.size();
                int removed_count = original_count - filtered_count;

                total_regions_filtered++;
                total_reads_removed += removed_count;

                // 输出质量统计信息
                if (!reads_vec.empty())
                {
                    double best_base_qual = reads_vec[0].second.base_quality_score;
                    double worst_kept_qual = reads_vec[std::min(max_reads_per_region - 1, (int)reads_vec.size() - 1)].second.base_quality_score;
                    int best_mapq = reads_vec[0].second.mapping_quality;
                    int worst_kept_mapq = reads_vec[std::min(max_reads_per_region - 1, (int)reads_vec.size() - 1)].second.mapping_quality;

                    std::cout << "    Base quality range: " << std::fixed << std::setprecision(2)
                              << worst_kept_qual << " - " << best_base_qual << std::endl;
                    std::cout << "    Mapping quality range: " << worst_kept_mapq << " - " << best_mapq << std::endl;
                }

                std::cout << "    Removed " << removed_count << " lower-quality reads" << std::endl;
            }
        }

        std::cout << "Filtering summary:" << std::endl;
        std::cout << "  Regions filtered: " << total_regions_filtered << std::endl;
        std::cout << "  Total reads removed: " << total_reads_removed << std::endl;
        std::cout << "  Final unique reads needed: " << all_needed_read_ids.size() << std::endl;
    }

    bool BatchRegionProcessor::load_all_fastq_reads(const char *fastq_file)
    {
        std::cout << "Loading FASTQ reads from: " << fastq_file << std::endl;

        std::ifstream fin(fastq_file);
        if (!fin.is_open())
        {
            std::cerr << "Failed to open FASTQ file: " << fastq_file << std::endl;
            return false;
        }

        std::string line, header, seq, plus, qual;
        int total_reads = 0;
        int loaded_reads = 0;

        while (std::getline(fin, header))
        {
            if (!std::getline(fin, seq))
                break;
            if (!std::getline(fin, plus))
                break;
            if (!std::getline(fin, qual))
                break;

            total_reads++;

            // 提取read ID
            std::string read_id = header.substr(1); // 跳过 '@'
            auto pos = read_id.find(' ');
            if (pos != std::string::npos)
            {
                read_id = read_id.substr(0, pos);
            }

            // 只加载需要的reads
            if (all_needed_read_ids.find(read_id) != all_needed_read_ids.end())
            {
                all_fastq_reads[read_id] = seq;
                all_fastq_quals[read_id] = qual;
                loaded_reads++;
            }

            // 进度显示
            if (total_reads % 100000 == 0)
            {
                std::cout << "  Processed " << total_reads << " reads, loaded " << loaded_reads << std::endl;
            }
        }

        std::cout << "Loaded " << loaded_reads << "/" << all_needed_read_ids.size()
                  << " needed reads from " << total_reads << " total reads" << std::endl;

        // 检查是否有缺失的reads
        std::set<std::string> missing_reads;
        for (const auto &read_id : all_needed_read_ids)
        {
            if (all_fastq_reads.find(read_id) == all_fastq_reads.end())
            {
                missing_reads.insert(read_id);
            }
        }

        if (!missing_reads.empty())
        {
            std::cout << "Warning: " << missing_reads.size() << " reads not found in FASTQ" << std::endl;
            if (missing_reads.size() <= 10)
            {
                for (const auto &read_id : missing_reads)
                {
                    std::cout << "  Missing: " << read_id << std::endl;
                }
            }
        }

        return true;
    }

    bool BatchRegionProcessor::process_all_regions()
    {
        int successful_regions = 0;
        int total_regions = regions.size();

        for (size_t i = 0; i < regions.size(); ++i)
        {
            const auto &region = regions[i];

            // std::cout << "\n--- Processing region " << (i + 1) << "/" << total_regions
            //           << ": " << region.region_str << " ---" << std::endl;

            // 获取该区域的reads
            auto region_it = region_reads.find(region.safe_name);
            if (region_it == region_reads.end() || region_it->second.empty())
            {
                std::cout << "No reads found for region " << region.region_str << ", skipping..." << std::endl;
                continue;
            }

            const auto &reads_map = region_it->second;
            // std::cout << "Processing " << reads_map.size() << " reads..." << std::endl;

            // 生成输出文件路径
            std::string output_fastq = region.output_dir + "/" + region.safe_name + "_reads.fastq";
            std::string gfa_output = region.output_dir + "/" + region.safe_name + "_graph.gfa";
            std::string truncation_file = region.output_dir + "/" + region.safe_name + "_truncation_info.json";
            std::string groundtruth_file = region.output_dir + "/" + region.safe_name + "_groundtruth.json";

            // 🔥 新增：获取参考基因组序列
            // std::string reference_sequence = get_reference_sequence(region);
            std::string reference_sequence = get_reference_sequence_with_spikein(region);
            bool has_reference = !reference_sequence.empty();

            // 写入截断后的FASTQ文件
            std::ofstream fout(output_fastq);
            if (!fout.is_open())
            {
                std::cerr << "Failed to create output FASTQ: " << output_fastq << std::endl;
                continue;
            }

            std::vector<std::string> sequences;
            int trimmed_count = 0;

            // 先处理普通的reads
            for (const auto &read_pair : reads_map)
            {
                const std::string &read_id = read_pair.first;
                const ReadInfo &info = read_pair.second;

                auto fastq_it = all_fastq_reads.find(read_id);
                if (fastq_it == all_fastq_reads.end())
                {
                    continue;
                }

                const std::string &original_seq = fastq_it->second;
                const std::string &original_qual = all_fastq_quals[read_id];

                // // 写 FASTQ 截断使用的序列
                // // const std::string& original_seq = /* 你当前取的那份 SEQ */;
                // {
                //     const int seq_len = static_cast<int>(original_seq.size());
                //     std::cout << "[DBG] read_id=" << read_id
                //               << " original_len=" << seq_len << std::endl;
                //     std::cout << "[DBG] original_seq=" << original_seq << std::endl;
                // }
            
                std::string final_seq, final_qual;

                if (info.need_trim)
                {
                    int seq_len = original_seq.length();
                    int start_pos = std::max(0, std::min(info.read_start, seq_len));
                    int end_pos = std::max(start_pos, std::min(info.read_end, seq_len));

                    if (end_pos > start_pos)
                    {
                        final_seq = original_seq.substr(start_pos, end_pos - start_pos);
                        final_qual = original_qual.substr(start_pos, end_pos - start_pos);
                        trimmed_count++;
                    }
                    else
                    {
                        continue;
                    }
                }
                else
                {
                    final_seq = original_seq;
                    final_qual = original_qual;
                }

                // 写入FASTQ
                fout << "@" << read_id;
                if (info.need_trim)
                {
                    fout << " trimmed:" << info.raw_read_start << "-" << (info.raw_read_end - 1);
                }
                fout << "\n"
                     << final_seq << "\n+\n"
                     << final_qual << "\n";

                sequences.push_back(final_seq);
            }

            // 🔥 将参考序列作为最后一个read添加
            if (has_reference)
            {
                std::string ref_read_id = "REFERENCE_" + region.safe_name;

                // 在FASTQ文件末尾写入参考序列
                fout << "@" << ref_read_id << "\n";
                fout << reference_sequence << "\n";
                fout << "+\n";
                // 生成假的质量分数（全部设为高质量'I' = ASCII 73, Phred 40）
                std::string qual(reference_sequence.length(), 'I');
                fout << qual << "\n";

                // 将参考序列添加到POA输入（作为最后一个序列）
                

                // std::cout << "Added reference sequence (" << reference_sequence.length()
                //           << " bp) as last read to POA input" << std::endl;
            }
            if (has_reference && region.use_ref_in_poa) {
                sequences.push_back(reference_sequence);
            }

            fout.close();

            // std::cout << "Generated " << sequences.size() << " sequences ("
            //           << trimmed_count << " trimmed";
            // if (has_reference) {
            //     std::cout << " + 1 reference";
            // }
            // std::cout << ")" << std::endl;
            
            // 保存截断信息JSON
            save_truncation_info_json(reads_map, truncation_file);

            // 🔥 生成groundtruth信息（在POA之前）
            if (has_reference)
            {
                generate_groundtruth_info(region, reference_sequence, groundtruth_file);
            }

            // 运行POA
            if (sequences.size() >= 2)
            {
                // std::cout << "Running abPOA..." << std::endl;
                int poa_result = poa_utils::run_abpoa_on_sequences(sequences, gfa_output);

                if (poa_result == 0)
                {
                    // std::cout << "✓ Successfully generated GFA: " << gfa_output << std::endl;
                    
                    // 只有 abPOA 引入了 ref 的分组，才在 GFA 上标 ref path
                    if (has_reference && region.use_ref_in_poa) {
                        mark_reference_path_in_gfa(gfa_output, reference_sequence, groundtruth_file);
                    }

                    successful_regions++;
                }
                else
                {
                    std::cerr << "✗ abPOA failed for region " << region.region_str << std::endl;
                }
            }
            else
            {
                std::cout << "⚠ Skipping abPOA (need ≥2 sequences, got " << sequences.size() << ")" << std::endl;
            }

            // 内存清理...
            // std::cout << "Cleaning up memory for region " << region.safe_name << "..." << std::endl;
            region_reads.erase(region.safe_name);
            sequences.clear();
            sequences.shrink_to_fit();
            print_memory_usage();
            // std::cout << "Memory cleanup completed for region " << region.safe_name << std::endl;
        }

        std::cout << "\nProcessed " << successful_regions << "/" << total_regions << " regions successfully" << std::endl;
        return successful_regions > 0;
    }

    // 新增：获取参考基因组序列的函数
    // 新实现：不再调用 samtools；直接用 htslib/faidx
    std::string BatchRegionProcessor::get_reference_sequence(const RegionInfo &region)
    {
        if (reference_fasta_file.empty())
        {
            std::cerr << "Reference FASTA path is empty.\n";
            return "";
        }

        // --- 轻量缓存：避免多次加载同一个 .fai ---
        // （静态局部，进程内复用；如果你有多参考基因组，也兼容）
        struct FaiCache
        {
            faidx_t *fai = nullptr;
            std::string path;
            ~FaiCache()
            {
                if (fai)
                    fai_destroy(fai);
            }
        };
        static FaiCache cache;

        auto ensure_fai_loaded = [&](const std::string &fasta_path) -> faidx_t *
        {
            if (cache.fai && cache.path == fasta_path)
                return cache.fai;

            // 如果切换了参考，先释放旧的
            if (cache.fai)
            {
                fai_destroy(cache.fai);
                cache.fai = nullptr;
                cache.path.clear();
            }

            // 先尝试加载现有索引
            faidx_t *fai = fai_load(fasta_path.c_str());
            if (!fai)
            {
                // 没有索引时自动构建（需要对 fasta 目录有写权限）
                std::cerr << "[faidx] Index not found. Building index for: " << fasta_path << std::endl;
                if (fai_build(fasta_path.c_str()) == -1)
                {
                    std::cerr << "[faidx] Failed to build index (.fai) for: " << fasta_path << std::endl;
                    return nullptr;
                }
                fai = fai_load(fasta_path.c_str());
                if (!fai)
                {
                    std::cerr << "[faidx] Failed to load index after building for: " << fasta_path << std::endl;
                    return nullptr;
                }
            }
            cache.fai = fai;
            cache.path = fasta_path;
            return cache.fai;
        };

        faidx_t *fai = ensure_fai_loaded(reference_fasta_file);
        if (!fai)
        {
            std::cerr << "Failed to load FASTA index for: " << reference_fasta_file << std::endl;
            return "";
        }

        // htslib 支持 "chr:start-end" 直接抓取
        int fetch_len = 0;
        const std::string &region_str = region.region_str;
        char *seq_c = fai_fetch(fai, region_str.c_str(), &fetch_len);
        if (!seq_c || fetch_len <= 0)
        {
            std::cerr << "fai_fetch failed for region: " << region_str
                      << " (check contig name and interval)\n";
            if (seq_c)
                free(seq_c);
            return "";
        }

        // 拷贝到 std::string 并释放 htslib 缓冲
        std::string result(seq_c, seq_c + fetch_len);
        free(seq_c);

        // 统一成大写（可选）
        std::transform(result.begin(), result.end(), result.begin(), ::toupper);

        // std::cout << "Retrieved reference sequence: " << result.size()
        //           << " bp for " << region_str << std::endl;
        return result;
    }

    // 新增：生成groundtruth信息的函数
    void BatchRegionProcessor::generate_groundtruth_info(const RegionInfo &region,
                                                         const std::string &reference_seq,
                                                         const std::string &output_file)
    {
        nlohmann::json groundtruth;

        groundtruth["region_name"] = region.safe_name;
        groundtruth["region_str"] = region.region_str;
        groundtruth["reference_sequence"] = reference_seq;
        groundtruth["reference_length"] = reference_seq.length();
        groundtruth["reference_read_id"] = "REFERENCE_" + region.safe_name;
        groundtruth["reference_read_position"] = "last"; // 标记参考read在FASTQ中的位置
        groundtruth["status"] = "pre_poa";               // 标记当前状态

        try
        {
            std::ofstream out(output_file);
            out << groundtruth.dump(2) << std::endl;
            out.close();

            // std::cout << "Initial groundtruth info saved to: " << output_file << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error saving groundtruth info: " << e.what() << std::endl;
        }
    }

    // 新增：分析GFA中的参考路径
    nlohmann::json BatchRegionProcessor::analyze_reference_path_in_gfa(
        const std::vector<std::string> &gfa_lines,
        const std::string &reference_seq)
    {

        nlohmann::json analysis;
        std::map<std::string, std::string> nodes;
        std::vector<std::pair<std::string, std::string>> edges;

        // 解析GFA文件
        for (const auto &line : gfa_lines)
        {
            if (line.empty())
                continue;

            if (line[0] == 'S')
            { // Segment (node)
                std::istringstream iss(line);
                std::string type, node_id, sequence;
                iss >> type >> node_id >> sequence;
                nodes[node_id] = sequence;
            }
            else if (line[0] == 'L')
            { // Link (edge)
                std::istringstream iss(line);
                std::string type, from_node, from_orient, to_node, to_orient, overlap;
                iss >> type >> from_node >> from_orient >> to_node >> to_orient >> overlap;
                edges.push_back({from_node, to_node});
            }
        }

        analysis["total_nodes"] = nodes.size();
        analysis["total_edges"] = edges.size();
        analysis["reference_sequence_length"] = reference_seq.length();

        // 寻找与参考序列匹配的节点
        std::vector<nlohmann::json> matching_nodes;
        for (const auto &node_pair : nodes)
        {
            const std::string &node_id = node_pair.first;
            const std::string &node_seq = node_pair.second;

            // 检查节点序列是否在参考序列中
            size_t pos = reference_seq.find(node_seq);
            if (pos != std::string::npos)
            {
                nlohmann::json match_info;
                match_info["node_id"] = node_id;
                match_info["node_sequence"] = node_seq;
                match_info["position_in_reference"] = pos;
                match_info["node_length"] = node_seq.length();
                matching_nodes.push_back(match_info);
            }
        }

        analysis["matching_nodes"] = matching_nodes;
        analysis["matching_nodes_count"] = matching_nodes.size();

        // 计算覆盖率
        int total_covered = 0;
        for (const auto &match : matching_nodes)
        {
            total_covered += match["node_length"].get<int>();
        }
        analysis["coverage_ratio"] = static_cast<double>(total_covered) / reference_seq.length();

        return analysis;
    }

    // 新增：在GFA中标记参考路径的函数
    void BatchRegionProcessor::mark_reference_path_in_gfa(const std::string &gfa_file,
                                                          const std::string &reference_seq,
                                                          const std::string &groundtruth_file)
    {
        try
        {
            // 读取并解析GFA文件
            std::ifstream gfa_in(gfa_file);
            std::vector<std::string> gfa_lines;
            std::string line;

            while (std::getline(gfa_in, line))
            {
                gfa_lines.push_back(line);
            }
            gfa_in.close();

            // 分析GFA找到参考路径
            auto path_info = analyze_reference_path_in_gfa(gfa_lines, reference_seq);

            // 更新groundtruth文件
            std::ifstream gt_in(groundtruth_file);
            nlohmann::json groundtruth;
            gt_in >> groundtruth;
            gt_in.close();

            // 添加路径信息
            groundtruth["gfa_analysis"] = path_info;
            groundtruth["status"] = "post_poa";
            groundtruth["gfa_file"] = gfa_file;

            std::ofstream gt_out(groundtruth_file);
            gt_out << groundtruth.dump(2) << std::endl;
            gt_out.close();

            // std::cout << "Updated groundtruth with GFA analysis: " << groundtruth_file << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error analyzing GFA: " << e.what() << std::endl;
        }
    }

    void BatchRegionProcessor::print_final_statistics()
    {
        std::cout << "\n"
                  << std::string(60, '=') << std::endl;
        std::cout << "FINAL STATISTICS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        std::cout << "Total regions processed: " << regions.size() << std::endl;
        std::cout << "Total unique reads found: " << all_needed_read_ids.size() << std::endl;
        std::cout << "Reads loaded from FASTQ: " << all_fastq_reads.size() << std::endl;

        // // 按区域统计
        // 打印错的信息
        // for (const auto& region : regions) {
        //     auto it = region_reads.find(region.safe_name);
        //     int read_count = (it != region_reads.end()) ? it->second.size() : 0;
        //     std::cout << "  " << region.region_str << ": " << read_count << " reads" << std::endl;
        // }
    }

    // 辅助函数实现
    bool parse_region(const char *region, std::string &chrom, int32_t &start, int32_t &end)
    {
        std::string region_str(region);

        size_t colon_pos = region_str.find(':');
        if (colon_pos == std::string::npos)
            return false;

        chrom = region_str.substr(0, colon_pos);

        size_t dash_pos = region_str.find('-', colon_pos + 1);
        if (dash_pos == std::string::npos)
            return false;

        try
        {
            start = std::stoi(region_str.substr(colon_pos + 1, dash_pos - colon_pos - 1));
            end = std::stoi(region_str.substr(dash_pos + 1));
        }
        catch (const std::exception &e)
        {
            return false;
        }

        return true;
    }

    int32_t get_query_consumed_length(const uint32_t *cigar, int n_cigar)
    {
        int32_t query_len = 0;
        for (int i = 0; i < n_cigar; i++)
        {
            int op = bam_cigar_op(cigar[i]);
            int len = bam_cigar_oplen(cigar[i]);

            if (op == BAM_CMATCH || op == BAM_CINS || op == BAM_CSOFT_CLIP ||
                op == BAM_CEQUAL || op == BAM_CDIFF)
            {
                query_len += len;
            }
        }
        return query_len;
    }

    void calculate_trim_positions(const uint32_t *cigar, int n_cigar,
                                  int32_t ref_start, int32_t target_start, int32_t target_end,
                                  int32_t &read_start, int32_t &read_end)
    {
        int32_t ref_pos = ref_start;
        int32_t query_pos = 0;

        read_start = -1;
        read_end = -1;

        for (int i = 0; i < n_cigar; i++)
        {
            int op = bam_cigar_op(cigar[i]);
            int len = bam_cigar_oplen(cigar[i]);

            if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF)
            {
                for (int j = 0; j < len; j++)
                {
                    if (read_start == -1 && ref_pos >= target_start)
                    {
                        read_start = query_pos;
                    }

                    if (ref_pos >= target_end)
                    {
                        if (read_end == -1)
                        {
                            read_end = query_pos;
                        }
                        goto finish;
                    }

                    ref_pos++;
                    query_pos++;
                }
            }
            // else if (op == BAM_CINS || op == BAM_CSOFT_CLIP)
            // {
            //     query_pos += len;
            // }
            else if (op == BAM_CINS) {
                // 插入：只消耗 query；只有“窗口内部”的插入最终会被包含
                query_pos += len;
            }
            else if (op == BAM_CSOFT_CLIP) {
                if (read_start == -1) {
                    // 还没进入窗口：开头的 S，qpos 前移但不计入
                    query_pos += len;
                } else {
                    // 已在窗口内：尾部 S 属于窗口外，不能再拼
                    if (read_end == -1) read_end = query_pos;
                    goto finish;  // 立刻终止
                }
            }            
        else if (op == BAM_CDEL || op == BAM_CREF_SKIP)
        {
            for (int j = 0; j < len; j++)
            {
                if (read_start == -1 && ref_pos >= target_start)
                {
                    read_start = query_pos;
                }
                if (ref_pos >= target_end) {
                    if (read_end == -1) {
                        // 删除/跳跃不消耗 query，窗口在缺口结束时不应再前进 query_pos
                        read_end = query_pos;
                    }
                    goto finish;
                }
                ref_pos++;
            }
        }
        }

    finish:
        if (read_start == -1)
            read_start = 0;
        if (read_end == -1)
            read_end = query_pos;
        if (read_end < read_start)
            read_end = read_start;
    }

    void save_truncation_info_json(const std::map<std::string, ReadInfo> &reads_map,
                                   const std::string &output_file)
    {
        nlohmann::json truncation_data;

        for (const auto &pair : reads_map)
        {
            const std::string &read_id = pair.first;
            const ReadInfo &info = pair.second;

            truncation_data[read_id] = {
                {"truncated_start", info.raw_read_start},
                {"truncated_end", info.raw_read_end},
                {"original_length", info.original_length},
                {"flag", info.flag}  // 0/1
            };
        }

        std::ofstream file(output_file);
        if (file.is_open())
        {
            file << truncation_data.dump(2);
            file.close();
            // std::cout << "Truncation info saved: " << output_file << std::endl;
        }
        else
        {
            std::cerr << "Failed to save truncation info: " << output_file << std::endl;
        }
    }
    void print_memory_usage()
    {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line))
        {
            if (line.substr(0, 6) == "VmRSS:")
            {
                // std::cout << "  Current memory usage: " << line << std::endl;
                break;
            }
        }
    }
    // 新增：从BAM记录中计算碱基质量分数
    double BatchRegionProcessor::calculate_base_quality_from_bam(bam1_t *b)
    {
        uint8_t *qual = bam_get_qual(b);
        int32_t seq_len = b->core.l_qseq;

        if (seq_len == 0)
        {
            return 0.0;
        }

        // 检查是否有质量分数（有些BAM可能没有存储质量）
        if (qual[0] == 0xff)
        { // htslib中表示没有质量分数的特殊值
            return 0.0;
        }

        double total_quality = 0.0;
        int valid_bases = 0;

        for (int32_t i = 0; i < seq_len; i++)
        {
            uint8_t phred_score = qual[i];

            // 过滤异常值（通常质量分数不会超过60）
            if (phred_score <= 60)
            {
                total_quality += phred_score;
                valid_bases++;
            }
        }

        if (valid_bases == 0)
        {
            return 0.0;
        }

        return total_quality / valid_bases;
    }

    static inline bool is_simple_bases(const std::string &s)
    {
        for (char c : s)
        {
            char u = std::toupper(static_cast<unsigned char>(c));
            if (!(u == 'A' || u == 'C' || u == 'G' || u == 'T' || u == 'N'))
                return false;
        }
        return true;
    }

    // —— 解析 GT，优先分相 'a|b'，我们取 **右侧 b** 作为 hap2；
    // 若是非分相 'a/b'，没有左右含义，则取最小的非 0 等位（若均为0则返回0）——
    static int allele_from_GT_hap2_policy(const std::string &gt)
    {
        // 分相优先
        size_t bar = gt.find('|');
        if (bar != std::string::npos)
        {
            // hap2 = 右侧
            try
            {
                return std::stoi(gt.substr(bar + 1));
            }
            catch (...)
            {
                return -1;
            }
        }
        // 非分相
        size_t slash = gt.find('/');
        if (slash != std::string::npos)
        {
            int a = -1, b = -1;
            try
            {
                a = std::stoi(gt.substr(0, slash));
            }
            catch (...)
            {
            }
            try
            {
                b = std::stoi(gt.substr(slash + 1));
            }
            catch (...)
            {
            }
            // 取最小的非0（稳定单倍体策略）
            int pick = -1;
            if (a > 0 && b > 0)
                pick = std::min(a, b);
            else if (a > 0)
                pick = a;
            else if (b > 0)
                pick = b;
            else
                pick = 0; // 两个都是 0
            return pick;
        }
        // 其他非常规形式，尽量解析成单数字
        try
        {
            return std::stoi(gt);
        }
        catch (...)
        {
            return -1;
        }
    }

    bool BatchRegionProcessor::load_variants_from_vcf(const std::string &vcf_gz_path)
    {
        htsFile *fp = hts_open(vcf_gz_path.c_str(), "r");
        if (!fp)
        {
            std::cerr << "[SPIKEIN] Failed to open VCF: " << vcf_gz_path << std::endl;
            return false;
        }
        kstring_t ks = {0, 0, nullptr};
        size_t kept = 0, skipped = 0;

        while (hts_getline(fp, KS_SEP_LINE, &ks) >= 0)
        {
            const char *s = ks.s;
            if (!s || s[0] == '#')
                continue;

            // 简易按 \t 切分
            std::vector<std::string> col;
            col.reserve(12);
            const char *p = s;
            const char *q = s;
            while (*q)
            {
                if (*q == '\t')
                {
                    col.emplace_back(p, q - p);
                    p = q + 1;
                }
                ++q;
            }
            col.emplace_back(p, q - p);
            if (col.size() < 10)
            {
                skipped++;
                continue;
            }

            const std::string &chrom = col[0];
            const std::string &pos_s = col[1];
            const std::string &ref = col[3];
            const std::string &alt_s = col[4];
            const std::string &format = col[8];
            const std::string &sample = col[9];

            // 找出 GT 在 FORMAT 里的索引
            int gt_idx = -1;
            {
                int idx = 0;
                size_t i = 0, j = 0;
                while (j <= format.size())
                {
                    if (j == format.size() || format[j] == ':')
                    {
                        if (format.compare(i, j - i, "GT") == 0)
                        {
                            gt_idx = idx;
                            break;
                        }
                        idx++;
                        i = j + 1;
                    }
                    j++;
                }
            }
            if (gt_idx < 0)
            {
                skipped++;
                continue;
            }

            // 提取样本的 GT
            std::string gt;
            {
                int idx = 0;
                size_t i = 0, j = 0;
                while (j <= sample.size())
                {
                    if (j == sample.size() || sample[j] == ':')
                    {
                        if (idx == gt_idx)
                        {
                            gt = sample.substr(i, j - i);
                            break;
                        }
                        idx++;
                        i = j + 1;
                    }
                    j++;
                }
            }
            if (gt.empty())
            {
                skipped++;
                continue;
            }

            // —— 关键：按 “hap2” 策略选择等位 ——
            int pick = allele_from_GT_hap2_policy(gt);
            if (pick <= 0)
            { // 0=REF 或 无效
                skipped++;
                continue;
            }

            // ALT 可能是逗号分隔
            std::vector<std::string> alts;
            {
                size_t i = 0, j = 0;
                while (j <= alt_s.size())
                {
                    if (j == alt_s.size() || alt_s[j] == ',')
                    {
                        alts.emplace_back(alt_s.substr(i, j - i));
                        i = j + 1;
                    }
                    j++;
                }
            }
            if (pick > (int)alts.size())
            {
                skipped++;
                continue;
            }
            std::string alt = alts[pick - 1];

            // ALT="*" 代表缺失，转成空串以便统一 replace
            if (alt == "*")
                alt.clear();

            // 只要“简单 indel”：REF/ALT 均为 A/C/G/T/N（ALT 可为空串）
            // if (!(is_simple_bases(ref) && (alt.empty() || is_simple_bases(alt))))
            // {
            //     skipped++;
            //     continue;
            // }
            // 严格 indel：长度不同
            // if ((int)ref.size() == (int)alt.size())
            // {
            //     skipped++;
            //     continue;
            // }

            int32_t pos1 = 0;
            try
            {
                pos1 = std::stoi(pos_s);
            }
            catch (...)
            {
                skipped++;
                continue;
            }
            if (pos1 <= 0)
            {
                skipped++;
                continue;
            }

            // 记录（按染色体分桶，稍后排序）
            variants_by_chrom[chrom].push_back(Variant{pos1, ref, alt});
            kept++;
        }
        if (ks.s)
            free(ks.s);
        hts_close(fp);

        // 每个染色体内按 POS 升序；若同一 POS 有多次（极少数数据会这样），保留第一条
        for (auto &kv : variants_by_chrom)
        {
            auto &v = kv.second;
            std::sort(v.begin(), v.end(), [](const Variant &a, const Variant &b)
                      { return a.pos_1based < b.pos_1based; });
            std::vector<Variant> uniq;
            uniq.reserve(v.size());
            int32_t last_pos = -1;
            for (auto &x : v)
            {
                if (x.pos_1based != last_pos)
                {
                    uniq.push_back(x);
                    last_pos = x.pos_1based;
                }
                // 同 POS 后续的忽略（避免冲突）
            }
            v.swap(uniq);
        }

        std::cout << "[SPIKEIN] VCF loaded for hap2: kept=" << kept << ", skipped=" << skipped
                  << ", chroms=" << variants_by_chrom.size() << std::endl;
        return kept > 0;
    }
    void BatchRegionProcessor::apply_spikeins(const std::string &chrom,
                                              int32_t win_start, int32_t win_end,
                                              std::string &seq)
    {
        auto it = variants_by_chrom.find(chrom);
        if (it == variants_by_chrom.end() || it->second.empty())
            return;

        const auto &vec = it->second;

        // 找到第一个可能落入窗口的变异
        auto lower = std::lower_bound(vec.begin(), vec.end(), win_start,
                                      [](const Variant &v, int32_t st)
                                      { return v.pos_1based < st; });

        long long shift = 0;
        long long last_l = -1, last_r = -1; // 防止重叠改写

        for (auto vit = lower; vit != vec.end(); ++vit)
        {
            const Variant &var = *vit;
            if (var.pos_1based > win_end)
                break;

            const int ref_len = (int)var.ref.size();
            const int alt_len = (int)var.alt.size();

            // 允许部分落入窗口：把编辑裁剪到 [win_start, win_end] 内
            long long ref_l = var.pos_1based;
            long long ref_r = var.pos_1based + ref_len - 1;

            // 与窗口的交集
            long long ov_l = std::max<long long>(ref_l, win_start);
            long long ov_r = std::min<long long>(ref_r, win_end);
            if (ov_l > ov_r) continue; // 无交集

            // 相对 REF 的左右裁剪量（基于 VCF 左锚定，REF/ALT 同步裁剪）
            int left_clip  = (int)(ov_l - ref_l);
            int right_clip = (int)(ref_r - ov_r);

            std::string ref_clip = var.ref;
            std::string alt_clip = var.alt;

            // 保护边界
            if (left_clip  > 0 && left_clip  < (int)ref_clip.size()) {
                if ((int)alt_clip.size() >= left_clip) alt_clip.erase(0, left_clip);
                ref_clip.erase(0, left_clip);
            }
            if (right_clip > 0 && right_clip < (int)ref_clip.size()) {
                if ((int)alt_clip.size() >= right_clip) alt_clip.erase(alt_clip.size() - right_clip);
                ref_clip.erase(ref_clip.size() - right_clip);
            }
            if (ref_clip.empty() && alt_clip.empty()) continue;

            // 在当前 seq 上的坐标（考虑 shift）
            long long idx = (long long)(ov_l - win_start) + shift;
            if (idx < 0 || idx > (long long)seq.size()) continue;

            // 同位点相邻变异避免重叠改写
            if (last_r <= idx || idx <= last_l) {
                // 校验 REF，仅对裁剪后的 ref_clip 长度校验
                if ((size_t)idx + ref_clip.size() <= seq.size()) {
                    std::string cur = seq.substr((size_t)idx, ref_clip.size());
                    auto upper = [](std::string s){ std::transform(s.begin(), s.end(), s.begin(), ::toupper); return s; };
                    if (upper(cur) != upper(ref_clip)) {
                        // REF 不匹配，跳过该条变异
                        continue;
                    }
                    // 应用替换
                    seq.replace((size_t)idx, ref_clip.size(), alt_clip);
            
                    long long delta = (long long)alt_clip.size() - (long long)ref_clip.size();
                    shift += delta;
            
                    // 更新“已应用区间”到 ALT 新长度区间
                    last_l = idx;
                    last_r = idx + (long long)alt_clip.size();
                }
            } else {
                // 与上一次修改重叠，跳过
                continue;
            }
        }
        //     // 只在 REF 完整落入窗口时应用
        //     if (var.pos_1based < win_start)
        //         continue;
        //     if (var.pos_1based + ref_len - 1 > win_end)
        //         continue;

        //     long long idx = (long long)(var.pos_1based - win_start) + shift;
        //     if (idx < 0 || idx + ref_len > (long long)seq.size())
        //         continue;

        //     // 避免与上一次改写重叠
        //     if (!(last_r <= idx || (idx + ref_len) <= last_l))
        //         continue;

        //     // REF 校验（大小写宽松）
        //     if (seq.compare((size_t)idx, (size_t)ref_len, var.ref) != 0)
        //     {
        //         std::string cur = seq.substr((size_t)idx, (size_t)ref_len);
        //         std::string refU = var.ref, curU = cur;
        //         std::transform(refU.begin(), refU.end(), refU.begin(), ::toupper);
        //         std::transform(curU.begin(), curU.end(), curU.begin(), ::toupper);
        //         if (refU != curU)
        //             continue;
        //     }

        //     // 替换（ALT 为空串即删除）
        //     seq.replace((size_t)idx, (size_t)ref_len, var.alt);

        //     long long delta = (long long)alt_len - (long long)ref_len;
        //     shift += delta;

        //     last_l = idx;
        //     last_r = idx + alt_len; // alt_len==0 时，last_l==last_r
    }
    // void BatchRegionProcessor::apply_spikeins(const std::string &chrom,
    //                                           int32_t win_start, int32_t win_end,
    //                                           std::string &seq)
    // {
    //     // 查找染色体的变异信息
    //     auto it = variants_by_chrom.find(chrom);
    //     if (it == variants_by_chrom.end() || it->second.empty())
    //     {
    //         std::cout << "[DEBUG] No variants found for chromosome: " << chrom << std::endl;
    //         return; // 如果没有变异数据，直接返回
    //     }
    //     std::cout << "[DEBUG] win_start:"<< win_start << "   win_end:" << win_end << std::endl;

    //     const auto &vec = it->second; // 获取染色体的变异列表
    //     auto lower = std::lower_bound(vec.begin(), vec.end(), win_start,
    //                                   [](const Variant &v, int32_t st)
    //                                   { return v.pos_1based < st; });

    //     long long shift = 0;
    //     long long last_l = -1, last_r = -1; // 用于防止替换区域重叠

    //     // 遍历所有变异
    //     for (auto vit = lower; vit != vec.end(); ++vit)
    //     {
    //         const Variant &var = *vit;

    //         // 输出变异信息
    //         std::cout << "[DEBUG] Processing variant: " << var.pos_1based
    //                   << ", ref: " << var.ref << ", alt: " << var.alt << std::endl;

    //         if (var.pos_1based > win_end)
    //         {
    //             std::cout << "[DEBUG] Variant " << var.pos_1based << " is outside the window (end)." << std::endl;
    //             break; // 如果变异位置超出窗口范围，退出循环
    //         }

    //         const int ref_len = (int)var.ref.size(); // 变异参考碱基长度
    //         const int alt_len = (int)var.alt.size(); // 变异替代碱基长度

    //         // 检查变异是否完全在窗口内
    //         if (var.pos_1based < win_start || var.pos_1based + ref_len - 1 > win_end)
    //         {
    //             std::cout << "[DEBUG] Variant " << var.pos_1based << " is outside the window (start or end)." << std::endl;
    //             continue; // 如果变异不在窗口内，跳过
    //         }

    //         long long idx = (long long)(var.pos_1based - win_start) + shift; // 计算变异在序列中的索引
    //         std::cout << "[DEBUG] Calculated idx: " << idx << " for variant at position " << var.pos_1based << std::endl;

    //         if (idx < 0 || idx + ref_len > (long long)seq.size())
    //         {
    //             std::cout << "[DEBUG] Variant position out of sequence bounds. Skipping." << std::endl;
    //             continue; // 如果计算的索引超出序列范围，跳过
    //         }

    //         // 避免与上次替换区域重叠
    //         if (!(last_r <= idx || (idx + ref_len) <= last_l))
    //         {
    //             std::cout << "[DEBUG] Variant at idx " << idx << " overlaps with previous replacement. Skipping." << std::endl;
    //             continue; // 如果当前替换区域与上次替换的区域重叠，跳过
    //         }

    //         // 输出参考碱基的校验信息
    //         std::string ref_seq_in_window = seq.substr((size_t)idx, (size_t)ref_len);
    //         std::cout << "[DEBUG] Reference sequence in window: " << ref_seq_in_window << std::endl;

    //         // 校验参考碱基是否与变异的参考碱基一致
    //         if (seq.compare((size_t)idx, (size_t)ref_len, var.ref) != 0)
    //         {
    //             std::cout << "[DEBUG] Reference mismatch at idx " << idx
    //                       << ": expected " << var.ref << ", got " << ref_seq_in_window << std::endl;
    //             continue; // 如果参考碱基不一致，跳过
    //         }

    //         // 执行替换操作
    //         std::cout << "[DEBUG] Replacing " << var.ref << " with " << var.alt
    //                   << " at position " << idx << std::endl;
    //         seq.replace((size_t)idx, (size_t)ref_len, var.alt);

    //         // 输出替换后的序列片段
    //         std::string updated_seq = seq.substr((size_t)idx, (size_t)alt_len);
    //         std::cout << "[DEBUG] Updated sequence at idx " << idx << ": " << updated_seq << std::endl;

    //         // 更新 shift，处理替换后序列长度的变化
    //         long long delta = (long long)alt_len - (long long)ref_len; // 计算替换后的长度变化
    //         shift += delta;
    //         std::cout << "[DEBUG] Updated shift: " << shift << std::endl;

    //         last_l = idx;           // 更新上次替换的开始位置
    //         last_r = idx + alt_len; // 更新上次替换的结束位置
    //         std::cout << "[DEBUG] Updated last_l: " << last_l << ", last_r: " << last_r << std::endl;
    //     }
    // }

    std::string BatchRegionProcessor::get_reference_sequence_with_spikein(const RegionInfo &region)
    {
        std::string seq = get_reference_sequence(region);
        if (!seq.empty())
            apply_spikeins(region.chrom, region.start, region.end, seq);
        return seq;
    }

    int BatchRegionProcessor::count_errors_in_window(bam1_t* b, int32_t window_start, int32_t window_end) {
        // std::cout << "Window start: " << window_start << ", Window end: " << window_end << std::endl;
        // std::cout << "BAM record position: " << b->core.pos << ", End position: " << bam_endpos(b) << std::endl;
    
        const uint32_t *cigar = bam_get_cigar(b);  // 获取CIGAR操作
        int n_cigar = b->core.n_cigar;
        int total_len = 0;
        int error_count = 0;
    
        // 正确计算skip_len：从窗口起始位置到当前读取的起始位置的差值
        int skip_len = window_start;
        // std::cout << "Skip len: " << skip_len << std::endl;
    
        // 遍历CIGAR操作
        for (int i = 0; i < n_cigar; ++i) {
            int op = bam_cigar_op(cigar[i]);
            int len = bam_cigar_oplen(cigar[i]);
    
            // std::cout << "CIGAR operation: " << op << ", length: " << len << std::endl;
    
            // 跳过不在窗口内的部分
            if (skip_len > 0) {
                // std::cout << "Skipping " << skip_len << " bases before this CIGAR operation" << std::endl;
                if (skip_len >= len) {
                    skip_len -= len;  // 跳过整个CIGAR操作
                } else {
                    len -= skip_len;  // 只跳过一部分
                    skip_len = 0;     // 剩余部分继续处理
                }
            }
    
            if (skip_len == 0) {
                total_len += len;
    
                // 只计算插入、删除和错配为错误
                if (op == BAM_CINS || op == BAM_CDEL || op == BAM_CDIFF) {
                    error_count += len;
                    // std::cout << "Error count increased by: " << len << ", total: " << error_count << std::endl;
                }
            }
    
            // 如果已经覆盖了窗口，退出
            if (total_len >= (window_end - window_start)) {
                break;
            }
        }
    
        // std::cout << "Final error count: " << error_count << std::endl;
        return error_count;
    }
    

    
    
    // int BatchRegionProcessor::count_errors_in_window(bam1_t* b, int32_t window_start, int32_t window_end) {
    //     const uint32_t *cigar = bam_get_cigar(b);  // 获取CIGAR字符串
    //     int n_cigar = b->core.n_cigar;             // 获取CIGAR操作的数量
    //     int total_len = 0;                         // 记录已处理的碱基数
    //     int error_count = 0;                       // 记录错误数

    //     // 计算CIGAR前面需要跳过的部分
    //     int skip_len = window_start - b->core.pos;
    //     if (skip_len < 0) {
    //         skip_len = 0;
    //     }

    //     // 遍历所有CIGAR操作
    //     for (int i = 0; i < n_cigar; ++i) {
    //         int op = bam_cigar_op(cigar[i]);  // 获取CIGAR操作类型
    //         int len = bam_cigar_oplen(cigar[i]);  // 获取该操作的长度

    //         // 如果CIGAR操作的长度超出了需要跳过的部分
    //         if (skip_len > 0) {
    //             if (skip_len >= len) {
    //                 skip_len -= len;  // 跳过这一部分
    //             } else {
    //                 // 如果跳过部分小于当前CIGAR操作长度，更新CIGAR长度并开始计算
    //                 len -= skip_len;
    //                 skip_len = 0;
    //             }
    //         }

    //         // 只处理窗口内的CIGAR操作
    //         if (skip_len == 0) {
    //             total_len += len;

    //             // 错误的CIGAR操作：插入、删除、错配
    //             if (op == BAM_CINS || op == BAM_CDEL || op == BAM_CMATCH || op == BAM_CSOFT_CLIP) {
    //                 error_count += len;  // 错误数累加
    //             }

    //             // 如果已经覆盖了100bp并且错误数没有超过5个，直接返回
    //             if (total_len >= (window_end - window_start) && error_count <= 5) {
    //                 return error_count;  // 满足条件，直接返回
    //             }
    //         }

    //         // 如果当前操作已经处理完毕，且已经覆盖到窗口结束位置，则退出
    //         if (total_len >= (window_end - window_start)) {
    //             break;
    //         }
    //     }

    //     return error_count;  // 返回错误数
    // }
    

} // namespace extract_utils
