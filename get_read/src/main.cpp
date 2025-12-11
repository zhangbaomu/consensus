#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_set>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <sys/stat.h>
#include "../include/extract_utils.h"
#include "../include/poa_utils.h"

#include <chrono>          // 添加这行
#include <sstream>         // 添加这行
#include <regex>           // 添加这行（如果使用正则表达式）

// 创建目录的辅助函数
bool create_directory(const std::string& path) {
    struct stat st = {0};
    if (stat(path.c_str(), &st) == -1) {
        if (mkdir(path.c_str(), 0755) == -1) {
            std::cerr << "Error: Failed to create directory: " << path << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    using clock = std::chrono::steady_clock;
    auto total_start = clock::now();
    // 检查命令行参数
    if (argc < 4 || argc > 7) {
        std::cerr << "Usage: " << argv[0] << " <regions_file> <output_dir> <bam_file> [--flat-output] [--enable-consensus-fasta] [--chroms chr1,chr2]" << std::endl;
        std::cerr << "regions.txt should contain one region per line in format: chr1:10000-20000" << std::endl;
        return 1;
    }
    
    const char* regions_file = argv[1];
    std::string base_output_dir= argv[2];
    const char* bam_file = argv[3];

    bool flat_output_mode = false;
    bool consensus_enabled = false;
    std::unordered_set<std::string> allowed_chroms;
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--flat-output") {
            flat_output_mode = true;
        } else if (arg == "--enable-consensus-fasta") {
            consensus_enabled = true;
        } else if (arg.rfind("--chroms", 0) == 0) {
            std::string list;
            if (arg == "--chroms" && i + 1 < argc) {
                list = argv[++i];
            } else {
                auto pos = arg.find('=');
                if (pos != std::string::npos) list = arg.substr(pos + 1);
            }
            if (!list.empty()) {
                size_t start = 0;
                while (start < list.size()) {
                    size_t comma = list.find(',', start);
                    std::string item = list.substr(start, (comma == std::string::npos ? std::string::npos : comma - start));
                    if (!item.empty()) allowed_chroms.insert(item);
                    if (comma == std::string::npos) break;
                    start = comma + 1;
                }
            }
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
    }

    std::cout << "Reading BAM file: " << bam_file << std::endl;
    // 添加参考基因组文件路径
    const char* reference_fasta = "/home/user/zhangbaomu/data/ref/grch38/GRCh38_full_analysis_set_plus_decoy_hla.fa";
    
    // 创建批处理器时传递参考基因组文件
    
    // 固定的输入文件路径
    // const char* bam_file = "/home/user/zhangbaomu/basecall/data/PAU71468/out.minimap.sorted.fat01.bam";
    // const char* fastq_file = "/home/user/zhangbaomu/basecall/data/PAU71468/out.fastq";

    // const char* bam_file = "/home/user/zhangbaomu/basecall/data/PAU_dorado_latest/dorado_hac_5.2.0.sort.bam";
    // const char* fastq_file = "/home/user/zhangbaomu/basecall/data/PAU_dorado_latest/dorado_hac_5.2.0.fastq";
    // const char* bam_file = "/home/user/zhangbaomu/basecall/data/PAU_dorado_latest/dorado_sup_5.2.0.sort.bam";
    // const char* fastq_file = "/home/user/zhangbaomu/basecall/data/PAU_dorado_latest/dorado_sup_5.2.0.fastq";
    
    // 基础输出目录
    // std::string base_output_dir = "/home/user/zhangbaomu/basecall/test/test_get_read/regions_data";
    
    // 创建基础输出目录
    if (!create_directory(base_output_dir)) {
        return 1;
    }
    
    // 创建批处理器
    extract_utils::BatchRegionProcessor processor(base_output_dir, reference_fasta);
    processor.set_flat_output_mode(flat_output_mode);
    processor.set_consensus_enabled(consensus_enabled);
    if (!allowed_chroms.empty()) {
        processor.set_allowed_chroms(allowed_chroms);
    }
    
    processor.set_max_reads_per_region(200);  // 每个区域最多reads
    processor.set_min_mapping_quality(1);   // 最低mapping quality
    std::cout << "=== Batch Region Processing Pipeline ===" << std::endl;
    
    // 步骤1: 加载所有区域
    auto t1_start = clock::now();
    std::cout << "\nStep 1: Loading regions from file..." << std::endl;
    if (!processor.load_regions_from_file(regions_file)) {
        std::cerr << "Failed to load regions from file!" << std::endl;
        return 1;
    }
    auto t1_end = clock::now();
    std::cout << "Step 1 took "
              << std::chrono::duration_cast<std::chrono::seconds>(t1_end - t1_start).count()
              << " s\n";

    // === 新增：加载 HG002 的 VCF（一次性载入内存）===
    const std::string hg002_vcf = "/home/user/zhangbaomu/basecall/tools/clair3/hg002/GRCh38_HG2-T2TQ100-V1.1_stvar.NO_STAR.pass.vcf.gz";
    if (!processor.load_variants_from_vcf(hg002_vcf))
    {
        std::cerr << "[SPIKEIN] Warning: no variants loaded; clean reference will be used.\n";
    }

    // 步骤2: 一次性扫描BAM文件，提取所有区域的read信息
    auto t2_start = clock::now();
    std::cout << "\nStep 2: Scanning BAM file for all regions..." << std::endl;
    if (!processor.extract_all_reads_from_bam(bam_file)) {
        std::cerr << "Failed to extract reads from BAM file!" << std::endl;
        return 1;
    }
    auto t2_end = clock::now();
    std::cout << "Step 2 took "
              << std::chrono::duration_cast<std::chrono::seconds>(t2_end - t2_start).count()
              << " s\n";
           
    // 步骤3: 一次性扫描FASTQ文件，加载所有需要的reads
    // auto t3_start = clock::now();   
    // std::cout << "\nStep 3: Loading all required reads from FASTQ..." << std::endl;
    // if (!processor.load_all_fastq_reads(fastq_file)) {
    //     std::cerr << "Failed to load reads from FASTQ file!" << std::endl;
    //     return 1;
    // }
    // auto t3_end = clock::now();

    
    // 步骤4: 处理所有区域（截断和POA）
    auto t4_start = clock::now();
    std::cout << "\nStep 4: Processing all regions..." << std::endl;
    if (!processor.process_all_regions()) {
        std::cerr << "Failed to process all regions!" << std::endl;
        return 1;
    }
    auto t4_end = clock::now();
    std::cout << "Step 4 took "
              << std::chrono::duration_cast<std::chrono::seconds>(t4_end - t4_start).count()
              << " s\n";

    // 打印最终统计
    processor.print_final_statistics();
    
    auto total_end = clock::now();
    std::cout << "Total elapsed time: "
              << std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count()
              << " s\n";

    std::cout << "\n✓ All regions processed successfully!" << std::endl;
    return 0;
}
