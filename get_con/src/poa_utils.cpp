// poa_utils.cpp
#include "../include/poa_utils.h"
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <vector>

// 重要：用extern "C"包装abPOA的C头文件
extern "C" {
#include "../include/abpoa.h"
}

namespace poa_utils {

std::vector<std::string> load_sequences_from_fastq(const std::string& fastq_file) {
    std::vector<std::string> sequences;
    std::ifstream fin(fastq_file);
    
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open file " << fastq_file << std::endl;
        return sequences;
    }
    
    std::string line, header, seq, plus, qual;
    
    
    while (std::getline(fin, header) ) {
        if (!std::getline(fin, seq)) break;
        if (!std::getline(fin, plus)) break;  
        if (!std::getline(fin, qual)) break;
        
        // 检查序列是否为空
        if (!seq.empty()) {
            sequences.push_back(seq);
            
        }
    }
    
    std::cout << "Loaded " << sequences.size() << " sequences from " << fastq_file << std::endl;
    return sequences;
}

int run_abpoa_on_sequences(const std::vector<std::string>& sequences,
                          const std::string& output_gfa_file) {

    if (sequences.empty()) {
        std::cerr << "No sequences to process!" << std::endl;
        return -1;
    }

    // std::cout << "Starting abPOA processing with " << sequences.size() << " sequences..." << std::endl;

    // 使用vector自动管理内存，避免手动new/delete
    std::vector<std::vector<uint8_t>> uint8_sequences;
    std::vector<uint8_t*> seq_ptrs;
    std::vector<int> seq_lens;
    
    // 预分配内存
    uint8_sequences.reserve(sequences.size());
    seq_ptrs.reserve(sequences.size());
    seq_lens.reserve(sequences.size());

    // 转换序列格式：字符串 -> uint8_t数组
    for (size_t i = 0; i < sequences.size(); ++i) {
        const auto& seq = sequences[i];
        
        if (seq.empty()) {
            std::cout << "Warning: Empty sequence at index " << i << ", skipping..." << std::endl;
            continue;
        }

        // 使用vector自动管理内存
        std::vector<uint8_t> uint8_seq;
        uint8_seq.reserve(seq.length());

        // 转换ACGT -> 0123
        for (char c : seq) {
            switch (c) {
                case 'A': case 'a': uint8_seq.push_back(0); break;
                case 'C': case 'c': uint8_seq.push_back(1); break;
                case 'G': case 'g': uint8_seq.push_back(2); break;
                case 'T': case 't': uint8_seq.push_back(3); break;
                case 'N': case 'n': uint8_seq.push_back(0); break;  // N作为A处理
                default: 
                    std::cout << "Warning: Unknown base '" << c << "' in sequence " << i << ", treating as A" << std::endl;
                    uint8_seq.push_back(0);
                    break;
            }
        }

        if (!uint8_seq.empty()) {
            uint8_sequences.push_back(std::move(uint8_seq));
            seq_ptrs.push_back(uint8_sequences.back().data());
            seq_lens.push_back(static_cast<int>(seq.length()));
        }
    }

    if (seq_ptrs.empty()) {
        std::cerr << "No valid sequences after conversion!" << std::endl;
        return -1;
    }

    // std::cout << "Converted " << seq_ptrs.size() << " valid sequences" << std::endl;

    // 初始化abPOA
    // std::cout << "Initializing abPOA..." << std::endl;
    
    abpoa_para_t *abpt = abpoa_init_para();
    if (!abpt) {
        std::cerr << "Failed to initialize abPOA parameters!" << std::endl;
        return -1;
    }

    // 设置参数
    abpt->align_mode = ABPOA_GLOBAL_MODE;
    abpt->out_gfa = 1;
    abpt->max_n_cons = 1;  // 限制共识序列数量
    abpt->wb = 10;         // 减少band width (默认通常更大)
    abpt->wf = 0.01;       // 减少额外带宽比例
    
    abpoa_post_set_para(abpt);

    abpoa_t *ab = abpoa_init();
    if (!ab) {
        std::cerr << "Failed to initialize abPOA!" << std::endl;
        abpoa_free_para(abpt);
        return -1;
    }
    
    // 打开输出文件
    // std::cout << "Opening output file: " << output_gfa_file << std::endl;
    
    FILE *gfa_fp = fopen(output_gfa_file.c_str(), "w");
    if (!gfa_fp) {
        std::cerr << "Cannot open output file: " << output_gfa_file << std::endl;
        abpoa_free(ab);
        abpoa_free_para(abpt);
        return -1;
    }

    // std::cout << "Running abPOA MSA..." << std::endl;
    
    // 调用abPOA - 这里最容易出错
    int result = abpoa_msa(ab, abpt, 
                          static_cast<int>(seq_ptrs.size()),  // n_seq
                          NULL,                               // seq_names
                          seq_lens.data(),                    // seq_lens
                          seq_ptrs.data(),                    // seqs
                          NULL,                               // qual_weights
                          gfa_fp);                            // out_fp

    // std::cout << "abPOA MSA finished with result code: " << result << std::endl;
    
    // 清理资源
    fclose(gfa_fp);
    abpoa_free(ab);
    abpoa_free_para(abpt);
    uint8_sequences.clear();
    uint8_sequences.shrink_to_fit();
    seq_ptrs.clear();
    seq_ptrs.shrink_to_fit();
    seq_lens.clear();
    seq_lens.shrink_to_fit();
    
    if (result == 0) {
        // std::cout << "Success! GFA written to: " << output_gfa_file << std::endl;
    } else {
        std::cerr << "abPOA failed with error code: " << result << std::endl;
    }

    return result;
}

}