#include "cli.h"
#include <getopt.h>
#include <iostream>
#include <algorithm>
#include <cstdlib>

CLI parse_cli(int argc, char** argv){
    CLI o; int c;
    const char* optstr = "b:o:t:w:r:R:V:L:h"; // r=overlap
    while ((c = getopt(argc, argv, optstr)) != -1){
        switch(c){
            case 'b': o.bam = optarg; break;
            case 'o': o.out_fa = optarg; break;
            case 't': o.threads = std::max(1, atoi(optarg)); break;
            case 'w': o.win_len = std::max(100, atoi(optarg)); break;
            case 'r': o.overlap = std::max(1, atoi(optarg)); break;
            case 'R': o.eval_ref = optarg; break;
            case 'V': o.eval_vcf = optarg; break;
            case 'L': {
                long long v = atoll(optarg);
                if (v > 0) o.eval_max_len = static_cast<size_t>(v);
                break;
            }
            case 'h': default:
                std::cerr << "Usage: consensus2 -b in.bam [-o out.fa] [-t threads] [-w 1000] [-r 50]"
                             " [-R ref.fa] [-V variants.vcf.gz] [-L max_eval_len]\n";
                exit(1);
        }
    }
    if (o.bam.empty()){ std::cerr << "-b BAM required\n"; exit(1);}    
    return o;
}
