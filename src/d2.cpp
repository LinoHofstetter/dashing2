#include "d2.h"
#include <filesystem>

namespace dashing2 {
int cmp_main(int argc, char **argv, DistanceCallback callback, SketchingResult &sketch1, SketchingResult &sketch2, bool cmp_objects);
int contain_main(int argc, char **argv);
int wsketch_main(int argc, char **argv);
int sketch_main(int argc, char **argv);
int printmin_main(int argc, char **argv);
std::string Dashing2Options::to_string() const {
    size_t m = 4096;
    std::string ret(m, '\0');
    auto pos = std::sprintf(&ret[0], "Dashing2Options;k:%d", k_);
    if(w_ > 0) pos += std::sprintf(&ret[pos], ";w:%d", w_);
    pos += std::sprintf(&ret[pos], ";%s", parse_by_seq_ ? "parsebyseq": "parsebyfile");
    if(trim_chr_) pos += std::sprintf(&ret[pos], ";trimchr");
    pos += std::sprintf(&ret[pos], ";sketchsize:%zu", sketchsize_);
    if(count_threshold_ > 0)
        pos += std::sprintf(&ret[pos], ";%u", count_threshold_);
    pos += std::sprintf(&ret[pos], ";sketchtype:%s",
            kmer_result_ == ONE_PERM ? "onepermsetsketch"
                  : kmer_result_ == FULL_SETSKETCH ? (sspace_ == SPACE_SET ? "fullsetsketch": sspace_ == SPACE_MULTISET ? "bagminhash": sspace_ == SPACE_PSET ? "probminhash": sspace_ == SPACE_EDIT_DISTANCE ? "orderminhash": "unknown")
                  : kmer_result_ == FULL_MMER_SEQUENCE ? (use128() ? "mmerseq128": "mmerseq64")
                  : kmer_result_ == FULL_MMER_SET ? (use128() ? "mmerset128": "mmerset64"): "fullyunknown"
    );
    if(kmer_result_ == FULL_MMER_COUNTDICT)
        pos += std::sprintf(&ret[pos], ",kmercountsf64");
    pos += std::sprintf(&ret[pos], ";%s", ::dashing2::to_string(dtype_).data());
    if(spacing_.size()) {
        pos += std::sprintf(&ret[pos], ";spacing:%s", spacing_.data());
    }
    if(outprefix_.size()) pos += std::sprintf(&ret[pos], ";outprefix:%s", outprefix_.data());
    if(cssize_) pos += std::sprintf(&ret[pos], ";counting=countsketch%zu\n", cssize_);
    if(bed_parse_normalize_intervals_) pos += std::sprintf(&ret[pos], ";normalize_intervals");
    if(by_chrom_) pos += std::sprintf(&ret[pos], ";sketchbychrom");
    if(homopolymer_compress_minimizers_) pos += std::sprintf(&ret[pos], ";hp-compress-minimizers");
    if(canonicalize()) pos += std::sprintf(&ret[pos], ";canon");
    if(fs_) {
        pos += std::sprintf(&ret[pos], ";%s", fs_->to_string().data());
    }
    ret.resize(pos);
    return ret;
}

void Dashing2Options::filterset(const std::string &fsarg) {
    if(fsarg.empty()) return;
    auto i = fsarg.find_last_of(':');
    filterset(fsarg.substr(0, i), (i != std::string::npos && static_cast<char>(fsarg[i + 1] & 0xdf) != 'K'));
}

void Dashing2Options::filterset(const std::string &path, bool is_kmer) {
    fs_.reset(new FilterSet());
    if(is_kmer) {
        std::string cmd;
        if(endswith(path, ".xz")) cmd = "xz -dc ";
        else if(endswith(path, ".gz")) cmd = "gzip -dc ";
        else if(endswith(path, ".bz2")) cmd = "bzip2 -dc ";
        else if(endswith(path, ".zst")) cmd = "zstd -dc ";
        else cmd = "";
        std::FILE *ifp;
        if(cmd.empty()) {
            if((ifp = bfopen(cmd.data(), "rb")) == 0)
                THROW_EXCEPTION(std::runtime_error("Failed to open file "s + path + " for reading"));
        } else {
            if((ifp= ::popen((cmd + path).data(), "r")) == 0)
                THROW_EXCEPTION(std::runtime_error("Failed to run command "s + cmd + path));
        }
        uint64_t u6; u128_t u12;
        void *const ptr = use128() ? (void *)&u12: (void *)&u6;
        for(const auto is(use128() ? 16: 8);std::fread(ptr, is, 1, ifp) == 1;) {
            if(use128()) fs_->add(u12);
            else         fs_->add(u6);
        }
        if(cmd.empty())
            std::fclose(ifp);
        else
            ::pclose(ifp);
    } else {
        for_each_substr([&](const std::string &subpath) {
            //std::fprintf(stderr, "Doing for_each_substr for subpath = %s\n", subpath.data());
            const auto sp = subpath.data();
            auto lfunc = [&](auto x) {fs_->add(maskfn(x));};
            if(use128()) {
                if(unsigned(k_) <= enc_.nremperres128()) {
                    auto encoder(enc_.to_u128());
                    encoder.for_each(lfunc, sp);
                } else {
                    rh128_.for_each_hash(lfunc, sp);
                }
            } else if(unsigned(k_) <= enc_.nremperres64()) {
                enc_.for_each(lfunc, sp);
            } else {
                rh_.for_each_hash(lfunc, sp);
            }
        }, path);
    }
    fs_->finalize();
}
void Dashing2Options::validate() const {
    if(canonicalize() && rh_.hashtype() != bns::DNA) {
        canonicalize(false);
    }
    const int lim = use128() ? enc_.nremperres128() : enc_.nremperres64();
    if(entmin && (k_ > lim || !sp_.unspaced())) {
        std::fprintf(stderr, "Warning: entropy minimization is not supported for spaced seeds or rolling hashers; falling back to random minimization.\n");
        entmin = false;
    }
}

bool entmin = false;
int verbosity = 0;


void set_verbosity(Verbosity level) { //added to be able to modify verbosity from outside the library
    verbosity = level;
}

} // dashing2

int main_usage() {
    std::fprintf(stderr, "dashing2 has several subcommands: sketch, cmp, wsketch, and contain.\n");
    std::fprintf(stderr, "Usage can be seen in those subcommands. (e.g., `dashing2 sketch -h`)\n\n");
    std::fprintf(stderr, "\tsketch: converts FastX into k-mer sets/sketches, and sketches BigWig and BED files; also contains functionality from cmp, for one-step sketch and comparisons\n"
                         "This is probably the most common subcommand to use.\n\n"
    );
    std::fprintf(stderr, "\tcmp: compares previously sketched/decomposed k-mer sets and emits results. alias: dist\n\n");
    std::fprintf(stderr, "\tcontain: Takes a k-mer database (built with dashing2 sketch --save-kmers), then computes coverage for all k-mer references using input streams.\n");
    std::fprintf(stderr, "\twsketch: Takes a tuple of [1-3] input binary files [(u32 or u64), (float or double), (u32 or u64)] and performs weighted minhash sketching.\n"
                         "Three files are treated as Compressed Sparse Row (CSR)-format, where the third file contains indptr values, specifying the lengths of consecutive runs of pairs in the first two files corresponding to each row.\n"
                         "wsketch is for sketching binary files which have already been summed, whereas sketch is for parsing and sketching (from Fast{qa}, BED, BigWig)\n");
    std::fprintf(stderr, "\n\nMiscellania:\n");
    std::fprintf(stderr, "printmin: Emit minimizer sequence sets in human-readable form.\n");
    return 1;
}
using namespace dashing2;


void sketch_wrapper(const std::string &input_filepaths, const std::string &sketch_output_dir, const std::string &flag1) {
    std::vector<std::string> args = {
        "dashing2",               // Command (not actually used but placeholders for argv[0])
        "sketch",
        flag1,                 
        "--cache", //cache sketches -> CHECK IF THIS IS NEEDED
        "--outprefix", sketch_output_dir, //specify where to save sketches
        "-F",                     // Indicate to read files from a list
        input_filepaths,             // path to file that holds paths to the fasta files needed to be sketched
    };

    std::vector<char*> argv;
    for (auto& arg : args) { // Convert std::string arguments to char*
        argv.push_back(&arg.front());
    }
    int argc = argv.size();
    int result = dashing2_main(argc, argv.data());

    if (result != 0) {
       throw std::runtime_error("d2.cpp/sketch_wrapper() failed"); 
    }
    return;
}

int cmp_presketched(const std::string &sketch1, const std::string &sketch2) {
    float distance_0_1 = 0.0;
    DistanceCallback callback = [&](size_t i, size_t j, float distance) {
        if (i == 0 && j == 1) {
            distance_0_1 = distance;
        }
    };

    std::vector<std::string> args = {
        "dashing2",                // Command 
        "cmp",                     // Subcommand
        "--presketched",           // Flag to indicate pre-sketched files
        sketch1,  // Path to first precomputed sketch
        sketch2  // Path to second precomputed sketch
    };

    std::vector<char*> argv;
    for (auto& arg : args) { // Convert std::string arguments to char*
        argv.push_back(&arg.front());
    }
    int argc = argv.size();
    int result = dashing2_main(argc, argv.data(), callback);

    if (result != 0) {
       throw std::runtime_error("d2.cpp/sketch_wrapper() failed"); 
    }

    return distance_0_1;
}

float cmp_sketches(std::shared_ptr<dashing2::SketchingResult> sketch1, std::shared_ptr<dashing2::SketchingResult> sketch2) {
    float distance_0_1 = 0.0;
    DistanceCallback callback = [&](size_t i, size_t j, float distance) {
        if (i == 0 && j == 1) {
            distance_0_1 = distance;
        }
    };

    std::vector<std::string> args = {
        "dashing2",                // Command 
        "cmp",                     // Subcommand
        "--presketched",           // Flag to indicate pre-sketched files
        "EMPTY_PATH",  // Path to first precomputed sketch -> not needed in implementation, name is stored in names_ field after loading from disc
        "EMPTY_PATH"  // Path to second precomputed sketch -> same here
    };

    std::vector<char*> argv;
    std::vector<std::unique_ptr<char[]>> arg_buffers; // Buffers to keep the C-style strings alive

    for (const auto& arg : args) {
        std::unique_ptr<char[]> buffer = std::make_unique<char[]>(arg.size() + 1);
        std::strcpy(buffer.get(), arg.c_str());
        argv.push_back(buffer.get());
        arg_buffers.push_back(std::move(buffer)); // Keep the buffer alive
    }

    argv.push_back(nullptr); // Ensure null-termination
    int argc = argv.size() - 1; // Don't count the null terminator

    int result = dashing2_main(argc, argv.data(), callback, sketch1, sketch2, true); 
    
    if (result != 0) {
       throw std::runtime_error("cmp_sketches() failed"); 
    }

    //std::cout << "CMP-OBJECTS DISTANCE = " << distance_0_1 << std::endl;

    return distance_0_1;
}

// Function to call dashing2_main and generate the output file
void exact_kmc(const int kmer_size, const std::string& fasta_filepath, const std::string& out_prefix) {
    // Extract the part between the last '/' and the next '.' in the filepath
    size_t last_slash_pos = fasta_filepath.find_last_of('/');
    size_t dot_pos = fasta_filepath.find('.', last_slash_pos);
    if (last_slash_pos == std::string::npos || dot_pos == std::string::npos) {
        throw std::runtime_error("Invalid filepath format");
    }
    std::string name = fasta_filepath.substr(last_slash_pos + 1, dot_pos - last_slash_pos - 1);
    std::string outfile = out_prefix + name + "_kmers.d2";

    std::vector<std::string> args = {
        "dashing2",                // Command 
        "sketch",                  // Subcommand
        "--set",                   // Flag to indicate pre-sketched files
        "-k" + std::to_string(kmer_size), // k-mer size flag with value
        "-o",                      // Output flag
        outfile,                   // Output file path
        fasta_filepath             // Input FASTA file path
    };

    std::vector<char*> argv;
    std::vector<std::unique_ptr<char[]>> arg_buffers; // Buffers to keep the C-style strings alive

    for (const auto& arg : args) {
        std::unique_ptr<char[]> buffer = std::make_unique<char[]>(arg.size() + 1);
        std::strcpy(buffer.get(), arg.c_str());
        argv.push_back(buffer.get());
        arg_buffers.push_back(std::move(buffer)); // Keep the buffer alive
    }

    argv.push_back(nullptr); // Ensure null-termination
    int argc = argv.size() - 1; // Don't count the null terminator

    int result = dashing2_main(argc, argv.data()); 
    
    if (result != 0) {
        throw std::runtime_error("exact_kmc() failed"); 
    }

    return;
}


//Workaround for default arguments
int dashing2_main(int argc, char **argv, DistanceCallback callback) {
    if (verbosity >= Verbosity::DEBUG){ //added for debugging
        std::cout << "Inside reduced args dashing2_main, about to real dashing2_main" << std::endl;
    }
    auto sketch1 = std::make_shared<dashing2::SketchingResult>();
    auto sketch2 = std::make_shared<dashing2::SketchingResult>();
    bool cmp_objects = false;

    // Delegate to the full version of dashing2_main
    return dashing2_main(argc, argv, callback, sketch1, sketch2, cmp_objects);
}

int dashing2_main(int argc, char **argv, DistanceCallback callback, std::shared_ptr<dashing2::SketchingResult> sketch1, std::shared_ptr<dashing2::SketchingResult> sketch2, bool cmp_objects) {
    if (verbosity >= Verbosity::DEBUG){ //added for debugging
        std::cout << "Inside dashing2_main, about to call command functions based on arguments" << std::endl;
    }
    std::string cmd(std::filesystem::absolute(std::filesystem::path(argv[0])));
    for(char **s = (argv + 1); *s; cmd += std::string(" ") + *s++);
    if (verbosity >= Verbosity::DEBUG){
        std::fprintf(stderr, "#Calling Dashing2 version %s with command '%s'\n", DASHING2_VERSION, cmd.data());
    }
    if(argc > 1) {
        if (verbosity >= Verbosity::DEBUG){ //added for debugging
            std::cout << "Inside argc > 1 in dashing2_main" << std::endl;
        }
        if(std::strcmp(argv[1], "sketch") == 0)
            return sketch_main(argc - 1, argv + 1);
        if(std::strcmp(argv[1], "cmp") == 0 || std::strcmp(argv[1], "dist") == 0) {
            return cmp_main(argc - 1, argv + 1, callback, *sketch1, *sketch2, cmp_objects);
        }
        if(std::strcmp(argv[1], "wsketch") == 0)
            return wsketch_main(argc - 1, argv + 1);
        if(std::strcmp(argv[1], "contain") == 0)
            return contain_main(argc - 1, argv + 1);
        if(std::strcmp(argv[1], "printmin") == 0) {
            return printmin_main(argc - 1, argv + 1);
        }
    }
    return main_usage();
}
