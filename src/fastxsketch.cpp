#include "fastxsketch.h"
#include "mio.hpp"
#include "sketch_core.h"
#include <variant>

//#include <optional>
namespace dashing2 {
using namespace variation;


void FastxSketchingResult::print() {
    std::fprintf(stderr, "%s\n", str().data());
}

using BKRegT = std::conditional_t<(sizeof(RegT) == 4), uint32_t, std::conditional_t<(sizeof(RegT) == 8), uint64_t, u128_t>>;

template<typename C, typename T>
void pop_push(C &c, T &&x, size_t k) {
    if(c.size() < k) c.push(std::move(x));
    else if(x < c.top()) {c.pop(); c.push(std::move(x));}
}

template<typename SrcT, typename CountT=uint32_t>
void bottomk(const std::vector<SrcT> &src, std::vector<BKRegT> &ret, double threshold=0., const CountT *ptr=(CountT *)nullptr, int weighted=-1) {
    if(weighted < 0) weighted = ptr != 0;
    const size_t k = ret.size(), sz = src.size();
    std::priority_queue<BKRegT> pq;
    std::priority_queue<std::pair<double, BKRegT>> wpq;
    for(size_t i = 0; i < sz; ++i) {
        const auto item = src[i];
        const CountT count = ptr ? ptr[i]: CountT(1);
        if(count > threshold) {
            if(weighted) {
                const std::pair<double, BKRegT> key {double(item / count), item};
                pop_push(wpq, key, k);
            } else {
                const BKRegT key = item;
                pop_push(pq, key, k);
            }
        }
    }
    if(weighted) {
        for(size_t i = k; i > 0;ret[--i] = wpq.top().second, wpq.pop());
    } else {
        for(size_t i = k; i > 0;ret[--i] = pq.top(), pq.pop());
    }
}

int32_t num_threads() {
    int nt = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        nt = omp_get_num_threads();
    }
#endif
    return nt;
}
/*  T: type of elements in sketch
    ptr: where the sketch will be loaded into memory
    ss: sketch size
    returns number of elements/registers in sketch

    Commented it out because I moved the definition of the complete load_copy template to the headerfile
*/
/*
template<typename T, size_t chunk_size> // = 65536, deleted definition of chunk_size, because this now takes place in the headerfile
size_t load_copy(const std::string &path, T *ptr, double *cardinality, const size_t ss) {
    T *const origptr = ptr;
    if(path.size() > 3 && std::equal(path.data() + path.size() - 3, &path[path.size()], ".gz")) { //case for .gz files
        gzFile fp = gzopen(path.data(), "rb");
        if(!fp) return 0; //THROW_EXCEPTION(std::runtime_error(std::string("Failed to open file at ") + path));
        gzread(fp, cardinality, sizeof(*cardinality)); //read cardinality into cardinality variable
        for(int nr; 
            !gzeof(fp) && (nr = gzread(fp, ptr, sizeof(T) * chunk_size)) == sizeof(T) * chunk_size;
            ptr += nr / sizeof(T)); //read the registers into 
        gzclose(fp);
        return ptr - origptr;
    } else if(path.size() > 3 && std::equal(path.data() + path.size() - 3, &path[path.size()], ".xz")) { //case for .xz files
        auto cmd = std::string("xz -dc ") + path;
        std::FILE *fp = ::popen(cmd.data(), "r");
        if(fp == 0) return 0;
        std::fread(cardinality, sizeof(*cardinality), 1, fp);
        for(auto up = (uint8_t *)ptr;!std::feof(fp) && std::fread(up, sizeof(T), chunk_size, fp) == chunk_size; up += chunk_size * sizeof(T));
        ::pclose(fp);
        return ptr - origptr;
    }
    std::FILE *fp = bfopen(path.data(), "rb");
    if(!fp) THROW_EXCEPTION(std::runtime_error(std::string("Failed to open ") + path));
    std::fread(cardinality, sizeof(*cardinality), 1, fp);
    const int fd = ::fileno(fp);
    size_t sz = 0;
    if(!::isatty(fd)) {
        struct stat st;
        if(::fstat(fd, &st)) THROW_EXCEPTION(std::runtime_error(std::string("Failed to fstat") + path));
        if(!st.st_size) {
            std::fprintf(stderr, "Warning: Empty file found at %s\n", path.data());
            return 0;
        }
        size_t expected_bytes = st.st_size - 8;
        const size_t expected_sketch_nb = ss * sizeof(T);
        if(expected_bytes != expected_sketch_nb) {
            std::fprintf(stderr, "Expected %zu bytes of sketch, found %zu\n", expected_sketch_nb, expected_bytes);
        }
        size_t nb = std::fread(ptr, 1, expected_bytes, fp);
        if(nb != expected_bytes) {
            std::fprintf(stderr, "Read %zu bytes instead of %zu for file %s\n", nb, expected_bytes, path.data());
            perror("Error reading.");
            THROW_EXCEPTION(std::runtime_error("Error in reading from file"));
        }
        sz = expected_bytes / sizeof(T);
    } else {
        auto up = (uint8_t *)ptr;
        for(;!std::feof(fp) && std::fread(up, sizeof(T), chunk_size, fp) == chunk_size; up += chunk_size * sizeof(T));
        sz = (up - (uint8_t *)ptr) / sizeof(T);
    }
    std::fclose(fp);
    return sz;
}*/

std::string FastxSketchingResult::str() const {
    std::string msg = "FastxSketchingResult @" + to_string(this) + ';';
    if(names_.size()) {
        if(names_.size() < 10) {
            for(const auto &n: names_) msg += n + ",";
        }
        msg += to_string(names_.size()) + " names;";
    }
    if(auto pfsz(nperfile_.size()); pfsz > 0) {
        msg += "sketchedbysequence, ";
        msg += to_string(pfsz) + " seqs";
    } else {msg += "sketchbyline";}
    msg += ';';
    if(signatures_.size()) {
        msg += to_string(signatures_.size()) + " signatures;";
    }
    if(kmers_.size()) {
        msg += to_string(kmers_.size()) + " kmers;";
    }
    if(auto kcsz = kmercounts_.size()) {
        msg += to_string(kcsz) + " kmercounts;";
        long double s = 0., ss = 0.;
        for(const auto v: kmercounts_)
            s += v, ss += v * v;
        msg += "mean: ";
        msg += to_string(double(s / kcsz));
        std::cerr << msg << '\n';
        msg = msg + ", std " + to_string(double(std::sqrt(ss / kcsz - std::pow(s / kcsz, 2.))));
        std::cerr << msg << '\n';
    }
    return msg;
}

//Cardinality Estimation
//It looks like this function isn't used anywhere (by global search)
INLINE double compute_cardest(const RegT *ptr, const size_t m) {
    double s = 0.;
#if _OPENMP >= 201307L
    #pragma omp simd reduction(+:s)
#endif
    for(size_t i = 0; i < m; ++i) {
        s += ptr[i];
    }
    DBG_ONLY(std::fprintf(stderr, "Sum manually is %g, compared to accumulate with ld %g. diff: %0.20Lg\n", s, double(std::accumulate(ptr, ptr + m, 0.L)), std::accumulate(ptr, ptr + m, 0.L) - static_cast<long double>(s));)
    return m / s;
}




FastxSketchingResult &fastx2sketch(FastxSketchingResult &ret, Dashing2Options &opts, const std::vector<std::string> &paths, std::string outpath) {
    if(paths.empty()) THROW_EXCEPTION(std::invalid_argument("Can't sketch empty path set"));
    std::vector<std::pair<size_t, uint64_t>> filesizes = get_filesizes(paths);
    const size_t nt = std::max(opts.nthreads(), 1u);
    const size_t ss = opts.sketchsize();
    KSeqHolder kseqs(nt);
    std::vector<BagMinHash> bmhs;
    std::vector<ProbMinHash> pmhs;
    std::vector<OPSetSketch> opss;
    std::vector<FullSetSketch> fss;
    std::vector<OrderMinHash> omhs;
    std::vector<Counter> ctrs;
    std::vector<VSetSketch> cfss;
    static_assert(sizeof(pmhs[0].res_[0]) == sizeof(uint64_t), "Must be 64-bit");
    static_assert(sizeof(bmhs[0].track_ids_[0]) == sizeof(uint64_t), "Must be 64-bit");
    static_assert(sizeof(opss[0].ids()[0]) == sizeof(uint64_t), "Must be 64-bit");
    static_assert(sizeof(fss[0].ids()[0]) == sizeof(uint64_t), "Must be 64-bit");
    auto make = [&](auto &x) {
        x.reserve(nt);
        for(size_t i = 0; i < nt; ++i)
            x.emplace_back(ss);
    };
    //initialize containers for sketches
    auto make_save = [&](auto &x) {
        x.reserve(nt);
        for(size_t i = 0; i < nt; ++i)
            x.emplace_back(ss, opts.save_kmers_, opts.save_kmercounts_);
    };
    //opts.sspace_ defines the type of sketching algorithm used
    if(opts.sspace_ == SPACE_SET) {
        if (verbosity >= Verbosity::DEBUG){
            std::cout << "opts.sspace_ == SPACE_SET" << std::endl;
        }
        if(opts.kmer_result_ == ONE_PERM) {
            if (verbosity >= Verbosity::DEBUG){
                std::cout << "opts.kmer_result_ == ONE_PERM" << std::endl;
            }
            make(opss);
            for(auto &x: opss) x.set_mincount(opts.count_threshold_); //filter out infrequent k-mers
        } else if(opts.kmer_result_ == FULL_SETSKETCH) { //Here the space for the registers is reserved
            if (verbosity >= Verbosity::DEBUG){
                std::cout << "opts.kmer_result_ == FULL_SETSKETCH" << std::endl;
            }
            if(opts.sketch_compressed_set) {
                if (verbosity >= Verbosity::DEBUG){
                    std::cout << "opts.sketch_compressed_set == TRUE" << std::endl;
                }
                cfss.reserve(nt);
                for(size_t i = 0; i < nt; ++i) {
                    if(opts.fd_level_ == .5) {
                        if (verbosity >= Verbosity::DEBUG){
                            std::cout << "NibbleSetS case" << std::endl;
                        }
                        cfss.emplace_back(NibbleSetS(opts.count_threshold_, ss, opts.compressed_b_, opts.compressed_a_));
                    } else if(opts.fd_level_ == 1.) {
                        if (verbosity >= Verbosity::DEBUG){
                            std::cout << "ByteSetS case" << std::endl;
                        }
                        cfss.emplace_back(ByteSetS(opts.count_threshold_, ss, opts.compressed_b_, opts.compressed_a_));
                    } else if(opts.fd_level_ == 2.) {
                        if (verbosity >= Verbosity::DEBUG){
                            std::cout << "ShortSetS case" << std::endl;
                        }
                        cfss.emplace_back(ShortSetS(opts.count_threshold_, ss, opts.compressed_b_, opts.compressed_a_));
                    } else if(opts.fd_level_ == 4.) {
                        if (verbosity >= Verbosity::DEBUG){
                            std::cout << "UintSetS case" << std::endl;
                        }
                        cfss.emplace_back(UintSetS(opts.count_threshold_, ss, opts.compressed_b_, opts.compressed_a_));
                    }
                }
            } else {
                if (verbosity >= Verbosity::DEBUG){
                    std::cout << "opts.sketch_compressed_set = FALSE" << std::endl;
                }
                fss.reserve(nt);
                for(size_t i = 0; i < nt; ++i)
                    fss.emplace_back(opts.count_threshold_, ss, opts.save_kmers_, opts.save_kmercounts_);
                if (verbosity >= Verbosity::DEBUG){
                    std::cout << "after for loop, check if object initialization happens before" << std::endl;
                }
            }
        }
    } else if(opts.sspace_ == SPACE_MULTISET) { 
        if (verbosity >= Verbosity::DEBUG){
            std::cout << "opts.sspace_ == SPACE_MULTISET" << std::endl;
        }
        make_save(bmhs);
    }
    else if(opts.sspace_ == SPACE_PSET) {
        if (verbosity >= Verbosity::DEBUG){
            std::cout << "opts.sspace_ == SPACE_PSET" << std::endl;
        }
        make(pmhs);
    } 
    else if(opts.sspace_ == SPACE_EDIT_DISTANCE) {
        if (verbosity >= Verbosity::DEBUG){
            std::cout << "opts.sspace_ == SPACE_EDIT_DISTANCE" << std::endl;
        }
        if(opts.parse_by_seq_) {
            omhs.reserve(nt);
            for(size_t i = 0; i < nt; omhs.emplace_back(ss, opts.k_), ++i);
        } else {
            THROW_EXCEPTION(std::invalid_argument("Space edit distance is only available in parse-by-seq mode, as it is only defined on strings rather than string collections."));
        }
    }
    while(ctrs.size() < nt) ctrs.emplace_back(opts.cssize()); //The ctrs vector is being populated with Counter objects until its size matches the number of threads (nt). The Counter objects are initialized with the size specified by opts.cssize().
    if (verbosity >= Verbosity::DEBUG){
        std::cout << "About to reset sketch data structures for tid" << std::endl;
    }
#define __RESET(tid) do { \
        if(!opss.empty()) opss[tid].reset();\
        else if(!fss.empty()) fss[tid].reset();\
        else if(!cfss.empty()) std::visit([](auto &x) {x.clear();}, cfss[tid]);\
        else if(!bmhs.empty()) bmhs[tid].reset();\
        else if(!pmhs.empty()) pmhs[tid].reset();\
        /*else if(!omhs.empty()) omhs[tid].reset();*/\
        if(ctrs.size() > unsigned(tid)) ctrs[tid].reset();\
    } while(0)


    //initialization for output to file
    const uint64_t nitems = paths.size();
    std::string kmeroutpath, kmernamesoutpath;
    if(outpath.size() && outpath != "-" && outpath != "/dev/stdout") {
        const size_t offset = sizeof(nitems) * 2 + sizeof(double) * nitems;
        ::truncate(outpath.data(), offset);
        ret.signatures_.assign(outpath, offset);
        if(opts.save_kmers_) {
            kmeroutpath = outpath + ".kmer64";
            kmernamesoutpath = kmeroutpath + ".names.txt";
        }
    }
    //setup output files
    if(kmeroutpath.size()) {
        std::FILE *fp = bfopen(kmeroutpath.data(), "w");
        uint32_t dtype = (uint32_t)opts.input_mode() | (int(opts.canonicalize()) << 8);
        uint32_t sketchsize = opts.sketchsize_;
        uint32_t k = opts.k_;
        uint32_t w = opts.w_ < 0 ? opts.k_: opts.w_;
        checked_fwrite(fp, &dtype, sizeof(dtype));
        checked_fwrite(fp, &sketchsize, sizeof(sketchsize));
        checked_fwrite(fp, &k, sizeof(k));
        checked_fwrite(fp, &w, sizeof(w));
        checked_fwrite(fp, &opts.seedseed_, sizeof(opts.seedseed_));
        if((fp = bfreopen(kmernamesoutpath.data(), "wb", fp)) == 0) THROW_EXCEPTION(std::runtime_error("Failed to open "s + kmernamesoutpath + " for writing."));
        if(bns::filesize(kmeroutpath.data()) != 24) THROW_EXCEPTION(std::runtime_error("kmer out path is the wrong size (expected 16, got "s + std::to_string(bns::filesize(kmeroutpath.data()))));
        static_assert(sizeof(uint32_t) * 4 + sizeof(uint64_t) == 24, "Sanity check");
        ret.kmers_.assign(kmeroutpath, 24);
        for(const auto &n: paths) {
            checked_fwrite(n.data(), 1, n.size(), fp);
            std::fputc('\n', fp);
        }
        std::fclose(fp);
    }
    const int sigshift = opts.sigshift();
    const size_t sigvecsize64 = nitems * ss >> sigshift;
    ret.signatures_.resize(sigvecsize64);
    if(verbosity >= DEBUG && outpath.size()) {
        const size_t offset = sizeof(nitems) * 2 + sizeof(double) * nitems;
        std::fprintf(stderr, "Assigning vector of size %zu to mmap'd file of size %zu with offset %zu\n", ret.signatures_.size(), offset + ret.signatures_.size() * sizeof(RegT), offset);
    }
    if(opts.sspace_ == SPACE_EDIT_DISTANCE) {
        THROW_EXCEPTION(std::runtime_error("edit distance is only available in parse by seq mode"));
    }
    ret.destination_files_.resize(nitems);
    if(opts.save_kmers_) {
        ret.kmerfiles_.resize(nitems);
    }
    if(opts.save_kmercounts_ || opts.kmer_result_ == FULL_MMER_COUNTDICT) {
        ret.kmercountfiles_.resize(nitems);
    }
    ret.cardinalities_.resize(nitems, -1.);
#ifndef NDEBUG
    for(size_t i = 0; i < ret.names_.size(); ++i) {
        std::fprintf(stderr, "name %zu is %s\n", i, ret.names_[i].data());
    }
    std::fprintf(stderr, "kmer result type: %s\n", to_string(opts.kmer_result_).data());
    std::fprintf(stderr, "sketching space type: %s\n", to_string(opts.sspace_).data());
#endif
    // We make an exception for iskmer - we only use this if
    //
    if(opts.save_kmers_ && opts.kmer_result_ != FULL_MMER_SEQUENCE) {
        ret.kmers_.resize(ss * nitems);
    }
    if(opts.save_kmercounts_) {
        ret.kmercounts_.resize(ss * nitems);
    }
    if(opts.kmer_result_ == FULL_MMER_SET) {
        ret.kmerfiles_.resize(ret.destination_files_.size());
    }
    OMP_PFOR_DYN
    for(size_t i = 0; i < nitems; ++i) {
        /* The loop iterates over nitems, which represents the number of items to process.
            In a parallel execution environment (OMP_ONLY), it fetches the current thread ID.
            myind is set based on whether filesizes is provided. This is likely an index into the paths vector.
            mss is calculated as the sketch size (ss) multiplied by myind, determining the starting position in the signatures vector.
            path is the file path to process.
        */
        if (verbosity >= Verbosity::DEBUG) {
            std::cout << "PL: entered the for loop" << std::endl;
        }
        int tid = 0;
        OMP_ONLY(tid = omp_get_thread_num();)
        //const int tid = OMP_ELSE(omp_get_thread_num(), 0);
        //const auto starttime = std::chrono::high_resolution_clock::now();
        auto myind = filesizes.size() ? filesizes[i].second: uint64_t(i);
        const size_t mss = ss * myind;
        auto &path = paths[myind];
        //std::fprintf(stderr, "parsing from path = %s\n", path.data());
        /*Sets up the destination paths for saving the sketch, k-mer counts, and k-mer IDs.
        */
        std::string &destination = ret.destination_files_[myind];
        destination = makedest(opts, path, opts.kmer_result_ == FULL_MMER_COUNTDICT);
        const std::string destination_prefix = destination.substr(0, destination.find_last_of('.'));
        std::string kmer_destination_prefix = makedest(opts, path, true);
        kmer_destination_prefix = kmer_destination_prefix.substr(0, kmer_destination_prefix.find_last_of('.'));
        std::string destkmercounts = destination_prefix + ".kmercounts.f64";
        std::string destkmer = kmer_destination_prefix + ".kmer.u64";
        /*Checks if the sketch, k-mer counts, and k-mer IDs are already cached using check_compressed.
            If caching conditions are met, attempts to load the cached data from the files.
        */
        int dkt, dct, dft;
        bool dkif = check_compressed(destkmer, dkt);
        if (verbosity >= Verbosity::DEBUG){
            std::cout << "PL: dkif = " << dkif << std::endl;
        }
        const bool destisfile = check_compressed(destination, dft);
        if (verbosity >= Verbosity::DEBUG){
            std::cout << "PL: destisfile = " << destisfile << std::endl;
        }
        if(!dkif && opts.kmer_result_ == FULL_MMER_SET && destisfile) {
            dkif = 1; destkmer = destination;
        }
        const bool dkcif = check_compressed(destkmercounts, dct);
        if(ret.kmercountfiles_.size() > myind) ret.kmercountfiles_[myind] = destkmercounts;
        if(opts.cache_sketches_ &&
           (destisfile || (opts.kmer_result_ == FULL_MMER_COUNTDICT && dkif)) &&
           (!opts.save_kmers_ || dkif) &&
           ((!opts.save_kmercounts_ && opts.kmer_result_ != FULL_MMER_COUNTDICT) || dkcif)
        )
        {   //Cache handling logic
            //Load Cached data
            if(opts.kmer_result_ < FULL_MMER_SET) {
                if(ret.signatures_.size()) {
                    if (verbosity >= Verbosity::DEBUG){
                        std::cout << "ret.signatures_.size() = 1" << std::endl;
                    }
                    if(opts.sketch_compressed_set) {
                        if (verbosity >= Verbosity::DEBUG) {
                            std::cout << "PL: opts.sketch_compressed_set = TRUE (cache handling part)" << std::endl;
                        }
                
                        std::FILE *ifp = std::fopen(destination.data(), "rb");
                        std::fread(&ret.cardinalities_[myind], sizeof(double), 1, ifp);
                        std::array<long double, 4> arr;
                        std::fread(arr.data(), sizeof(long double), arr.size(), ifp);
                        auto &[a, b, fd_level, sketchsize] = arr;
                        if(fd_level != opts.fd_level_) {
                            THROW_EXCEPTION(std::runtime_error("fd level mismatch."));
                        }
                        if(sketchsize != ss) {
                            THROW_EXCEPTION(std::runtime_error("sketch size mismatch."));
                        }
                        RegT *const ptr = &ret.signatures_[(ss >> sigshift) * myind];
                        if(std::fread(ptr, sizeof(RegT), ss >> sigshift, ifp) != (ss >> sigshift)) THROW_EXCEPTION(std::runtime_error("Failed to read compressed signatures from file "s + destination));
                        if(std::fgetc(ifp) != EOF) {
                            THROW_EXCEPTION(std::runtime_error("File corrupted - ifp should be at eof."));
                        }
                        std::fclose(ifp);
                    } else {
                        assert(mss + ss <= ret.signatures_.size() || !std::fprintf(stderr, "mss %zu, ss %zu, sig size %zu\n", mss, ss, ret.signatures_.size()));
                        if (verbosity >= Verbosity::DEBUG) {
                            std::cout << "PL: opts.sketch_compressed_set = FALSE (cache handling part)" << std::endl;
                        }
                        //load copy reads sketch data from file and stores register values in ret.signatures, if successfull
                        if(load_copy(destination, &ret.signatures_[mss], &ret.cardinalities_[myind], ss) == 0) {
                            std::fprintf(stderr, "Sketch was not available in file %s... resketching.\n", destination.data());
                            goto perform_sketch;
                        }
                    }
                    //ret.cardinalities_[myind] = compute_cardest(&ret.signatures_[mss], ss);
                    DBG_ONLY(std::fprintf(stderr, "Sketch was loaded from %s and has card %g\n", destination.data(), ret.cardinalities_[myind]);)
                }
                if (verbosity >= Verbosity::DEBUG){
                    std::cout << "PL: About to load cached kmer stuff from file" << std::endl;
                }
                if(ret.kmers_.size()) //load cached kmer IDs from file
                    load_copy(destkmer, &ret.kmers_[mss], &ret.cardinalities_[myind], ss);
                if(ret.kmercounts_.size()) //load cached kmer counts from file
                    load_copy(destkmercounts, &ret.kmercounts_[mss], &ret.cardinalities_[myind], ss);
            } else if(opts.kmer_result_ <= FULL_MMER_SEQUENCE) {
                DBG_ONLY(std::fprintf(stderr, "Cached at path %s, %s, %s\n", destination.data(), destkmercounts.data(), destkmer.data());)
            }
            if(ret.kmerfiles_.size() > myind) {
                ret.kmerfiles_[myind] = destkmer;
            }
            continue;
        } else { //Skip caching
#ifndef NDEBUG
            if (verbosity >= Verbosity::DEBUG){
                std::cout << "PL: Skipped Caching" << std::endl;
            }
            std::fprintf(stderr, "We skipped caching because with %d as cache sketches\n", opts.cache_sketches_);
            std::fprintf(stderr, "destisfile: %d. is countdict %d. is kmerfile %d\n", destisfile, opts.kmer_result_ == FULL_MMER_COUNTDICT, dkif);
            std::fprintf(stderr, "kc save %d, kmer result %s, dkcif %d\n", opts.save_kmercounts_, to_string(opts.kmer_result_).data(), dkcif);
#endif
        }
        perform_sketch:
        __RESET(tid);
        //SKETCH COMPUTATION
        if (verbosity >= Verbosity::DEBUG){
            std::cout << "PL: About to compute sketch computation" << std::endl;
        }
        //Big complicated lambda function -> what does it do??
        auto perf_for_substrs = [&](const auto &func) __attribute__((__always_inline__)) {
            for_each_substr([&](const std::string &subpath) { //iterate over substrings
                auto lfunc = [&](auto x) __attribute__((__always_inline__)) { //lambda function
                    x = maskfn(x);
                    if((!opts.fs_ || !opts.fs_->in_set(x)) && opts.downsample_pass()) func(x);  //Checks if the masked item x is either not in a filter set (fs_) or passes a downsampling check.
                };
                auto lfunc2 = [&func](auto x) __attribute__((__always_inline__)) {func(maskfn(x));};
                const auto seqp = kseqs.kseqs_ + tid;
#define FUNC_FE(f) \
do {\
    if(!opts.fs_ && opts.kmer_downsample_frac_ == 1.) {\
        f(lfunc2, subpath.data(), seqp);\
    } else {\
        f(lfunc, subpath.data(), seqp);\
    } \
} while(0)
                //Hash function part
                if(opts.use128()) {
                    if(unsigned(opts.k_) <= opts.nremperres128()) {
                        if(entmin) {
                            auto encoder(opts.enc_.to_entmin128());
                            FUNC_FE(encoder.for_each);
                        } else {
                            auto encoder(opts.enc_.to_u128());
                            FUNC_FE(encoder.for_each);
                        }
                    } else {
                        FUNC_FE(opts.rh128_.for_each_hash);
                    }
                } else if(unsigned(opts.k_) <= opts.nremperres64()) {
                    if(entmin) {
                        auto encoder(opts.enc_.to_entmin64());
                        FUNC_FE(encoder.for_each);
                    } else {
                        auto encoder(opts.enc_);
                        FUNC_FE(encoder.for_each);
                    }
                } else {
                    FUNC_FE(opts.rh_.for_each_hash);
                }
#undef FUNC_FE
            }, path);
        };
        if( //Part to write to file apparently
            //Processing results and write to file
            (opts.sspace_ == SPACE_MULTISET || opts.sspace_ == SPACE_PSET || opts.kmer_result_ == FULL_MMER_SET || opts.kmer_result_ == FULL_MMER_COUNTDICT)
        )
        {
            if (verbosity >= Verbosity::DEBUG){
                std::cout << "PL: In file writing part" << std::endl;
                std::cout << "PL: opts.sspace_ == SPACE_MULTISET || SPACE_PSET, opts-kmer_result_ == FULL_MMER_SET || FULL_MMER_COUNTDICT" << std::endl;
            }
            auto &ctr = ctrs[tid];
            perf_for_substrs([&ctr](auto x) {ctr.add(x);});
            std::vector<u128_t> kmervec128;
            std::vector<uint64_t> kmervec64;
            std::vector<double> kmerveccounts;
            if(opts.kmer_result_ == FULL_MMER_SET || opts.kmer_result_ == FULL_MMER_COUNTDICT) {
                if(opts.use128())
                    ctr.finalize(kmervec128, kmerveccounts, opts.count_threshold_);
                else
                    ctr.finalize(kmervec64, kmerveccounts, opts.count_threshold_);
                ret.cardinalities_[myind] =
                    opts.kmer_result_ == FULL_MMER_SET ? std::max(kmervec128.size(), kmervec64.size())
                                                       : std::accumulate(kmerveccounts.begin(), kmerveccounts.end(), size_t(0));
            } else if(opts.sspace_ == SPACE_MULTISET) {
                ctr.finalize(bmhs[tid], opts.count_threshold_);
                ret.cardinalities_[myind] = bmhs[tid].total_weight();
                std::copy(bmhs[tid].data(), bmhs[tid].data() + ss, &ret.signatures_[mss]);
            } else if(opts.sspace_ == SPACE_PSET) {
                ctr.finalize(pmhs[tid], opts.count_threshold_);
                std::copy(pmhs[tid].data(), pmhs[tid].data() + ss, &ret.signatures_[mss]);
                ret.cardinalities_[myind] = pmhs[tid].total_weight();
            } else THROW_EXCEPTION(std::runtime_error("Unexpected space for counter-based m-mer encoding"));
                // Make bottom-k if we generated full k-mer sets or k-mer count dictionaries, and copy then over
            if(kmervec64.size() || kmervec128.size()) {
                if(ret.signatures_.size()) {
                    std::vector<BKRegT> keys(ss);
                    double *const kvcp = kmerveccounts.empty() ? static_cast<double *>(nullptr): kmerveccounts.data();
                    if(kmervec128.size()) bottomk(kmervec128, keys, opts.count_threshold_, kvcp);
                    else bottomk(kmervec64, keys, opts.count_threshold_, kvcp);
                    std::copy(keys.begin(), keys.end(), (BKRegT *)&ret.signatures_[mss]);
                }
            }
            //write results to file
            std::FILE * ofp{nullptr};
            if(opts.cache_sketches_ || opts.kmer_result_  == FULL_MMER_SET || opts.kmer_result_ == FULL_MMER_COUNTDICT) {
                std::fprintf(stderr, "Writing saved sketch to %s\n", destination.data());
                ofp = bfopen(destination.data(), "wb");
                if(!ofp) THROW_EXCEPTION(std::runtime_error(std::string("Failed to open std::FILE * at") + destination));
            }
            if(ofp) checked_fwrite(&ret.cardinalities_[myind], sizeof(ret.cardinalities_[myind]), 1, ofp);
            const void *buf = nullptr;
            size_t nb;
            const RegT *srcptr = nullptr;
            if(kmervec128.size()) {
                buf = (const void *)kmervec128.data();
                nb = kmervec128.size() * sizeof(u128_t);
            } else if(kmervec64.size()) {
                buf = (const void *)kmervec64.data();
                nb = kmervec64.size() * sizeof(uint64_t);
            } else if(opts.sspace_ == SPACE_MULTISET) {
                buf = (const void *)bmhs[tid].data();
                nb = ss * sizeof(RegT);
                srcptr = bmhs[tid].data();
            } else if(opts.sspace_ == SPACE_PSET) {
                buf = (const void *)pmhs[tid].data();
                nb = ss * sizeof(RegT);
                srcptr = pmhs[tid].data();
            } else nb = 0, srcptr = nullptr;
            if(srcptr && ret.signatures_.size())
                std::copy(srcptr, srcptr + ss, &ret.signatures_[mss]);
            if(ofp)
                checked_fwrite(ofp, buf, nb);
            if(opts.save_kmers_ && !(opts.kmer_result_ == FULL_MMER_SET || opts.kmer_result_ == FULL_MMER_SEQUENCE || opts.kmer_result_ == FULL_MMER_COUNTDICT)) {
                assert(ret.kmerfiles_.size());
                ret.kmerfiles_[myind] = destkmer;
                const uint64_t *ptr = opts.sspace_ == SPACE_PSET ? pmhs[tid].ids().data():
                                  opts.sspace_ == SPACE_MULTISET ? bmhs[tid].ids().data():
                                  opts.kmer_result_ == ONE_PERM ? opss[tid].ids().data() :
                                  opts.kmer_result_ == FULL_SETSKETCH ? fss[tid].ids().data():
                                      static_cast<uint64_t *>(nullptr);
                if(!ptr) THROW_EXCEPTION(std::runtime_error("This shouldn't happen"));
                DBG_ONLY(std::fprintf(stderr, "Opening destkmer %s\n", destkmer.data());)
                if((ofp = bfreopen(destkmer.data(), "wb", ofp)) == nullptr) THROW_EXCEPTION(std::runtime_error("Failed to write k-mer file"));

                checked_fwrite(ofp, ptr, sizeof(uint64_t) * ss);
                DBG_ONLY(std::fprintf(stderr, "About to copy to kmers of size %zu\n", ret.kmers_.size());)
                if(ret.kmers_.size())
                    std::copy(ptr, ptr + ss, &ret.kmers_[mss]);
            }
            if(opts.save_kmercounts_ || opts.kmer_result_ == FULL_MMER_COUNTDICT) {
                assert(ret.kmercountfiles_.size());
                ret.kmercountfiles_.at(i) = destkmercounts;
                if((ofp = bfreopen(destkmercounts.data(), "wb", ofp)) == nullptr) THROW_EXCEPTION(std::runtime_error("Failed to write k-mer counts"));
                std::vector<double> tmp(ss);
#define DO_IF(x) if(x.size()) {std::copy(x[tid].idcounts().begin(), x[tid].idcounts().end(), tmp.data());}
                if(opts.kmer_result_ == FULL_MMER_COUNTDICT || (opts.kmer_result_ == FULL_MMER_SET && opts.save_kmercounts_)) {
                    DBG_ONLY(std::fprintf(stderr, "kvc size %zu. Writing to file %s\n", kmerveccounts.size(), destkmercounts.data());)
                    tmp = std::move(kmerveccounts);
                } else DO_IF(pmhs) else DO_IF(bmhs) else DO_IF(opss) else DO_IF(fss)
#undef DO_IF
                const size_t nb = tmp.size() * sizeof(double);
                checked_fwrite(ofp, tmp.data(), nb);
                if(ret.kmercounts_.size()) {
                    std::copy(tmp.begin(), tmp.begin() + ss, &ret.kmercounts_[mss]);
                }
            }
            std::fclose(ofp);
        } //handling full_MMER
        else if(opts.kmer_result_ == FULL_MMER_SEQUENCE) {
            if (verbosity >= Verbosity::DEBUG){
                std::cout << "PL: opts.kmer_result_ == FULL_MMER_SEQUENCE" << std::endl;
            }
            ret.kmers_.clear();
            DBG_ONLY(std::fprintf(stderr, "Full mmer sequence\n");)
            std::FILE * ofp;
            if((ofp = bfopen(destination.data(), "wb")) == nullptr) THROW_EXCEPTION(std::runtime_error("Failed to open file for writing minimizer sequence"));
            void *dptr = nullptr;
            size_t m = 1 << 18;
            size_t l = 0;
            if(posix_memalign(&dptr, 16, (1 + opts.use128()) * m * sizeof(uint64_t))) THROW_EXCEPTION(std::bad_alloc());

            perf_for_substrs([&](auto x) {
                using DT = decltype(x);
                DT *ptr = (DT *)dptr;
                if(opts.homopolymer_compress_minimizers_ && l > 0 && ptr[l - 1] == x) return;
                if(l == m) {
                    size_t newm = m << 1;
                    void *newptr = nullptr;
                    if(posix_memalign((void **)&newptr, 16, newm * sizeof(DT))) THROW_EXCEPTION(std::bad_alloc());
                    std::copy(ptr, ptr + m, (DT *)newptr);
                    dptr = newptr;ptr = (DT *)dptr;
                    m = newm;
                }
                ptr[l++] = x;
            });
            assert(dptr);
            checked_fwrite(ofp, dptr, l * (1 + opts.use128()) * sizeof(uint64_t));
            ret.cardinalities_[myind] = l;
            std::free(dptr);
            std::fclose(ofp);
        } //Handling ONE_PERM, FULL_SETSKETCH
        else if(opts.kmer_result_ == ONE_PERM || opts.kmer_result_ == FULL_SETSKETCH) {
            if (verbosity >= Verbosity::DEBUG){
                std::cout << "PL: opts.kmer_result_ == ONE_PERM || FULL_SETSKETCH" << std::endl;
            }
            //Check if caching is enabled and try to read from file
            std::FILE * ofp{nullptr};
            if((opts.cache_sketches_) && (ofp = bfopen(destination.data(), "wb")) == nullptr)
                THROW_EXCEPTION(std::runtime_error(std::string("Failed to open file ") + destination + " for writing sketch."));
            //Check if sketch is empty
            if(opss.empty() && fss.empty() && cfss.empty()) THROW_EXCEPTION(std::runtime_error("Both opss and fss are empty\n"));
            const size_t opsssz = opss.size();
            auto &cret = ret.cardinalities_[myind];
            if(opsssz) { //opsssz stores size of opss 
                if (verbosity >= Verbosity::DEBUG){
                    std::cout << "PL: opsssz = TRUE" << std::endl;
                }
                assert(opss.size() > unsigned(tid)); //assert opps has element at position tid
                assert(opss.at(tid).total_updates() == 0);
                auto p = &opss[tid];
                perf_for_substrs([p](auto hv) {p->update(hv);}); //update sketch by calling update function
                assert(ret.cardinalities_.size() > i);
                cret = p->getcard(); //compute and store cardinality in cret

                if (verbosity >= Verbosity::DEBUG){
                    //mangled type name
                    std::cout << "PL: Type of object at opss[" << tid << "] is " << typeid(opss[tid]).name() << std::endl;

                    //demangle the type name for readability
                    int status;
                    char* realname = abi::__cxa_demangle(typeid(opss[tid]).name(), 0, 0, &status);
                    std::cout << "PL: Demangled type: " << (status == 0 ? realname : typeid(opss[tid]).name()) << std::endl;
                    free(realname); // Free the allocated memory
                }
            } else {
                if(opts.sketch_compressed_set) {
                    if (verbosity >= Verbosity::DEBUG){
                        std::cout << "PL: opts.sketch_compressed_set = TRUE" << std::endl;
                    }
                    //lambda function inside visit
                    std::visit([&](auto &x) { //apply update
                        perf_for_substrs([&x](auto hv) {
                            x.update(hv); //x is reference to sketch at tid
                        });
                        cret = x.cardinality(); //compute and store cardinality in cret

                        if (verbosity >= Verbosity::DEBUG) {
                            int status;
                            char* realname = abi::__cxa_demangle(typeid(x).name(), 0, 0, &status);
                            std::cout << "PL: Type of x is " << (status == 0 ? realname : typeid(x).name()) << std::endl;
                            free(realname); // remember to free the allocated memory
                        }


                    }, cfss.at(tid));
                } else {
                    if (verbosity >= Verbosity::DEBUG){
                        std::cout << "PL: opts.sketch_compressed_set = FALSE" << std::endl;
                    }
                    perf_for_substrs([p=&fss[tid]](auto hv) {p->update(hv);});
                    cret = fss[tid].getcard();
                    if (verbosity >= Verbosity::DEBUG){
                        //mangled type name
                        std::cout << "PL: Type of object at fss[" << tid << "] is " << typeid(fss[tid]).name() << std::endl;

                        //demangle the type name for readability
                        int status;
                        char* realname = abi::__cxa_demangle(typeid(fss[tid]).name(), 0, 0, &status);
                        std::cout << "PL: Demangled type: " << (status == 0 ? realname : typeid(fss[tid]).name()) << std::endl;
                        free(realname); // Free the allocated memory
                    }
                }
            }
            if(ofp) {
                if (verbosity >= Verbosity::DEBUG){
                    std::cout << "PL: ofp = TRUE, writing cardinality = " << cret << " to file" << std::endl;
                    std::cout << "opts.sspace_ = " << opts.sspace_ << std::endl;
                }
                checked_fwrite(ofp, &cret, sizeof(double)); //write cardinality to file
            }
            std::fflush(ofp);
            const uint64_t *ids = nullptr;
            const uint32_t *counts = nullptr;
            // Update this and VSetSketch above to filter down
            const RegT *ptr = opsssz ? opss[tid].data(): fss.size() ? fss[tid].data(): getdata(cfss[tid]);
            assert(ptr);
            //Here comes the part to write registers to file
            if(opts.save_kmers_)
                ids = opsssz ? opss[tid].ids().data(): fss[tid].ids().data();
            if(opts.save_kmercounts_)
                counts = opsssz ? opss[tid].idcounts().data(): fss[tid].idcounts().data();
            if(opts.sketch_compressed_set) {
                if (verbosity >= Verbosity::DEBUG){
                    std::cout << "About to write array with a,b,etc. to file" << std::endl;
                }
                //writes an array containing compressed_a_, compressed_b_, fd_level_, and sketchsize_ to the file. -> IMPORTANT
                std::array<long double, 4> arr{opts.compressed_a_, opts.compressed_b_, static_cast<long double>(opts.fd_level_), static_cast<long double>(opts.sketchsize_)};
                if(ofp) checked_fwrite(arr.data(), sizeof(long double), arr.size(), ofp);
                if(opts.fd_level_ == 0.5) {
                    if (verbosity >= Verbosity::DEBUG){
                        std::cout << "Process data as NibbleSetS" << std::endl;
                    }
                    const uint8_t *srcptr = std::get<NibbleSetS>(cfss[tid]).data();
                    for(size_t i = 0; i < opts.sketchsize_; i += 2) {
                        uint8_t reg = (srcptr[i] << 4) | srcptr[i + 1];
                        if(ofp) checked_fwrite(ptr, sizeof(reg), 1, ofp);
                    }
                } else {
                    if(ofp) checked_fwrite(ptr, sizeof(RegT), ss >> sigshift, ofp);
                }
            } else {
                if (verbosity >= Verbosity::DEBUG){
                        std::cout << "Writing to file UNCOMPRESSED" << std::endl;
                    }
                if(ofp) checked_fwrite(ofp, ptr, ss * sizeof(RegT));
            }
            //Also writing registers to SketchingResult object
            if(ofp) std::fclose(ofp);
            if(ptr && ret.signatures_.size()) {
                if(!opts.sketch_compressed_set) {
                    std::memcpy(&ret.signatures_[mss >> sigshift], ptr, ((ss * sizeof(RegT)) >> sigshift));
                } else {
                    const uint8_t *srcptr = std::get<NibbleSetS>(cfss[tid]).data();
                    uint8_t *destptr = (uint8_t *)&ret.signatures_[mss >> sigshift];
                    for(size_t i = 0; i < opts.sketchsize_; i += 2) {
                        *destptr++ = (srcptr[i] << 4) | srcptr[i + 1];
                    }
                }
            }
            if(ids && ret.kmers_.size())
                std::copy(ids, ids + ss, &ret.kmers_[mss]);
            if(counts && ret.kmercounts_.size())
                std::copy(counts, counts + ss, &ret.kmercounts_[mss]);
        } else THROW_EXCEPTION(std::runtime_error("Unexpected: Not FULL_MMER_SEQUENCE, FULL_MMER_SET, ONE_PERM, FULL_SETSKETCH, SPACE_MULTISET, or SPACE_PSET"));
    } // parallel paths loop
    ret.names_ = paths;
    return ret;
}


void seq_resize(std::vector<std::string>& seqs, const size_t num_seqs) {
    seqs.reserve(num_seqs); // Actually resizing is managed by emplace_back
}
// No-ops
void seq_resize(tmpseq::Seqs&, const size_t) noexcept {
}
void seq_resize(tmpseq::MemoryOrRAMSequences&, const size_t) {

}


} // dashing2
