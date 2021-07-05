#include <atomic>
#include <vector>
#include <cassert>
#include <cstring>
#include <tuple>
#include <string>
#include "flat_hash_map/flat_hash_map.hpp"
#include <stdio.h>


int str2ms(const char *s) {
    int ret =0;
    switch(s[0]) {
        case 'A': case 'a': ret = 0; break;
        case 'C': case 'c': ret = 1; break;
        case 'G': case 'g': ret = 2; break;
        case 'T': case 't': ret = 3; break;
        default: ret = 15;
    }
    ret <<= 4;
    switch(s[1]) {
        case 'A': case 'a': ret |= 0; break;
        case 'C': case 'c': ret |= 1; break;
        case 'G': case 'g': ret |= 2; break;
        case 'T': case 't': ret |= 3; break;
        default: ret |= 15;
    }
    return ret;
}

auto parse_file(std::FILE *ifp) {
    std::fprintf(stderr, "Starting\n");
    std::vector<uint32_t> contigids;
    std::vector<uint32_t> starts, stops;
    std::vector<uint8_t> startms, stopms;
    std::atomic<uint64_t> idcounter;
    std::vector<bool> strand;
    ska::flat_hash_map<std::string, uint32_t> contignames;
    idcounter.store(0);
    char *lptr = nullptr;
    size_t linesz = 0;
    std::string cname;
    size_t ln = 0;
    ssize_t rc;
    for(;
        //std::feof(ifp)
        ;++ln) {
        if((rc = ::getline(&lptr, &linesz, ifp)) < 0) throw 1;
        if(ln % 1 == 0) std::fprintf(stderr, "Processed %zu lines, last rc is %zd\n", ln, rc);
        std::fputs(lptr, stdout);
        const uint64_t myid = idcounter++;
        
        char *p = std::strchr(lptr, '\t');
        assert(p);
        char *p2 = std::strchr(p + 1, '\t');
        assert(p2);
        if((*p == 'C' || *p == 'c') && p[1] == 'h' && p[2] == 'r')
            p += 3;
        cname.assign(p, p2);
        uint32_t mycid = cname.size();
        auto cnit = contignames.find(cname);
        if(cnit != contignames.end()) mycid = cnit->second;
        else contignames.emplace(cname, mycid);
        contigids.push_back(mycid);
        p = std::strchr(p2 + 1, '\t');
        ssize_t srpos = std::strtoll(p, &p, 10);
        ssize_t sppos = std::strtoll(p, &p, 10);
        std::fprintf(stderr, "poses: %zd/%zd\n", srpos, sppos);
        p = std::strchr(p + 1, '\t'); // Skip the length field
        assert(p[1] == '+' || p[1] == '-' || !std::fprintf(stderr, "p: %s, %d\n", p, p[1]));
        strand.push_back(p[1] == '+');
        p += 3;
        startms.push_back(str2ms(p));
        p = std::strchr(p, '\t') + 1;
        std::fprintf(stderr, "p for ms: %s\n", p);
        stopms.push_back(str2ms(p));
        p = std::strchr(std::strchr(p, '\t') + 1, '\t') + 1; // Skip two fields
        std::fprintf(stderr, "P: '%s'\n", p);
        std::exit(1);
    }
    return std::make_tuple(contigids, starts, stops, startms, stopms, strand, contignames);
}

int main(int argc, char **argv) {
    std::FILE *ifp = std::fopen(argc == 1 ? "/dev/stdin": argv[1], "r");
    auto [cids, starts, stops, startms, stopms, strand, contignames] = parse_file(ifp);
    std::fclose(ifp);
    return 0;
}
