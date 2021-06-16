#pragma once
#include "btree/map.h"
#include "enums.h"
#include "sketch/hash.h"
#include "sketch/div.h"
#include "sketch/common.h"

namespace dashing2 {

namespace hash = sketch::hash;
template<typename T, size_t pow2=false, typename Hasher = hash::MultiplyAddXoRot<31>>
struct LazyOnePermSetSketch {
private:
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>, "Must be integral and unsigned");
    size_t m_;
    // Solution: hash reversibly, track the maximum IDs
    std::vector<T> registers_;
    static_assert(std::is_integral_v<T> || std::is_same_v<T, u128_t>, "LazyOnePermSetSketch is to be used with integral types");
    std::vector<double> counts_;
    using SigT = std::conditional_t<(sizeof(T) == 4), float, std::conditional_t<(sizeof(T) == 8), double, long double>>;
    std::unique_ptr<std::vector<SigT>> as_sigs_;
    std::unique_ptr<std::vector<uint64_t>> original_ids_;
    std::unique_ptr<std::vector<uint32_t>> idcounts_;
    size_t mask_;
    int shift_;
    double count_threshold_;
    uint64_t total_updates_ = 0;
    Hasher hasher_;
    // MultiplyAddXoRot
    // is already enough to pass Rabbit/SmallCrush
    schism::Schismatic<uint64_t> div_;
    double mincount_ = 0.;
    std::vector<btree::map<T, uint32_t>> potentials_;
    double card_ = -1.;
public:
    LazyOnePermSetSketch(const LazyOnePermSetSketch &o) = default;
    LazyOnePermSetSketch& operator=(const LazyOnePermSetSketch &o) = default;
    LazyOnePermSetSketch& operator=(LazyOnePermSetSketch &&o) = default;
    LazyOnePermSetSketch(LazyOnePermSetSketch &&) = default;
    LazyOnePermSetSketch(size_t m, uint64_t seed=0x321b919a61cb41f7ul): hasher_(seed), div_(m) {
        if(pow2)
            m = sketch::integral::roundup(m);
        else if(m & 1) ++m;
        m_ = m;
        registers_.resize(m_, T(0));
        counts_.resize(m_);
        div_ = schism::Schismatic<uint64_t>(m_);
        mask_ = m_ - 1;
        shift_ = sketch::integral::ilog2(m_);
        reset();
    }
    void set_mincount(double v) {
        if(v > 0.) {
            mincount_ = v;
            potentials_.resize(size());
        }
    }
    template<typename O>
    INLINE void update(T id,  O) {update(id);}
    T decode(T x) const {
        return hasher_.inverse(x);
    }
    INLINE void update(const T oid) {
        ++total_updates_;
        const T id = hasher_(oid);
        size_t idx;
        if constexpr(pow2) {
            idx = hasher_(id) & mask_;
        } else {
            auto hid = hasher_(id);
            auto di = div_.div(hid);
            auto mo = hid - m_ * di;
            assert(di == (hid / m_));
            assert(mo == (hid % m_));
            assert(di < hid);
            idx = mo;
            //std::fprintf(stderr, "di = %zu, mo = %zu, idx = %zu, id = %zu\n", size_t(di), size_t(mo), size_t(idx), size_t(id));
        }
        assert(idx < size());
        auto &cref = counts_[idx];
        auto &rref = registers_[idx];
        if(mincount_ > 0.) {
            // If mincount > 0, then
            if(rref > id) {
                auto &pos = potentials_[idx];
                auto it = pos.find(id);
                if(it == pos.end()) it = pos.emplace(id, 1).first;
                else ++it->second;
                if(it->second >= mincount_) {
                    rref = id;
                    cref = it->second;
                    for(auto pit = pos.begin(); pit != pos.end();) {
                        if(pit->first >= id)
                        pit = pos.erase(pit);
                        else ++pit;
                    }
                    return;
                }
            } else cref += (rref == id);
        } else {
            if(rref > id) {
                rref = id; cref = 1.;
            } else cref += (rref == id);
        }
    }
#if 0
    size_t total_updates() const;
    std::vector<uint32_t> &idcounts();
    std::vector<uint64_t> &ids();
    static constexpr size_t get_modv();
    size_t size() const;
    SigT *data();
    void reset();
    double getcard();
#endif

    static constexpr long double omul =
        sizeof(T) == 16 ? 0x1p-128L:
        sizeof(T) == 8 ? 0x1p-64L:
        sizeof(T) == 4 ? 0x1p-32L:
        sizeof(T) == 2 ? 0x1p-16L:
        sizeof(T) == 1 ? 0x1p-8L: 0.L;
    static_assert(omul != 0.L, "sanity check");
    template<typename T2=SigT>
    std::vector<T2> to_sigs() const {
        std::vector<T2> ret(size());
        std::transform(registers_.begin(), registers_.end(), ret.begin(), [sz2=size()/2](auto x) -> T2 {
            if(std::is_integral_v<T2>) {
                return x; // save as truncation/min hash value by twiddling
            } else {
                return -std::log((get_modv() - x) * omul);
            }
        });
        return ret;
    }
    void reset() {
        std::fill_n(registers_.data(), registers_.size(), std::numeric_limits<T>::max());
        std::memset(counts_.data(), 0, counts_.size() * sizeof(double));
        as_sigs_.reset();
        card_ = -1.;
        for(auto &p: potentials_)
            p.clear();
    }
    double getcard() {
        if(card_ > 0.) return card_;
        //std::fprintf(stderr, "size: %zu\n", sz);
        long double sum = std::accumulate(registers_.begin(), registers_.end(), 0.L,
            [](auto x, auto y) {return x + (y ? y * omul: 0.L);}
        );
        return card_ = sum ? double(std::pow(m_, 2) / sum): std::numeric_limits<double>::infinity();
    }
    SigT *data() {
        if(as_sigs_) return as_sigs_->data();
        as_sigs_.reset(new std::vector<SigT>(registers_.size()));
        auto asp = as_sigs_->data();
        const long double mul = -SigT(1) / (m_ - std::count(registers_.begin(), registers_.end(), std::numeric_limits<T>::max()));
        for(size_t i = 0; i < m_; ++i) {
            //const auto lv = registers_[i];
            if(registers_[i] == std::numeric_limits<T>::max()) continue;
            asp[i] = mul * std::log(omul * (std::numeric_limits<T>::max() - registers_[i] + 1));
        }
        return asp;
    }
    std::vector<uint64_t> &ids() {
        auto p = new std::vector<uint64_t>(registers_.size());
        original_ids_.reset(p);
        std::transform(registers_.begin(), registers_.end(), p->begin(), [this](T x) {return this->decode(x);});
        return *p;
    }
    std::vector<uint32_t> &idcounts() {
        auto p = new std::vector<uint32_t>(size());
        idcounts_.reset(p);
        std::copy(counts_.begin(), counts_.end(), p->data());
        return *p;
    }
    static constexpr size_t get_modv() {return std::numeric_limits<T>::max();}
    size_t total_updates() const {return total_updates_;}
    size_t size() const {return m_;}
};
} // namespace dashing2
