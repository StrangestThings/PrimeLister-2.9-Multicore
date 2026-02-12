#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <charconv>
#include <cstring>
#include <thread>
#include <array>

// ---- OpenMP Setup ----
#ifdef _OPENMP
#include <omp.h>
#else
#include <chrono>
inline double omp_get_wtime() {
    static const auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
inline int omp_get_num_procs() { return (int)std::thread::hardware_concurrency(); }
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#endif

using u64 = unsigned long long;

// ---------- Optimierte Helfer ----------
static inline u64 ceil_div_u64(u64 a, u64 b) {
    return (a + b - 1) / b;
}

static inline unsigned ctz64(uint64_t x) {
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (unsigned)idx;
#else
    return (unsigned)__builtin_ctzll(x);
#endif
}

// ---------- Datei-Ausgabe ----------
static inline void buf_append_u64(std::string& buf, u64 x) {
    char tmp[32];
    auto res = std::to_chars(tmp, tmp + 32, x);
    buf.append(tmp, (size_t)(res.ptr - tmp));
}

static void write_prime_table(const std::vector<u64>& P, u64 cols) {
    std::ofstream out("PrimeList.txt", std::ios::out | std::ios::binary);
    if (!out) return;

    std::string buf;
    buf.reserve(16 * 1024 * 1024);

    u64 count = 0;
    for (size_t i = 0; i < P.size(); ++i) {
        if (count == 0) {
            buf_append_u64(buf, (u64)(i + 1));
            buf.push_back('-');
            buf_append_u64(buf, (u64)(std::min)((size_t)(i + cols), P.size()));
            buf.push_back('\t');
        }
        buf_append_u64(buf, P[i]);
        buf.push_back('\t');
        if (++count == cols) {
            buf.push_back('\n');
            count = 0;
        }
        if (buf.size() >= 15 * 1024 * 1024) {
            out.write(buf.data(), (std::streamsize)buf.size());
            buf.clear();
        }
    }
    if (count != 0) buf.push_back('\n');
    out.write(buf.data(), (std::streamsize)buf.size());
}

// ---------- Kern-Algorithmus v2.9 (HYBRID) ----------
static std::vector<u64> primes_formeln(u64 N, u64 cols, bool silent) {
    u64 bereich = (N + 2ull) / 3ull + (15ull * (u64)(std::max)((size_t)1, (size_t)cols));
    if (bereich == 0) return {};

    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif

    // ADAPTIVE: mehr Segmente für besseres Load-Balancing bei großen N
    u64 target_segments = (N > 1000000000ull) ? ((u64)T * 6ull) : ((u64)T * 4ull);
    u64 SEG_IDX = (bereich + target_segments - 1) / target_segments;

    // Cache-optimierte Grenzen
    const u64 MIN_SEG = 1ull << 23;  // 1 MB (kleiner für kleine N)
    const u64 MAX_SEG = 1ull << 27;  // 16 MB
    if (SEG_IDX < MIN_SEG) SEG_IDX = MIN_SEG;
    if (SEG_IDX > MAX_SEG) SEG_IDX = MAX_SEG;
    if (SEG_IDX > bereich) SEG_IDX = bereich;

    u64 nsegs = (bereich + SEG_IDX - 1) / SEG_IDX;
    const long long M = (long long)bereich;

    long long end1 = (M - 11) / 7;
    if (end1 < 0) end1 = -1;

    long long end2 = (M <= 4) ? -1 : (long long)std::floor((-8.0L + std::sqrt(16.0L + 12.0L * (long double)M)) / 6.0L);
    long long end3 = (M <= 7) ? -1 : (long long)std::floor((-10.0L + std::sqrt(16.0L + 12.0L * (long double)M)) / 6.0L);

    std::vector<std::vector<u64>> buckets((size_t)nsegs);

    double density = 1.0 / std::log((double)N + 1.0);
    size_t estimated_per_seg = (size_t)(SEG_IDX * 3.0 * density * 1.15);
    for (auto& b : buckets) {
        b.reserve(estimated_per_seg);
    }

#pragma omp parallel
    {
        std::vector<uint64_t> mark_words;
        mark_words.reserve((size_t)((SEG_IDX + 63ull) >> 6));

#pragma omp for schedule(guided, 1)
        for (long long s = 0; s < (long long)nsegs; ++s) {
            u64 base = (u64)s * SEG_IDX;
            u64 end = (u64)(std::min)((size_t)bereich, (size_t)(base + SEG_IDX));
            size_t seg_len = (size_t)(end - base);
            size_t nwords = (seg_len + 63u) >> 6;

            mark_words.resize(nwords);
            std::memset(mark_words.data(), 0, nwords * sizeof(uint64_t));

            long long Mend = (long long)end;
            long long e1seg = (long long)(std::min)((size_t)end1, (size_t)((Mend - 11) / 7));
            long long e2seg = (Mend <= 4) ? -1 : (long long)(std::min)((size_t)end2, (size_t)std::floor((-8.0L + std::sqrt(16.0L + 12.0L * (long double)Mend)) / 6.0L));
            long long e3seg = (Mend <= 7) ? -1 : (long long)(std::min)((size_t)end3, (size_t)std::floor((-10.0L + std::sqrt(16.0L + 12.0L * (long double)Mend)) / 6.0L));

#define SETBIT(idx) mark_words[(idx) >> 6] |= (1ull << ((idx) & 63))

            u64 r10 = base % 10ull;
            u64 r14 = base % 14ull;

            u64 i1 = (r10 == 0 ? base : base + (10 - r10));
            if (i1 == 0 && base == 0) i1 = 10;
            for (u64 i = i1; i < end; i += 10) SETBIT((size_t)(i - base));

            for (u64 i = base + (r10 <= 7 ? 7 - r10 : 17 - r10); i < end; i += 10)
                SETBIT((size_t)(i - base));

            for (u64 i = base + (r14 <= 10 ? 10 - r14 : 24 - r14); i < end; i += 14)
                SETBIT((size_t)(i - base));

            u64 i14 = base + (r14 <= 1 ? 1 - r14 : 15 - r14);
            if (i14 == 1 && base == 0) i14 = 15;
            for (u64 i = i14; i < end; i += 14) SETBIT((size_t)(i - base));

            // Formel 1: Simpler Loop (kein Unrolling bei großem Overhead)
            for (long long i = 0; i <= e1seg; i += 2) {
                u64 iu = (u64)i;
                u64 step = 6ull * iu + 10ull, j0 = 7ull * iu + 10ull;
                if (j0 < base) j0 += ceil_div_u64(base - j0, step) * step;
                for (u64 j = j0; j < end; j += step) SETBIT((size_t)(j - base));
            }

            // Formel 2
            for (long long i = 1; i <= e2seg; i += 2) {
                u64 iu = (u64)i;
                u64 step = 6ull * iu + 8ull;
                u64 j0 = 3ull * iu * iu + 8ull * iu + 4ull;
                if (j0 < base) j0 += ceil_div_u64(base - j0, step) * step;
                for (u64 j = j0; j < end; j += step) SETBIT((size_t)(j - base));
            }

            // Formel 3: Branch-Optimierung
            for (long long i = 0; i <= e3seg; i += 2) {
                u64 iu = (u64)i;
                u64 step = 6ull * iu + 10ull;
                u64 idx = 3ull * iu * iu + 10ull * iu + 7ull;

                if (idx < base) {
                    u64 j0 = idx + ceil_div_u64(base - idx, step) * step;
                    for (u64 j = j0; j < end; j += step) SETBIT((size_t)(j - base));
                }
                else if (idx < end) {
                    SETBIT((size_t)(idx - base));
                    for (u64 j = idx + step; j < end; j += step) SETBIT((size_t)(j - base));
                }
            }

#undef SETBIT

            std::vector<u64>& local = buckets[(size_t)s];

            for (size_t wi = 0; wi < nwords; ++wi) {
                uint64_t inv = ~mark_words[wi];
                while (inv) {
                    unsigned b = ctz64(inv);
                    size_t off = (wi << 6) + b;
                    if (off >= seg_len) break;
                    u64 idx_v = base + off;
                    u64 p = ((idx_v & 1ull) == 0ull) ? (3ull * idx_v + 5ull) : (3ull * idx_v + 4ull);
                    if (p <= N) local.push_back(p);
                    inv &= (inv - 1ull);
                }
            }
        }
    }

    std::vector<u64> primes;
    size_t total_p = 1 + (N >= 3 ? 1 : 0);
    for (const auto& b : buckets) total_p += b.size();
    primes.reserve(total_p);

    primes.push_back(2);
    if (N >= 3) primes.push_back(3);

    for (const auto& b : buckets) {
        primes.insert(primes.end(), b.begin(), b.end());
    }

    return primes;
}

struct ExperimentConfig {
    const char* name;
#ifdef _OPENMP
    omp_sched_t schedule;
#endif
    int chunk;
    u64 seg_override;
    bool run;
};

static std::vector<u64> primes_formeln_cfg(u64 N, u64 cols, bool silent,
#ifdef _OPENMP
    omp_sched_t sched,
#endif
    int chunk,
    u64 seg_override)
{
    u64 bereich = (N + 2ull) / 3ull + (15ull * (u64)(std::max)((size_t)1, (size_t)cols));
    if (bereich == 0) return {};

    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
    omp_set_schedule(sched, chunk);
#endif

    u64 target_segments = (N > 1000000000ull) ? ((u64)T * 6ull) : ((u64)T * 4ull);
    u64 SEG_IDX = (bereich + target_segments - 1) / target_segments;
    const u64 MIN_SEG = 1ull << 23;
    const u64 MAX_SEG = 1ull << 27;
    if (SEG_IDX < MIN_SEG) SEG_IDX = MIN_SEG;
    if (SEG_IDX > MAX_SEG) SEG_IDX = MAX_SEG;
    if (seg_override != 0ull) SEG_IDX = seg_override;
    if (SEG_IDX > bereich) SEG_IDX = bereich;

    u64 nsegs = (bereich + SEG_IDX - 1) / SEG_IDX;
    const long long M = (long long)bereich;

    long long end1 = (M - 11) / 7;
    if (end1 < 0) end1 = -1;
    long long end2 = (M <= 4) ? -1 : (long long)std::floor((-8.0L + std::sqrt(16.0L + 12.0L * (long double)M)) / 6.0L);
    long long end3 = (M <= 7) ? -1 : (long long)std::floor((-10.0L + std::sqrt(16.0L + 12.0L * (long double)M)) / 6.0L);

    std::vector<std::vector<u64>> buckets((size_t)nsegs);
    double density = 1.0 / std::log((double)N + 1.0);
    size_t estimated_per_seg = (size_t)(SEG_IDX * 3.0 * density * 1.15);
    for (auto& b : buckets) b.reserve(estimated_per_seg);

#pragma omp parallel
    {
        std::vector<uint64_t> mark_words;
        mark_words.reserve((size_t)((SEG_IDX + 63ull) >> 6));

#pragma omp for schedule(runtime)
        for (long long s = 0; s < (long long)nsegs; ++s) {
            u64 base = (u64)s * SEG_IDX;
            u64 end = (u64)(std::min)((size_t)bereich, (size_t)(base + SEG_IDX));
            size_t seg_len = (size_t)(end - base);
            size_t nwords = (seg_len + 63u) >> 6;

            mark_words.resize(nwords);
            std::memset(mark_words.data(), 0, nwords * sizeof(uint64_t));

            long long Mend = (long long)end;
            long long e1seg = (long long)(std::min)((size_t)end1, (size_t)((Mend - 11) / 7));
            long long e2seg = (Mend <= 4) ? -1 : (long long)(std::min)((size_t)end2, (size_t)std::floor((-8.0L + std::sqrt(16.0L + 12.0L * (long double)Mend)) / 6.0L));
            long long e3seg = (Mend <= 7) ? -1 : (long long)(std::min)((size_t)end3, (size_t)std::floor((-10.0L + std::sqrt(16.0L + 12.0L * (long double)Mend)) / 6.0L));

#define SETBIT(idx) mark_words[(idx) >> 6] |= (1ull << ((idx) & 63))
            u64 r10 = base % 10ull;
            u64 r14 = base % 14ull;

            u64 i1 = (r10 == 0 ? base : base + (10 - r10));
            if (i1 == 0 && base == 0) i1 = 10;
            for (u64 i = i1; i < end; i += 10) SETBIT((size_t)(i - base));
            for (u64 i = base + (r10 <= 7 ? 7 - r10 : 17 - r10); i < end; i += 10) SETBIT((size_t)(i - base));
            for (u64 i = base + (r14 <= 10 ? 10 - r14 : 24 - r14); i < end; i += 14) SETBIT((size_t)(i - base));

            u64 i14 = base + (r14 <= 1 ? 1 - r14 : 15 - r14);
            if (i14 == 1 && base == 0) i14 = 15;
            for (u64 i = i14; i < end; i += 14) SETBIT((size_t)(i - base));

            for (long long i = 0; i <= e1seg; i += 2) {
                u64 iu = (u64)i;
                u64 step = 6ull * iu + 10ull, j0 = 7ull * iu + 10ull;
                if (j0 < base) j0 += ceil_div_u64(base - j0, step) * step;
                for (u64 j = j0; j < end; j += step) SETBIT((size_t)(j - base));
            }
            for (long long i = 1; i <= e2seg; i += 2) {
                u64 iu = (u64)i;
                u64 step = 6ull * iu + 8ull;
                u64 j0 = 3ull * iu * iu + 8ull * iu + 4ull;
                if (j0 < base) j0 += ceil_div_u64(base - j0, step) * step;
                for (u64 j = j0; j < end; j += step) SETBIT((size_t)(j - base));
            }
            for (long long i = 0; i <= e3seg; i += 2) {
                u64 iu = (u64)i;
                u64 step = 6ull * iu + 10ull;
                u64 idx = 3ull * iu * iu + 10ull * iu + 7ull;
                if (idx < base) {
                    u64 j0 = idx + ceil_div_u64(base - idx, step) * step;
                    for (u64 j = j0; j < end; j += step) SETBIT((size_t)(j - base));
                }
                else if (idx < end) {
                    SETBIT((size_t)(idx - base));
                    for (u64 j = idx + step; j < end; j += step) SETBIT((size_t)(j - base));
                }
            }
#undef SETBIT

            std::vector<u64>& local = buckets[(size_t)s];
            for (size_t wi = 0; wi < nwords; ++wi) {
                uint64_t inv = ~mark_words[wi];
                while (inv) {
                    unsigned b = ctz64(inv);
                    size_t off = (wi << 6) + b;
                    if (off >= seg_len) break;
                    u64 idx_v = base + off;
                    u64 p = ((idx_v & 1ull) == 0ull) ? (3ull * idx_v + 5ull) : (3ull * idx_v + 4ull);
                    if (p <= N) local.push_back(p);
                    inv &= (inv - 1ull);
                }
            }
        }
    }

    std::vector<u64> primes;
    size_t total_p = 1 + (N >= 3 ? 1 : 0);
    for (const auto& b : buckets) total_p += b.size();
    primes.reserve(total_p);
    primes.push_back(2);
    if (N >= 3) primes.push_back(3);
    for (const auto& b : buckets) primes.insert(primes.end(), b.begin(), b.end());
    return primes;
}

static void run_experiment_matrix(u64 testN, u64 cols) {
    std::cout << "\n[Matrix] Priorisierte Experimente (selbe Formeln, anderer Laufmodus)\n";
#ifdef _OPENMP
    std::array<ExperimentConfig, 8> cfgs = {{
        {"A0 baseline guided/chunk1, auto-seg", omp_sched_guided, 1, 0ull, true},
        {"A1 guided/chunk4, auto-seg", omp_sched_guided, 4, 0ull, true},
        {"A2 dynamic/chunk4, auto-seg", omp_sched_dynamic, 4, 0ull, true},
        {"A3 static/chunk1, auto-seg", omp_sched_static, 1, 0ull, true},
        {"B1 guided/chunk4, seg=8 Mi", omp_sched_guided, 4, 1ull << 23, true},
        {"B2 dynamic/chunk4, seg=16 Mi", omp_sched_dynamic, 4, 1ull << 24, true},
        {"B3 static/chunk1, seg=32 Mi", omp_sched_static, 1, 1ull << 25, true},
        {"B4 guided/chunk8, seg=64 Mi", omp_sched_guided, 8, 1ull << 26, true}
    }};

    struct Row { const char* name; double sec; size_t cnt; };
    std::vector<Row> rows;
    rows.reserve(cfgs.size());

    for (const auto& c : cfgs) {
        if (!c.run) continue;
        double t0 = omp_get_wtime();
        auto p = primes_formeln_cfg(testN, cols, true, c.schedule, c.chunk, c.seg_override);
        double sec = omp_get_wtime() - t0;
        rows.push_back({ c.name, sec, p.size() });
        std::cout << "  " << std::left << std::setw(36) << c.name << "  "
                  << std::fixed << std::setprecision(4) << sec << " s"
                  << "  primes=" << p.size() << "\n";
    }

    auto best = std::min_element(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.sec < b.sec; });
    if (best != rows.end()) {
        std::cout << "  -> BEST: " << best->name << " mit " << std::fixed << std::setprecision(4) << best->sec << " s\n";
    }
#else
    (void)testN; (void)cols;
    std::cout << "  OpenMP nicht aktiv, Matrix ist in diesem Build nicht verfuegbar.\n";
#endif
}

// ---------- MAIN ----------
int main() {
    std::ios::sync_with_stdio(false); std::cin.tie(NULL);

#if defined(_WIN32)
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif

    std::cout << "***************************************************\n";
    std::cout << "*  PrimeLister v2.9 Hybrid-Optimiert (2026)       *\n";
    std::cout << "***************************************************\n\n";

    int max_t = omp_get_num_procs();
    int best_t = 1; double min_t = 1e30;
    u64 calib_N = 1000000000;

    std::cout << "[Step 1] Kalibriere Hardware mit N = 1.000.000.000...\n";
    std::vector<int> test_configs = { 1, max_t / 4, max_t / 2, (3 * max_t) / 4, max_t };

    for (int t : test_configs) {
        if (t < 1 || t > max_t) continue;
#ifdef _OPENMP
        omp_set_num_threads(t);
#endif
        double start = omp_get_wtime();
        auto d = primes_formeln(calib_N, 1, true);
        double end = omp_get_wtime() - start;
        std::cout << "  - Threads " << std::setw(2) << t << ": "
            << std::fixed << std::setprecision(4) << end << "s\n";
        if (end < min_t) { min_t = end; best_t = t; }
    }
#ifdef _OPENMP
    omp_set_num_threads(best_t);
#endif
    std::cout << ">> Optimal: " << best_t << " Threads ("
        << std::fixed << std::setprecision(2) << (min_t * 1000) << " ms).\n\n";

    u64 N; u64 cols;
    std::cout << "[Step 2] Parameter eingeben\nLimit N: "; std::cin >> N;
    std::cout << "Spalten: "; std::cin >> cols;

    char do_matrix = 'n';
    std::cout << "Experiment-Matrix laufen lassen? (y/n): "; std::cin >> do_matrix;
    if (do_matrix == 'y' || do_matrix == 'Y') {
        u64 testN = (N < 100000000ull) ? N : 100000000ull;
        run_experiment_matrix(testN, cols);
    }

    std::cout << "\n[Step 3] Hauptberechnung laeuft...\n";
    double t0 = omp_get_wtime();
    auto P = primes_formeln(N, cols, false);
    double t1 = omp_get_wtime();

    std::cout << "\nFirst 100 primes:\n";
    size_t limit_first = (P.size() < 100) ? P.size() : 100;
    for (size_t i = 0; i < limit_first; ++i) std::cout << P[i] << ' ';

    std::cout << "\n\nLast 10 primes under limit:\n";
    size_t start_last = (P.size() > 10) ? (P.size() - 10) : 0;
    for (size_t i = start_last; i < P.size(); ++i) std::cout << P[i] << ' ';

    std::cout << "\n\nOverall primes found: " << P.size() << "\n";
    std::cout << "Overall calculation time [s]: " << std::fixed << std::setprecision(6) << (t1 - t0) << "\n";

    if (t1 - t0 > 0) {
        double rate = (double)P.size() / (t1 - t0);
        std::cout << "~ " << (u64)rate << " primes/s\n";
    }

    std::cout << "\nWriting PrimeList.txt ...\n";
    double tw0 = omp_get_wtime();
    write_prime_table(P, cols);
    double tw1 = omp_get_wtime();
    std::cout << "Write time: " << std::fixed << std::setprecision(3) << (tw1 - tw0) << "s\n";

    std::cout << "\nDone. Press Enter to exit...";
    std::cin.ignore(1000, '\n'); std::cin.get();
    return 0;
}
