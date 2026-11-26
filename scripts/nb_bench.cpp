// Timing / smoke test for an Acuity-built .nb on the Allwinner A733 NPU (VIPLite via the awnn_inference wrapper).
// Fills the inputs with random bytes (or a raw file per input), runs N times, prints latency stats,
// and optionally dumps every output tensor as float32 (output 0 -> out.bin, output 1 -> out.bin.1, ...).
//   nb_bench -m model.nb [-n 50] [-i0 in0.bin] [-i1 in1.bin] [-o out.bin] [-hz R] [-t SECONDS]
//   -hz R : pace runs at R per second (sleep between runs); -t S : run for S seconds instead of -n runs
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <cmath>
#include "awnn_inference/session.h"

static bool read_file(const std::string& p, std::vector<unsigned char>* d) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    d->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

int main(int argc, char** argv) {
    std::string model, in[2], out;
    int n = 50; double hz = 0, tsec = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i0") && i + 1 < argc) in[0] = argv[++i];
        else if (!strcmp(argv[i], "-i1") && i + 1 < argc) in[1] = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "-hz") && i + 1 < argc) hz = atof(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) tsec = atof(argv[++i]);
    }
    if (model.empty()) { printf("usage: nb_bench -m model.nb [-n N] [-i0 f] [-i1 f] [-o out.bin]\n"); return 1; }

    awnn::Session s;
    auto t0 = std::chrono::steady_clock::now();
    if (!s.load(model)) { printf("load failed\n"); return 2; }
    double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("loaded in %.1f ms, inputs=%d outputs=%d\n", load_ms, s.input_count(), s.output_count());

    srand(1);
    // the wrapper's input_count() only probes input 0, so try indices 0..1 directly
    for (int k = 0; k < 2; ++k) {
        awnn::BufferView v;
        if (!s.input_buffer(k, &v)) { if (k == 0) { printf("no input 0\n"); return 3; } break; }
        printf("input[%d]: %u bytes\n", k, v.bytes);
        std::vector<unsigned char> d;
        if (k < 2 && !in[k].empty() && read_file(in[k], &d)) {
            memcpy(v.data, d.data(), std::min<size_t>(d.size(), v.bytes));
            printf("  filled from %s (%zu bytes)\n", in[k].c_str(), d.size());
        } else {
            for (unsigned i = 0; i < v.bytes; ++i) ((unsigned char*)v.data)[i] = rand() & 0xff;
        }
    }

    std::vector<double> run, total;
    awnn::RunProfile p;
    for (int i = 0; i < 3; ++i) if (!s.run(&p)) { printf("warmup run failed\n"); return 4; }   // warmup
    auto tstart = std::chrono::steady_clock::now();
    auto el = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - tstart).count(); };
    for (int i = 0; tsec > 0 ? el() < tsec : i < n; ++i) {
        if (hz > 0) {   // wait for this run's slot
            double due = i / hz, now = el();
            if (due > now) std::this_thread::sleep_for(std::chrono::duration<double>(due - now));
        }
        if (!s.run(&p)) { printf("run failed at %d\n", i); return 4; }
        run.push_back(p.run_ms); total.push_back(p.total_ms);
    }
    double dur = el();
    printf("completed %zu runs in %.1f s = %.2f runs/s\n", run.size(), dur, run.size() / dur);
    auto stat = [](std::vector<double> v, const char* name) {
        std::sort(v.begin(), v.end());
        double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        printf("%-6s mean %.2f ms  p50 %.2f  p95 %.2f  p99 %.2f  min %.2f  max %.2f  (%.2f FPS)\n", name, m,
               v[v.size() / 2], v[(size_t)(v.size() * 0.95)], v[std::min(v.size() - 1, (size_t)(v.size() * 0.99))], v.front(), v.back(), 1000.0 / m);
    };
    stat(run, "run");
    stat(total, "total");

    std::vector<awnn::TensorView> o;
    s.output_tensors(&o);
    for (auto& t : o) {
        double mn = 1e30, mx = -1e30, sum = 0;
        for (unsigned i = 0; i < t.elements; ++i) { mn = std::min<double>(mn, t.data[i]); mx = std::max<double>(mx, t.data[i]); sum += t.data[i]; }
        printf("output[%d]: %u elems, min %.4f max %.4f mean %.4f\n", t.index, t.elements, mn, mx, sum / t.elements);
    }
    if (!out.empty() && !o.empty()) {
        for (size_t k = 0; k < o.size(); ++k) {
            std::string fn = k ? out + "." + std::to_string(k) : out;
            std::ofstream f(fn, std::ios::binary);
            f.write((char*)o[k].data, o[k].elements * sizeof(float));
            printf("wrote %s\n", fn.c_str());
        }
    }
    s.unload();
    return 0;
}
