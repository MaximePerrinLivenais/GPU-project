#include <benchmark/benchmark.h>
#include "misc/image-load.hh"
#include "lbp/lbp.cuh"
#include "cpu/lbp.hh"

static void BM_lbp_cpu(benchmark::State& state)
{
    auto image = load_image("../data/images/barcode-00-01.jpg");
    for (auto _ : state)
        compute_lbp_values_cpu(image, 16);
}

static void BM_lbp_gpu(benchmark::State& state)
{
    auto image = load_image("../data/images/barcode-00-01.jpg");
    for (auto _ : state)
        compute_lbp_values(image.data, image.cols, image.rows);
}

BENCHMARK(BM_lbp_cpu)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_lbp_gpu)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_MAIN();
