#include <benchmark/benchmark.h>

static void BM_lbp(benchmark::State& state)
{
    auto image = load_image("../data/images/barcode-00-01.jpg");
    for (auto _ : state)
        compute_lbp_values(image.data, image.cols, image.rows)

}

BENCHMARK(BM_lbp)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();
