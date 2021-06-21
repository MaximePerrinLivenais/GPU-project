#include "misc/image-load.hh"
#include "lbp/lbp.cuh"
#include "cpu/lbp.hh"
#include <opencv2/imgproc.hpp>
#include <benchmark/benchmark.h>
#include "cpu/pipeline.hh"
#include "pipeline/pipeline.hh"

static void BM_lbp_cpu(benchmark::State& state)
{
    auto image = load_image("../data/images/barcode-00-01.jpg");

    auto rows = state.range(0);

    cv::resize(image, image, cv::Size(rows, rows));

    for (auto _ : state)
        compute_lbp_values_cpu(image, 16);
}

static void BM_lbp_gpu(benchmark::State& state)
{
    auto image = load_image("../data/images/barcode-00-01.jpg");

    auto rows = state.range(0);

    cv::resize(image, image, cv::Size(rows, rows));

    for (auto _ : state)
        compute_lbp_values(image.data, image.cols, image.rows);
}

static void BM_pipeline_cpu(benchmark::State& state)
{
    auto image = load_image("../data/images/barcode-00-01.jpg");

    auto rows = state.range(0);

    cv::resize(image, image, cv::Size(rows, rows));

    for (auto _ : state)
        full_pipeline(image, 16);
}

static void BM_pipeline_gpu(benchmark::State& state)
{
    auto image = load_image("../data/images/barcode-00-01.jpg");

    auto rows = state.range(0);

    cv::resize(image, image, cv::Size(rows, rows));

    for (auto _ : state)
        launch_pipeline(image);
}

static void custom_arguments(benchmark::internal::Benchmark* b)
{
    for (auto i = 16; i <= 16384; i *= 2)
        b->Args({i});
}


BENCHMARK(BM_lbp_cpu)
    ->Apply(custom_arguments)
    ->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_lbp_gpu)
    ->Apply(custom_arguments)
    ->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_pipeline_cpu)
    ->Apply(custom_arguments)
    ->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_pipeline_gpu)
    ->Apply(custom_arguments)
    ->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();
