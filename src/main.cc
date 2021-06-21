#include "pipeline/pipeline.hh"
#include <iostream>
#include "misc/image-load.hh"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "[USAGE] ./gpu-implementation filepath\n";
        return 1;
    }

    auto image = load_image(argv[1]);
    cv::imwrite("grayscale.png", image);

    launch_pipeline(image);
}
