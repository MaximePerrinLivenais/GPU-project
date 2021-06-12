#include <opencv2/imgcodecs.hpp>

#include "lbp/lbp.cuh"
#include "misc/image-load.hh"

int main()
{
    auto image = load_image("../data/images/barcode-00-01.jpg");
    cv::imwrite("grayscale.png", image);

    compute_lbp_values(image.data, image.cols, image.rows);
}
