#include "FilterComparator.hpp"

lab3::FilterComparator::FilterComparator(const cv::Mat& img) :
    windows_
    {
        Window{ filter_names[static_cast<int>(Filters::gaussian)] },
        Window{ filter_names[static_cast<int>(Filters::median)] },
        Window{ filter_names[static_cast<int>(Filters::bilateral)] }
    },
    img_{ img.clone() },
    gaussianFilter_{ img_, def_filter_size },
    medianFilter_{ img_, def_filter_size },
    bilateralFilter_{ img_, bilateral_filter_size }
{
    /* Initialise output images. */
    gaussianFilter_.doFilter();
    medianFilter_.doFilter();
    bilateralFilter_.doFilter();
}

void lab3::FilterComparator::run()
{
    /* Create the trackbars and show the images. */
    initTrackBars_();
    windows_[static_cast<int>(Filters::gaussian)].showImg(gaussianFilter_.getResult());
    windows_[static_cast<int>(Filters::median)].showImg(medianFilter_.getResult());
    windows_[static_cast<int>(Filters::bilateral)].showImg(bilateralFilter_.getResult());

    /* Main loop. */
    bool done{ false };
    while (!done)
    {
        /* Exit on escape. */
        if (cv::waitKey(frame_delay) == static_cast<int>(lab3::Key::esc)) done = true;
    }

}

void lab3::FilterComparator::gaussianCallBack_(int val, void* ptr)
{
    FilterComparator* this_{ reinterpret_cast<FilterComparator*>(ptr) };

    /* Change parameters and reapply the filter. */
    this_->gaussianFilter_.setSize(this_->gaussVals_[0]);
    this_->gaussianFilter_.setParam(this_->gaussVals_[1] * val_coeff);
    this_->gaussianFilter_.doFilter();
    /* Display the result. */
    this_->windows_[static_cast<int>(Filters::gaussian)].showImg(this_->gaussianFilter_.getResult());
}

void lab3::FilterComparator::medianCallBack_(int val, void* ptr)
{
    FilterComparator* this_{ reinterpret_cast<FilterComparator*>(ptr) };

    /* Change parameter and reapply the filter. */
    this_->medianFilter_.setSize(this_->medVal_);
    this_->medianFilter_.doFilter();
    /* Display the result. */
    this_->windows_[static_cast<int>(Filters::median)].showImg(this_->medianFilter_.getResult());
}

void lab3::FilterComparator::bilateralCallBack_(int val, void* ptr)
{
    FilterComparator* this_{ reinterpret_cast<FilterComparator*>(ptr) };

    /* Change parameters and reapply the filter. */
    this_->bilateralFilter_.setParams(static_cast<double>(this_->biVals_[0]), this_->biVals_[1] * val_coeff);
    this_->bilateralFilter_.doFilter();
    /* Display the result. */
    this_->windows_[static_cast<int>(Filters::bilateral)].showImg(this_->bilateralFilter_.getResult());
}

void lab3::FilterComparator::initTrackBars_()
{
    /* Gaussian filter. */
    cv::createTrackbar(
        kernel_name.data(),
        filter_names[static_cast<int>(Filters::gaussian)].data(),
        &gaussVals_[static_cast<int>(GaussParam::size)],
        filter_counts[static_cast<int>(Filters::gaussian)][static_cast<int>(GaussParam::size)],
        gaussianCallBack_,
        this
    );
    cv::createTrackbar(
        sigma_name.data(),
        filter_names[static_cast<int>(Filters::gaussian)].data(),
        &gaussVals_[static_cast<int>(GaussParam::sigma)],
        filter_counts[static_cast<int>(Filters::gaussian)][static_cast<int>(GaussParam::sigma)],
        gaussianCallBack_,
        this
    );

    /* Median filter. */
    cv::createTrackbar(
        kernel_name.data(),
        filter_names[static_cast<int>(Filters::median)].data(),
        &medVal_,
        filter_counts[static_cast<int>(Filters::median)][static_cast<int>(MedianParam::size)],
        medianCallBack_,
        this
    );

    /* Bilateral filter. */
    cv::createTrackbar(
        range_sigma_name.data(),
        filter_names[static_cast<int>(Filters::bilateral)].data(),
        &biVals_[static_cast<int>(BilatParam::range_sigma)],
        filter_counts[static_cast<int>(Filters::bilateral)][static_cast<int>(BilatParam::range_sigma)],
        bilateralCallBack_,
        this
    );
    cv::createTrackbar(
        space_sigma_name.data(),
        filter_names[static_cast<int>(Filters::bilateral)].data(),
        &biVals_[static_cast<int>(BilatParam::space_sigma)],
        filter_counts[static_cast<int>(Filters::bilateral)][static_cast<int>(BilatParam::space_sigma)],
        bilateralCallBack_,
        this
    );
}
