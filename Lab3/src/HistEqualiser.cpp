#include "HistEqualiser.hpp"

#include "lab3.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

lab3::HistEqualiser::HistEqualiser(const cv::Mat& img) :
    src_{ img.clone() },
    img_{ img.clone() }
{
    cv::split(img_, channels_); // Split the channels of the image.
    computeHists();
}

void lab3::HistEqualiser::computeHists()
{
    /* Compute the histogram of each channel. */
    if (mode_ == Mode::rgb)
    {
        for (int i = 0; i < channel_num; ++i)
        {
            cv::calcHist(
                &channels_[i], 1, &hist_channel, cv::Mat{},
                histograms_[i], 1, &rgb_bins, &rgb_ranges_
            );
        }
    }
    else
    {
        for (int i = 0; i < channel_num; ++i)
        {
            cv::calcHist(
                &channels_[i], 1, &hist_channel, cv::Mat{},
                histograms_[i], 1, &hsv_bins[i], &hsv_ranges_[i]
            );
        }
    }
}

void lab3::HistEqualiser::equalise(HSV channel)
{
    if (mode_ == Mode::rgb)
    {
        /* Equalise all channels. */
        for (int i = 0; i < channel_num; ++i)
        {
            cv::equalizeHist(channels_[i], channels_[i]);
        }
    }
    else
    {
        /* Equalise just the specified channel. */
        cv::equalizeHist(channels_[static_cast<int>(channel)], channels_[static_cast<int>(channel)]);
    }

    cv::merge(channels_, img_); // Merge the results back in the image.
    computeHists(); // Recompute the histograms.
}

cv::Mat lab3::HistEqualiser::getResult() const
{
    return img_.clone();
}

void lab3::HistEqualiser::reset()
{
    img_ = src_.clone();
    if (mode_ == Mode::hsv) cv::cvtColor(img_, img_, cv::COLOR_BGR2HSV);
    cv::split(img_, channels_);
    computeHists();
}

void lab3::HistEqualiser::setMode(Mode mode)
{
    if (mode_ == mode) return;

    mode_ = mode;
    if (mode == Mode::rgb) cv::cvtColor(img_, img_, cv::COLOR_HSV2BGR);
    else cv::cvtColor(img_, img_, cv::COLOR_BGR2HSV);

    cv::split(img_, channels_);
    computeHists();
}

void lab3::HistEqualiser::show() const
{
    // Min/Max computation
    std::array<double, channel_num> hmax{ 0.0, 0.0, 0.0 };
    double min;
    cv::minMaxLoc(histograms_[static_cast<int>(lab3::RGB::b)], &min, &hmax[static_cast<int>(lab3::RGB::b)]);
    cv::minMaxLoc(histograms_[static_cast<int>(lab3::RGB::g)], &min, &hmax[static_cast<int>(lab3::RGB::g)]);
    cv::minMaxLoc(histograms_[static_cast<int>(lab3::RGB::r)], &min, &hmax[static_cast<int>(lab3::RGB::r)]);

    std::array colors
    {
        cv::Scalar{ rgb_max_val, 0, 0 },
        cv::Scalar{ 0, rgb_max_val, 0 },
        cv::Scalar{ 0, 0, rgb_max_val }
    };

    std::array hists
    {
        Window(hist_win_names[static_cast<int>(lab3::RGB::b)]),
        Window(hist_win_names[static_cast<int>(lab3::RGB::g)]),
        Window(hist_win_names[static_cast<int>(lab3::RGB::r)])
    };

    std::vector<cv::Mat> canvas(histograms_.size());

    // Display each histogram in a canvas
    for (int i = 0, end = histograms_.size(); i < end; i++)
    {
        canvas[i] = cv::Mat::ones(125, histograms_[0].rows, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < histograms_[0].rows - 1; j++)
        {
            cv::line(
                canvas[i],
                cv::Point(j, rows),
                cv::Point(j, rows - (histograms_[i].at<float>(j) * rows / hmax[i])),
                colors[i],
                1, 8, 0
            );
        }

        hists[i].showImg(canvas[i]);
    }

    Window srcWin{ img_win_name };
    srcWin.showImg(img_);

    cv::waitKey(0);
}
