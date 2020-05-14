#include "StreetAnaliser.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

using namespace lab4;

StreetAnaliser::StreetAnaliser(cv::Mat img) :
    srcImg_{ std::move(img) },
    outputImg_{ srcImg_.clone() },
    outputWin_{ output_win_name },
    lineCanny_{ img_, lane_win_names[static_cast<int>(TuningPhase::canny)] },
    lineHough_{ lineCanny_.getResultRef(), lane_win_names[static_cast<int>(TuningPhase::hough)] },
    circleHough_{ srcImg_, sign_win_name }
{
    /* Smooth the input image. */
    cv::GaussianBlur(srcImg_, img_, cv::Size{ gauss_width, gauss_height }, gauss_sigma);
}

void StreetAnaliser::run()
{
    Log::info("Starting the main loop.");
    bool done{ false };
    /* Main loop. */
    while (!done)
    {
        update_();
        if (cv::waitKey(frame_delay) == static_cast<int>(lab4::Key::esc)) done = true;
    }
}

void StreetAnaliser::drawLane_()
{
    using LineParam = LineHough::LineParam;

    Log::info("Updating lane...");

    if (std::optional lane{ lineHough_.getRelevantLines() }; lane.has_value())
    {
        auto [line1, line2] = lane.value();
        /* Compute the intersection of the two lines. */
        double cos1{ std::cos(line1[static_cast<int>(LineParam::theta)]) };
        double sin1{ std::sin(line1[static_cast<int>(LineParam::theta)]) };
        double cos2{ std::cos(line2[static_cast<int>(LineParam::theta)]) };
        double sin2{ std::sin(line2[static_cast<int>(LineParam::theta)]) };

        double den{ (sin2 * cos1) - (sin1 * cos2) };
        /* If the lines are not parallel, colour the lane. */
        if (den != 0)
        {
            /* Intersection point. */
            double y
            {
                ((line2[static_cast<int>(LineParam::rho)] * cos1) -
                (line1[static_cast<int>(LineParam::rho)] * cos2)) / den
            };
            double x{ (line1[static_cast<int>(LineParam::rho)] - (y * sin1)) / cos1 };

            std::vector<std::vector<cv::Point>> points(1);
            points[0].emplace_back(cvRound(x), cvRound(y));

            /* Base points. */
            int maxY{ img_.rows };
            x = (line1[static_cast<int>(LineParam::rho)] - (maxY * sin1)) / cos1;
            points[0].emplace_back(cvRound(x), maxY);

            x = (line2[static_cast<int>(LineParam::rho)] - (maxY * sin2)) / cos2;
            points[0].emplace_back(cvRound(x), maxY);

            /* Draw the polygon. */
            cv::fillPoly(
                outputImg_,
                points,
                cv::Scalar{ 0, 0, lane_colour_intensity },
                cv::LINE_AA
            );

            Log::info("SUCCESS");
        }
        else
        {
            Log::warn("Parallel lines detected, no colouring performed.");
        }
    }
    else
    {
        Log::warn("Not enough lines detected.");
    }
}

void StreetAnaliser::drawSign_()
{
    using CircleParam = CircleHough::CircleParam;

    Log::info("Updating sign...");

    if (std::optional sign{ circleHough_.getRelevantCircle() }; sign.has_value())
    {
        cv::circle(
            outputImg_,
            cv::Point
            {
                cvRound(sign.value()[static_cast<int>(CircleParam::x)]),
                cvRound(sign.value()[static_cast<int>(CircleParam::y)])
            },
            cvRound(sign.value()[static_cast<int>(CircleParam::r)]),
            cv::Scalar{ 0, sign_colour_intensity, 0 },
            cv::FILLED,
            cv::LINE_AA
        );
        Log::info("SUCCESS");
    }
    else
    {
        Log::warn("No sign detected.");
    }
}

void StreetAnaliser::update_()
{
    bool laneUpdate{ lineHough_.update(lineCanny_.update()) };
    bool signUpdate{ circleHough_.update() };

    if (laneUpdate || signUpdate)
    {
        outputImg_ = srcImg_.clone(); // Reset the output image.
        drawLane_(); // Draw the lane.
        drawSign_(); // Draw the sign.
        outputWin_.showImg(outputImg_); // Show the result.
    }
}
