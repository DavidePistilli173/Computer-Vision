#ifndef LAB3_MORPH_HPP
#define LAB3_MORPH_HPP

#include <opencv2/core.hpp>

namespace lab3
{
    class Morph
    {
    public:
        explicit Morph(const cv::Mat& img);

        /********** CONSTANTS **********/
        const int def_k_size{ 3 };

        void execute();
        cv::Mat getResult() const;

    private:
        cv::Mat img_;
        cv::Mat result_;
    };
}

#endif