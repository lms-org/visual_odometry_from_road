#ifndef SIMPLE_VISUAL_ODOMETRY_H
#define SIMPLE_VISUAL_ODOMETRY_H

#include <lms/module.h>
#define USE_OPENCV
#include <lms/imaging/image.h>
#include <kalman_filter/ctrv_vxy.h>

/**
 * @brief LMS module simple_visual_odometry
 **/
class SimpleVisualOdometry : public lms::Module {
    lms::ReadDataChannel<lms::imaging::Image> image;
    lms::WriteDataChannel<lms::imaging::Image> debugImage,trajectoryImage;
    lms::imaging::Image oldImage;
    std::vector<cv::Point2f> oldImagePoints;

    cv::Mat world2cam,cam2world;
    cv::Mat currentPosition;
    cv::Mat transRotNew,transRotOld;

    //tmp objects
    std::vector<cv::Point2f> newImagePoints;
    std::vector<uchar> status;
    kalman_filters::ctrv_vxy::MassModelUKF ukf;

    //test
    cv::Mat t_f;// = t_f + scale*(R_f*t);
    cv::Mat R_f;// = R*R_f;


public:
    bool initialize() override;
    bool deinitialize() override;
    bool cycle() override;
    void configsChanged();
};

#endif // SIMPLE_VISUAL_ODOMETRY_H
