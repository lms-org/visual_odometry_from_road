#ifndef SIMPLE_VISUAL_ODOMETRY_H
#define SIMPLE_VISUAL_ODOMETRY_H

#include <lms/module.h>
#define USE_OPENCV
#include <lms/imaging/image.h>
#include <kalman_filter/ctrv_vxy.h>
#include <lms/math/pose.h>

/**
 * @brief LMS module simple_visual_odometry
 **/
class SimpleVisualOdometry : public lms::Module {
    lms::ReadDataChannel<lms::imaging::Image> image;
    lms::WriteDataChannel<lms::imaging::Image> debugImage,trajectoryImage;
    lms::WriteDataChannel<lms::math::Pose2DHistory> poseHistory;
    lms::imaging::Image oldImage;
    std::vector<cv::Point2f> oldImagePoints;

    cv::Mat world2cam,cam2world;
    cv::Mat currentPosition;
    cv::Mat transRotNew,transRotOld;

    //tmp objects
    std::vector<cv::Point2f> newImagePoints;
    std::vector<uchar> status;
    kalman_filters::ctrv_vxy::MassModelUKF ukf;


public:
    bool initialize() override;
    bool deinitialize() override;
    bool cycle() override;
    void configsChanged();
    void detectFeaturePointsInOldImage(const cv::Rect rect, const int fastThreshold);
    void checkNewFeaturePoints(const cv::Rect rect);
    bool validateMeasurement(const float vx,const float vy,const float dPhi);
};

#endif // SIMPLE_VISUAL_ODOMETRY_H
