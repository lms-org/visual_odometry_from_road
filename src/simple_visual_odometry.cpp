#define USE_OPENCV
#include "simple_visual_odometry.h"
#include <opencv/cv.h>
#include "vo_features.h"
#include <lms/imaging/graphics.h>

bool SimpleVisualOdometry::initialize() {
    image = readChannel<lms::imaging::Image>("IMAGE");
    debugImage = writeChannel<lms::imaging::Image>("DEBUG_IMAGE");
    trajectoryImage = writeChannel<lms::imaging::Image>("TRAJECTORY_IMAGE");
    trajectoryImage->resize(512,512,lms::imaging::Format::BGRA);
    trajectoryImage->fill(0);
    configsChanged();
    currentPosition.create(3,1,CV_64F);
    currentPosition.at<double>(0) = 0;
    currentPosition.at<double>(1) = 0;
    currentPosition.at<double>(2) = 1;
    transRotNew.create(3,3,CV_64F);
    transRotOld = cv::Mat::eye(3,3,CV_64F);

    //test
    t_f.create(3,1,CV_64F);
    R_f.create(3,3,CV_64F);

    ukf.init();
    return true;
}

bool SimpleVisualOdometry::deinitialize() {
    return true;
}

//TODO We could try Kabasch_algoithm
bool SimpleVisualOdometry::cycle() {
    const float dt = 0.1;
    //logger.error("ANGLE BEFORE")<<ukf.lastState.phi();
    ukf.predict(dt);
    //logger.error("ANGLE AFTER")<<ukf.lastState.phi();

    int fastThreshold = config().get<int>("fastThreshold",20);
    bool drawDebug = config().get<bool>("drawDebug",false);
    //catch first round
    if(oldImage.width()==0 || oldImage.height() == 0){
        //first round
        oldImage = *image;
        return true;
    }
    //clear tmp objects
    newImagePoints.clear();
    status.clear();
    //get region of interest
    int xmin = 0;
    int xmax = image->width();
    int ymin = 0;
    int ymax = image->height();
    if(config().hasKey("xmin")){
        xmin = config().get<int>("xmin");
        xmax = config().get<int>("xmax");
        ymin = config().get<int>("ymin");
        ymax = config().get<int>("ymax");
    }
    cv::Rect rect(xmin,ymin,xmax-xmin,ymax-ymin);

    cv::Mat oldIm = oldImage.convertToOpenCVMat()(rect);
    int minFeatureCount = config().get<int>("minFeatureCount",1);
    bool alreadySearched = false;
    if(oldImagePoints.size() <minFeatureCount){
        alreadySearched = true;
        oldImagePoints.clear();
        featureDetection(oldIm, oldImagePoints,fastThreshold); //detect points
        if(oldImagePoints.size() == 0){
            oldImage = *image;
            logger.error("No features detected!");
            return false;
        }
    }
    cv::Mat newIm = image->convertToOpenCVMat()(rect);
    if(drawDebug){
        debugImage->resize(image->width(),image->height(),lms::imaging::Format::BGRA);
        debugImage->fill(0);
        lms::imaging::BGRAImageGraphics graphics(*debugImage);
        graphics.setColor(lms::imaging::blue);
        graphics.drawRect(rect.x,rect.y,rect.width,rect.height);
    }

    logger.debug("oldPoints")<<oldImagePoints.size();
    featureTracking(oldIm,newIm,oldImagePoints,newImagePoints, status); //track those features to the new image
    if(newImagePoints.size() <minFeatureCount){
        logger.warn("not enough points tracked!")<<newImagePoints.size();
        if(!alreadySearched){
            newImagePoints.clear();
            oldImagePoints.clear();
            status.clear();
            featureDetection(oldIm, oldImagePoints,fastThreshold); //detect points
        }
        logger.debug("detected new features")<<oldImagePoints.size();
        if(oldImagePoints.size() == 0){
            logger.error("No features detected!");
        }else{
            featureTracking(oldIm,newIm,oldImagePoints,newImagePoints, status); //track those features to the new image
            logger.debug("tracking new features")<<newImagePoints.size();
            if(newImagePoints.size() <= 1){
                logger.error("Not enough features could be tracked!")<<newImagePoints.size();
            }
        }
    }
    if(newImagePoints.size() > 1){
        //as we detect the points inside a subimage we have to get the actual position:
        std::vector<cv::Point2f> tmpOldImagePoints,tmpNewImagePoints;
        for(std::size_t i = 0; i < oldImagePoints.size(); i++){
            cv::Point2f newP = oldImagePoints[i];
            newP.x = newP.x + rect.x;
            newP.y = newP.y + rect.y;
            tmpOldImagePoints.push_back(newP);
            newP = newImagePoints[i];
            newP.x = newP.x + rect.x;
            newP.y = newP.y + rect.y;
            tmpNewImagePoints.push_back(newP);

        }
        if(drawDebug){
            lms::imaging::BGRAImageGraphics graphics(*debugImage);
            graphics.setColor(lms::imaging::blue);
            graphics.drawRect(rect.x,rect.y,rect.width,rect.height);
            graphics.setColor(lms::imaging::red);
            for(cv::Point2f p:tmpNewImagePoints){
                graphics.drawCross(p.x,p.y);
            }
        }
        /*
        //test
        //recovering the pose and the essential matrix
        double focal = 718.8560;//https://en.wikipedia.org/wiki/Focal_length
        cv::Point2d pp(image->width()/2, image->height()/2); //http://stackoverflow.com/questions/6658258/principle-point-in-camera-matrix-programming-issue
        cv::Mat E, R, t, mask;
        E = cv::findEssentialMat(tmpNewImagePoints, tmpOldImagePoints, focal, pp, RANSAC, 0.999, 1.0, mask);
        cv::recoverPose(E, tmpNewImagePoints, tmpOldImagePoints, R, t, focal, pp, mask);
        double scale = 1;
        logger.error("wasd")<<t;
        logger.error("wasd2")<<R;
        t_f = t_f + scale*(R_f*t);
        R_f = R*R_f;
        cv::Mat newPos;
        newPos.create(2,1,CV_32F);
        newPos.at<double>(0)=t_f.at<double>(0);
        newPos.at<double>(1)=t_f.at<double>(2);
        logger.error("newPOS")<<newPos.at<double>(0)<<" "<<newPos.at<double>(1);

        //END-test
        */

        //transform points to 2D-Coordinates
        std::vector<cv::Point2f> world_old,world_new;
        cv::perspectiveTransform(tmpOldImagePoints,world_old,cam2world);
        cv::perspectiveTransform(tmpNewImagePoints,world_new,cam2world);



        //######################################################

        //from http://math.stackexchange.com/questions/77462/finding-transformation-matrix-between-two-2d-coordinate-frames-pixel-plane-to-w
        //create data
        cv::Mat leftSide,rightSide;
        rightSide.create(2*world_old.size(),1, CV_64F);
        leftSide.create(2*world_old.size(),4,CV_64F);
        for(std::size_t i = 0; i < 2*world_old.size(); i+=2){
            leftSide.at<double>(i,0) = world_old[i/2].x;
            leftSide.at<double>(i,1) = -world_old[i/2].y;
            leftSide.at<double>(i,2) = 1;
            leftSide.at<double>(i,3) = 0;
            leftSide.at<double>(i+1,0) = world_old[i/2].y;
            leftSide.at<double>(i+1,1) = world_old[i/2].x;
            leftSide.at<double>(i+1,2) = 0;
            leftSide.at<double>(i+1,3) = 1;
            rightSide.at<double>(i,0) = world_new[i/2].x;
            rightSide.at<double>(i+1,0) = world_new[i/2].y;
        }
        //solve it
        cv::Mat res;
        cv::solve(leftSide,rightSide,res,cv::DECOMP_SVD); //TODO we could use pseudo-inverse
        //calculate the rotation/translation matrix
        float dx = res.at<double>(2);
        float dy = res.at<double>(3);
        float angle = std::atan2(res.at<double>(1),res.at<double>(0));
        //TODO use cos/sin from res
        transRotNew.at<double>(0,0) = std::cos(angle);
        transRotNew.at<double>(0,1) = -std::sin(angle);
        transRotNew.at<double>(1,0) = std::sin(angle);
        transRotNew.at<double>(1,1) = std::cos(angle);
        transRotNew.at<double>(0,2) = dx;
        transRotNew.at<double>(1,2) = dy;
        transRotNew.at<double>(2,0) = 0;
        transRotNew.at<double>(2,1) = 0;
        transRotNew.at<double>(2,2) = 1;
        //translate the current position
        //TODO Hier geht was schief
        transRotOld = transRotOld*transRotNew;
        //currentPosition = transRotNew*currentPosition;
        cv::Mat newPos = transRotOld*currentPosition;

        //update the ukf
        ukf.setMeasurementVec(dx/dt,dy/dt,angle/dt);
        //ukf.pu(dt);
        ukf.update();

        lms::imaging::BGRAImageGraphics traGraphics(*trajectoryImage);
        if(drawDebug){
            traGraphics.setColor(lms::imaging::red);
            traGraphics.drawPixel(newPos.at<double>(0)*512/30+256,newPos.at<double>(1)*512/30+256);
            traGraphics.setColor(lms::imaging::blue);
            traGraphics.drawPixel(ukf.lastState.x()*512/30+256,ukf.lastState.y()*512/30+256);
        }
    }else{
        //TODO we lost track
    }

    //set old values
    oldImage = *image;
    oldImagePoints = newImagePoints;
    if(drawDebug){
        //cv::namedWindow( "Camera", WINDOW_AUTOSIZE );
        cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
        cv::imshow( "Display window", trajectoryImage->convertToOpenCVMat() );                   // Show our image inside it.

        cv::waitKey(1);
    }
    return true;
}

void SimpleVisualOdometry::configsChanged(){
    cam2world.create(3,3,CV_64F);
    std::vector<float> points = config().getArray<float>("cam2world");
    if(points.size() != 9){
        logger.error("invalid cam2world");
        return;
    }
    int i = 0;
    for(int r = 0; r < 3; r++) {
        for(int c = 0; c < 3; c++) {
            cam2world.at<double>(r, c) = points[i];
            i++;
        }
    }
}
