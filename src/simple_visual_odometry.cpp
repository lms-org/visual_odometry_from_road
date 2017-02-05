#define USE_OPENCV
#include "simple_visual_odometry.h"
#include <opencv/cv.h>
#include "vo_features.h"
#include <lms/imaging/graphics.h>
#include <lms/exception.h>

bool SimpleVisualOdometry::initialize() {
    image = readChannel<lms::imaging::Image>("IMAGE");
    debugImage = writeChannel<lms::imaging::Image>("DEBUG_IMAGE");
    trajectoryImage = writeChannel<lms::imaging::Image>("TRAJECTORY_IMAGE");
    poseHistory = writeChannel<lms::math::Pose2DHistory>("POSE2D_HISTORY");
    poseHistory->posesMaxSize = 100;
    trajectoryImage->resize(512,512,lms::imaging::Format::BGRA);
    trajectoryImage->fill(0);
    configsChanged();
    currentPosition.create(3,1,CV_64F);
    currentPosition.at<double>(0) = 0;
    currentPosition.at<double>(1) = 0;
    currentPosition.at<double>(2) = 1;
    transRotNew.create(3,3,CV_64F);
    transRotOld = cv::Mat::eye(3,3,CV_64F);

    ukf.init();
    return true;
}

bool SimpleVisualOdometry::deinitialize() {
    return true;
}

bool SimpleVisualOdometry::cycle() {
    try{
        //use the predict of the ukf to calculate new xy position if even if we haven't found feature points
        const float dt = 0.1;
        ukf.predict(dt);


        //we try to find feature points in the old Image and try to redetect them in the new image
        //if we haven't found enough feature points in the new image we try to find more in the old image, therefore we store the old image
        //we crop the image to only find points on the road
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

        int minFeatureCount = config().get<int>("minFeatureCount",1);
        bool alreadySearched = false;
        if((int)oldImagePoints.size() <minFeatureCount){
            alreadySearched = true;
            /*
            oldImagePoints.clear();
            featureDetection(oldIm, oldImagePoints,fastThreshold); //detect points
            //TODO transform found points coord-sys of the full image
            */
            detectFeaturePointsInOldImage(rect,fastThreshold);
            if(oldImagePoints.size() == 0){
                oldImage = *image;
                logger.error("No features detected!");
                return false;
            }
        }
        //no need to clip the image as the oldPoints are in the frame of the full image!
        if(drawDebug){
            debugImage->resize(image->width(),image->height(),lms::imaging::Format::BGRA);
            debugImage->fill(0);
            lms::imaging::BGRAImageGraphics graphics(*debugImage);
            graphics.setColor(lms::imaging::blue);
            graphics.drawRect(rect.x,rect.y,rect.width,rect.height);
        }
        logger.debug("oldPoints")<<oldImagePoints.size();
        //track the old feature points
        featureTracking(rect);
        //TODO featureTracking(oldImFull,newIm,oldImagePoints,newImagePoints, status); //track those features to the new image
        //checkNewFeaturePoints(rect);
        if((int)newImagePoints.size() <minFeatureCount){
            logger.warn("not enough points tracked!")<<newImagePoints.size();
            if(!alreadySearched){
                detectFeaturePointsInOldImage(rect,fastThreshold);
            }else{
                logger.warn("already searched, found not enough points");
            }
            logger.debug("detected new features")<<oldImagePoints.size();
            if(oldImagePoints.size() == 0){
                logger.error("No features detected!");
            }else{
                featureTracking(rect);
                //TODO featureTracking(oldImFull,newIm,oldImagePoints,newImagePoints, status); //track those features to the new image
                //checkNewFeaturePoints(rect);
                logger.debug("tracking new features")<<newImagePoints.size();
                if(newImagePoints.size() <= 1){
                    logger.error("Not enough features could be tracked!")<<newImagePoints.size();
                }
            }
        }
        if(newImagePoints.size() > 1){
            if(drawDebug){
                lms::imaging::BGRAImageGraphics graphics(*debugImage);
                graphics.setColor(lms::imaging::blue);
                graphics.drawRect(rect.x,rect.y,rect.width,rect.height);
                graphics.setColor(lms::imaging::red);
                for(cv::Point2f p:newImagePoints){
                    graphics.drawCross(p.x,p.y);
                }
            }

            //transform points to 2D-Coordinates
            std::vector<cv::Point2f> world_old,world_new;
            cv::perspectiveTransform(oldImagePoints,world_old,cam2world);
            cv::perspectiveTransform(newImagePoints,world_new,cam2world);

            //######################################################
            //from http://math.stackexchange.com/questions/77462/finding-transformation-matrix-between-two-2d-coordinate-frames-pixel-plane-to-w
            //create data
            cv::Mat leftSide,rightSide;
            rightSide.create(2*world_old.size(),1, CV_64F);
            leftSide.create(2*world_old.size(),4,CV_64F);
            for(std::size_t i = 0; i < 2*world_old.size(); i+=2){
                //we have the new points and would like to know how to get to the old ones as they moved closer to us
                leftSide.at<double>(i,0) = world_new[i/2].x;
                leftSide.at<double>(i,1) = -world_new[i/2].y;
                leftSide.at<double>(i,2) = 1;
                leftSide.at<double>(i,3) = 0;
                leftSide.at<double>(i+1,0) = world_new[i/2].y;
                leftSide.at<double>(i+1,1) = world_new[i/2].x;
                leftSide.at<double>(i+1,2) = 0;
                leftSide.at<double>(i+1,3) = 1;
                rightSide.at<double>(i,0) = world_old[i/2].x;
                rightSide.at<double>(i+1,0) = world_old[i/2].y;
            }
            //solve it
            cv::Mat res;
            cv::solve(leftSide,rightSide,res,cv::DECOMP_SVD); //TODO we could use pseudo-inverse
            float dx = res.at<double>(2);
            float dy = res.at<double>(3);
            float angle = std::atan2(res.at<double>(1),res.at<double>(0));

            if(validateMeasurement(dx/dt,dy/dt,angle/dt)){
                //update the ukf
                logger.debug("updating ukf")<<dx/dt<<" "<<dy/dt<<" "<<angle/dt;

                ukf.setMeasurementVec(dx/dt,dy/dt,angle/dt);
                ukf.update();
                /*
                lms::imaging::BGRAImageGraphics traGraphics(*trajectoryImage);
                if(drawDebug){
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
                    transRotOld = transRotOld*transRotNew;
                    //currentPosition = transRotNew*currentPosition;
                    cv::Mat newPos = transRotOld*currentPosition;
                    traGraphics.setColor(lms::imaging::red);
                    traGraphics.drawPixel(newPos.at<double>(0)*512/30+256,-newPos.at<double>(1)*512/30+256);
                }
                */
            }else{
                logger.warn("not updating ukf, invalid values")<<dx/dt<<" "<<dy/dt<<" "<<angle/dt;
            }
        }else{
            //we lost track, no update for the ukf
        }
        //add new pose
        poseHistory->addPose(ukf.lastState.x(),ukf.lastState.y(),ukf.lastState.phi(),lms::Time::now().toFloat<std::milli, double>());
        //set old values
        oldImage = *image;
        oldImagePoints = newImagePoints;
        if(drawDebug){
            lms::imaging::BGRAImageGraphics traGraphics(*trajectoryImage);
            traGraphics.setColor(lms::imaging::blue);
            traGraphics.drawPixel(poseHistory->currentPose().x*512/30+256,-poseHistory->currentPose().y*512/30+256);
            //cv::namedWindow( "Camera", WINDOW_AUTOSIZE );
            cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
            cv::imshow( "Display window", trajectoryImage->convertToOpenCVMat() );                   // Show our image inside it.

            cv::waitKey(1);
        }
    }catch(std::exception &e){
        logger.error("exception thrown")<<e.what()<<" reinitialising ukf";
        ukf.init(ukf.lastState.x(),ukf.lastState.y(),ukf.lastState.phi());
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

void SimpleVisualOdometry::detectFeaturePointsInOldImage(cv::Rect rect, const int fastThreshold){
    newImagePoints.clear();
    oldImagePoints.clear();
    status.clear();
    cv::Mat oldImClipped = oldImage.convertToOpenCVMat()(rect);
    vo_features::featureDetection(oldImClipped, oldImagePoints,fastThreshold); //detect points
    for(cv::Point2f &v:oldImagePoints){
        v.x += rect.x;
        v.y += rect.y;
    }
    //TODO transform found points coord-sys of the full image

}

void SimpleVisualOdometry::checkNewFeaturePoints(const cv::Rect rect){
    if(newImagePoints.size() != oldImagePoints.size()){
        throw lms::LmsException("newImagePoints size does not match oldImagePoints size");
    }
    for(int i = 0; i < (int)newImagePoints.size();){
        if(rect.contains(newImagePoints[i])){
            i++;
        }else{
            newImagePoints.erase(newImagePoints.begin()+i);
            oldImagePoints.erase(oldImagePoints.begin()+i);
        }
    }
}

void SimpleVisualOdometry::featureTracking(cv::Rect rect){
    newImagePoints.clear();
    cv::Mat oldImFull = oldImage.convertToOpenCVMat();
    cv::Mat newIm = image->convertToOpenCVMat();
    for(cv::Point2f &v:oldImagePoints){
        v.x -= rect.x;
        v.y -= rect.y;
    }
    vo_features::featureTracking(oldImFull(rect),newIm(rect),oldImagePoints,newImagePoints, status); //track those features to the new image
    for(cv::Point2f &v:newImagePoints){
        v.x += rect.x;
        v.y += rect.y;
    }
    for(cv::Point2f &v:oldImagePoints){
        v.x += rect.x;
        v.y += rect.y;
    }
    checkNewFeaturePoints(rect);
}


bool SimpleVisualOdometry::validateMeasurement(const float vx, const float vy, const float dPhi){
    float vxMax = config().get<float>("vxMax",5);
    float vyMax = config().get<float>("vyMax",5);
    float vxMin = config().get<float>("vxMin",-0.1);
    float vyMin = config().get<float>("vyMin",-0.1);
    float omegaMax = config().get<float>("omegaMax",0.5);
    if(vx > vxMax || vx < vxMin){
        return false;
    }
    if(vy > vyMax || vy < vyMin){
        return false;
    }
    if(std::fabs(dPhi) > omegaMax){
        return false;
    }
    return true;

}


//TODO We could try Kabasch_algoithm

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
