#ifndef LANE_NET_DETECTOR_H
#define LANE_NET_DETECTOR_H


#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

class LaneNetDetectorPrivate;
class LaneNetDetector {
public:
    typedef boost::shared_ptr<LaneNetDetector> Ptr;
    LaneNetDetector();
    ~LaneNetDetector();

    bool init(const std::string& model_path, float gpu_memory_fraction = 0.7, int gpu_id = 0);
    bool detectLane(const cv::Mat& input_img, cv::Mat& lane_ret);
    bool detectLane(const cv::Mat& input_img, std::vector<std::vector<cv::Point2i> >& lane_clusters);
    bool detectLane(const cv::Mat& input_img, cv::Mat& lane_ret, std::vector<std::vector<cv::Point2i> >& lane_clusters);

private:
    LaneNetDetectorPrivate* _ptr;
};




#endif