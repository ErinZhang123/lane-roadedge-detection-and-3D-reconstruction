#include "LaneNetDetector.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

void clusters2pcd(std::vector<std::vector<cv::Point2i> > lane_clusters,std::string pcdName){
    Eigen::Matrix3f K,R;
    R= Eigen::AngleAxisf(1.57, Eigen::Vector3f::UnitX())
    *Eigen::AngleAxisf(1.57, Eigen::Vector3f::UnitZ());
    //Eigen::AngleAxisf(0.861723, Eigen::Vector3f::UnitX())
    //*Eigen::AngleAxisf(-1.48802,  Eigen::Vector3f::UnitY())
    //*Eigen::AngleAxisf(1.63929, Eigen::Vector3f::UnitZ());
    
    Eigen::MatrixXf r2c(4,4),t(1,4),radar_plane(1,4),cam_plane(1,4);
    t<<-0.7,-0.4,0.3,1;
    r2c.topLeftCorner(3,3)=R;
    r2c.topRightCorner(4,1)=t;
    r2c.bottomLeftCorner(1,3)=Eigen::MatrixXf::Zero(1,3);
    radar_plane<<0.239956,0.0229191,-0.970513,-0.918869;
    cam_plane=radar_plane*r2c.inverse();
    //std::cout<<r2c;
    float A=cam_plane(0,0),B=cam_plane(0,1),C=cam_plane(0,2),D=cam_plane(0,3);
    std::cout<<A<<' '<<B<<' '<<C<<std::endl;
    Eigen::Vector3f ABC(A,B,C);
    K<<1673.764472,0,615.269162,0,1671.726840,486.603777,0,0,1;
    
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width=0;
    for(int cluster_num=0;cluster_num<lane_clusters.size();cluster_num++)
    cloud.width    += lane_clusters[cluster_num].size();
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (cloud.width * cloud.height);
    int cloudidx=0;
    for(int cluster_num=0;cluster_num<lane_clusters.size();cluster_num++){
        std::vector<cv::Point2i> points=lane_clusters[cluster_num];
            
        for(int idx=0;idx<points.size();idx++){
            cv::Point2i output_p=points[idx];
            float img_px=output_p.x*1280./512.;
            float img_py=output_p.y*640./256.+320;
            Eigen::Vector3f img_p(img_px,img_py,1);
            Eigen::Vector3f cam_p=K.inverse()*img_p;
            float deno=ABC.transpose()*cam_p;
            float k=-D/deno;
            Eigen::MatrixXf ground_p(4,1);
            ground_p.topLeftCorner(3,1)=k*cam_p;
            ground_p(3,0)=1;
            //std::cout<<r2c.inverse()<<std::endl;
            Eigen::Vector4f lane_radar=r2c.inverse()*ground_p;
            cloud.points[cloudidx].x = lane_radar(0,0);
            cloud.points[cloudidx].y = lane_radar(1,0);
            cloud.points[cloudidx].z = lane_radar(2,0);
            cloudidx++;
            }
            //std::string pcdName="/home/kyxu/pcl_test/build/lane"+std::to_string(cluster_num)+".pcd";
            //pcl::io::savePCDFileASCII (pcdName, cloud);
            //std::cerr << "Saved " << cloud.points.size () << " data points to "+pcdName << std::endl;
        }
        //std::string pcdName="/home/kyxu/pcl_test/build/lane_2.pcd";
        pcl::io::savePCDFileASCII (pcdName, cloud);
        std::cerr << "Saved " << cloud.points.size () << " data points to "+pcdName << std::endl;
}
int main(int argc, char** argv) {

    std::string img_dir = "/home/kyxu/lane/mydata/image/";
    std::string model_path = "/home/kyxu/lane/lanenet-lane-detection-master/model/pb_model2/frozen_model.pb";
    LOG(INFO) << "begin to init detector";
    LaneNetDetector::Ptr detector(new LaneNetDetector());
    LOG(INFO) << "begin to init detector";
    detector->init(model_path, 0.5);
    LOG(INFO) << "begin to init detector";
    std::chrono::steady_clock::time_point t1, t2;
    std::chrono::duration<double> time_used;
    cv::Mat lane_ret, image;
    std::vector<std::vector<cv::Point2i> > lane_clusters;
    
    
    for (int i = 1; i <= 3; ++i) {
        std::string img_path = "/home/kyxu/lane/mydata/image/000" + std::to_string(i) + ".png";
        std::cout<<img_path;
        image = cv::imread(img_path);
        cv::imshow("ori_img" + std::to_string(i), image);

        t1 = std::chrono::steady_clock::now();
        detector->detectLane(image, lane_ret, lane_clusters);
        std::string pcdName="/home/kyxu/pcl_test/build/lane"+std::to_string(i)+".pcd";
        clusters2pcd(lane_clusters,pcdName);
        t2 = std::chrono::steady_clock::now();
        time_used = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        std::cout << "Use time: " << time_used.count() << " s." << std::endl;
        cv::imshow("lane" + std::to_string(i), lane_ret);
        int key = cv::waitKey(0);
    }
    int key = cv::waitKey(0);


    return 1;
}