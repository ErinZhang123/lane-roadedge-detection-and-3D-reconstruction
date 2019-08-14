#include "LaneNetDetector.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <tensorflow/c/c_api.h>
#include <glog/logging.h>
#include <cstring>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

std::map<float, std::vector<uint8_t> > gpu_memory_fraction_mapping =
        {{0.0,{0x32,0x00}},
         {0.1,{0x32,0x09,0x09,0x9a,0x99,0x99,0x99,0x99,0x99,0xb9,0x3f}},
         {0.2,{0x32,0x09,0x09,0x9a,0x99,0x99,0x99,0x99,0x99,0xc9,0x3f}},
         {0.3,{0x32,0x09,0x09,0x33,0x33,0x33,0x33,0x33,0x33,0xd3,0x3f}},
         {0.4,{0x32,0x09,0x09,0x9a,0x99,0x99,0x99,0x99,0x99,0xd9,0x3f}},
         {0.5,{0x32,0x09,0x09,0x00,0x00,0x00,0x00,0x00,0x00,0xe0,0x3f}},
         {0.6,{0x32,0x09,0x09,0x33,0x33,0x33,0x33,0x33,0x33,0xe3,0x3f}},
         {0.7,{0x32,0x09,0x09,0x66,0x66,0x66,0x66,0x66,0x66,0xe6,0x3f}},
         {0.8,{0x32,0x09,0x09,0x9a,0x99,0x99,0x99,0x99,0x99,0xe9,0x3f}},
         {0.9,{0x32,0x09,0x09,0xcd,0xcc,0xcc,0xcc,0xcc,0xcc,0xec,0x3f}},
         {1.0,{0x32,0x09,0x09,0x00,0x00,0x00,0x00,0x00,0x00,0xf0,0x3f}}};

void free_buffer(void* data, size_t length) { free(data); }

bool compare_size(const std::vector<cv::Point2i>& vec1, const std::vector<cv::Point2i>& vec2) {
    return vec1.size() > vec2.size();
}

class LaneNetDetectorPrivate {
public:
    LaneNetDetectorPrivate(LaneNetDetector* pthis) : _pthis(pthis) {
        _graph = TF_NewGraph();
        _status = TF_NewStatus();
        _input_size = cv::Size(512, 256);
        _mean_mat = cv::Mat(_input_size, CV_32FC3, cv::Scalar(103.939, 116.779, 123.68));
        _colors = {cv::Vec3b(255, 0, 0), 
                   cv::Vec3b(0, 255, 0), 
                   cv::Vec3b(0, 0, 255), 
                   cv::Vec3b(125, 125, 0), 
                   cv::Vec3b(0, 125, 125), 
                   cv::Vec3b(125, 0, 125), 
                   cv::Vec3b(50, 100, 50), 
                   cv::Vec3b(100, 50, 100)};
    }
    ~LaneNetDetectorPrivate() {
        TF_CloseSession(_session, _status);
        assert(TF_GetCode(_status) == TF_OK);
        TF_DeleteSession(_session, _status);
        assert(TF_GetCode(_status) == TF_OK);
        TF_DeleteStatus(_status);
        TF_DeleteGraph(_graph);
    }

    bool init(const std::string& model_path, int gpu_id, float gpu_memory_fraction) {
        if (!this->LoadGraph(model_path.data(), gpu_id, gpu_memory_fraction)) {
            LOG(ERROR) << "Load graph " << model_path << " failed!";
            return false;
        }

        // setup graph inputs
        TF_Operation *placeholder = TF_GraphOperationByName(_graph, "input_tensor");
        _inputs.push_back({placeholder, 0});
    
        // setup graph outputs
        TF_Operation *binary_seg = TF_GraphOperationByName(_graph, "lanenet_model/Reshape_1");
        TF_Operation *instance_seg = TF_GraphOperationByName(_graph, "lanenet_model/pix_embedding_relu");
        _outputs.push_back({binary_seg, 0});
        _outputs.push_back({instance_seg, 0});
        return true;
    }

    bool inference(const cv::Mat& input_img, cv::Mat& binary_seg_ret, cv::Mat& instance_seg_ret) {
        cv::Mat img_cal;
        this->preProcess(input_img, img_cal);

        // create image tensor
        const int64_t tensorDims[4] = {1, img_cal.rows, img_cal.cols, 3};
        size_t data_len = size_t(img_cal.cols * img_cal.rows * 3 * sizeof(float));
        TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, tensorDims, 4, data_len);
        std::memcpy(TF_TensorData(input_tensor), img_cal.data, std::min(data_len, TF_TensorByteSize(input_tensor)));
        
        std::vector<TF_Tensor *> input_values;
        input_values.push_back(input_tensor);
        std::vector<TF_Tensor *> output_values(_outputs.size(), nullptr);
    
        // session run
        TF_SessionRun(_session, nullptr,
                      &_inputs[0], &input_values[0], int(_inputs.size()),
                      &_outputs[0], &output_values[0], int(_outputs.size()),
                      nullptr, 0, nullptr, _status);
        if (TF_GetCode(_status) != TF_OK) {
            LOG(ERROR) << "Unable to run session " << TF_Message(_status);
            return false;
        }
    
        // collect detect results
        float *binary_seg = static_cast<float *>(TF_TensorData(output_values[0]));
        float *instance_seg = static_cast<float *>(TF_TensorData(output_values[1]));
         
        binary_seg_ret = cv::Mat(img_cal.size(), CV_32FC3, binary_seg);//2019.4.18
        instance_seg_ret = cv::Mat(img_cal.size(), CV_32FC4, instance_seg);

        for (int i = 0; i < output_values.size(); ++i) {
            TF_DeleteTensor(output_values[i]);
        }
        for (int i = 0; i < input_values.size(); ++i) {
            TF_DeleteTensor(input_values[i]);
        }
        
        cv::Mat gray(img_cal.size(), CV_8UC1, cv::Scalar(0));
        cv::Mat gray1(img_cal.size(), CV_8UC1, cv::Scalar(0));//2019.4.18
        std::cout<<gray.size()<<std::endl;
        std::cout<<binary_seg_ret.size()<<std::endl;
        for (int i = 0; i < binary_seg_ret.rows; ++i) {
            float* binary_seg_data = binary_seg_ret.ptr<float>(i);
            
            //float* instance_seg_data = instance_seg_ret.ptr<float>(i);
            for (int j = 0; j < binary_seg_ret.cols; ++j) {
                //LOG(INFO) << i << " " << j << " " << instance_seg_data[j * 4] << " " << 
                //          instance_seg_data[j * 4 + 1] << " " << instance_seg_data[j * 4 + 2] << " " << 
                //          instance_seg_data[j * 4 + 3];
                //2019.4.18
                if (binary_seg_data[j * 3] > binary_seg_data[j * 3 + 1]) {
                    if(binary_seg_data[j * 3]>binary_seg_data[j * 3+2])
                    {gray.at<uchar>(i, j) = 0;gray1.at<uchar>(i, j) = 0;}
                    else {gray.at<uchar>(i,j) = 100;gray1.at<uchar>(i, j) = 255;}

                }
                else if(binary_seg_data[j * 3+1]>binary_seg_data[j * 3+2])
                {gray.at<uchar>(i, j) = 255;gray1.at<uchar>(i, j) = 255;}
                else {gray.at<uchar>(i, j) = 100;gray1.at<uchar>(i, j) =255;}
            
            }
        }
        
        binary_seg_ret = gray1;//2019.4.18
        binaryPostProcess(binary_seg_ret);
        cv::imshow("binary", gray);
        cv::imshow("instance", instance_seg_ret);
        return true;
    }
    void postProcess(const cv::Mat& binary_seg, const cv::Mat& instance_seg, cv::Mat& lane_ret) {
        
        std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > lane_features;
        std::vector<cv::Point2i> lane_idx;
        getLaneFeatures(binary_seg, instance_seg, lane_features, lane_idx);
        
        std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > centers;
        std::vector<int> cluster_idx;
        int num_cluster = meanShift(lane_features, centers, cluster_idx);

        // for (int i = 0; i < centers.size(); ++i) {
        //     for (int j = i + 1; j < centers.size(); ++j) {
        //         LOG(INFO) << i << " " << j << " " << (centers[i]-centers[j]).squaredNorm();
        //     }
        // }

        lane_ret = cv::Mat(binary_seg.size(), CV_8UC3, cv::Scalar(0,0,0));
        for (int i = 0; i < cluster_idx.size(); ++i) {
            if (cluster_idx[i] >= 8) {
                continue;
            }
            lane_ret.at<cv::Vec3b>(lane_idx[i].y, lane_idx[i].x) = _colors[cluster_idx[i]];
        }
    }
    void postProcess(const cv::Mat& binary_seg, const cv::Mat& instance_seg, std::vector<std::vector<cv::Point2i> >& lane_clusters) {
        
        lane_clusters.clear();
        std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > lane_features;
        std::vector<cv::Point2i> lane_idx;
        getLaneFeatures(binary_seg, instance_seg, lane_features, lane_idx);
        
        std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > centers;
        std::vector<int> cluster_idx;
        int num_cluster = meanShift(lane_features, centers, cluster_idx);
        splitCluster(cluster_idx, lane_idx, lane_clusters, num_cluster);
    }
    void postProcess(const cv::Mat& binary_seg, const cv::Mat& instance_seg, cv::Mat& lane_ret, std::vector<std::vector<cv::Point2i> >& lane_clusters) {

        lane_clusters.clear();
        std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > lane_features;
        std::vector<cv::Point2i> lane_idx;
        getLaneFeatures(binary_seg, instance_seg, lane_features, lane_idx);
        
        std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > centers;
        std::vector<int> cluster_idx;
        int num_cluster = meanShift(lane_features, centers, cluster_idx);
        splitCluster(cluster_idx, lane_idx, lane_clusters, num_cluster);

        lane_ret = cv::Mat(binary_seg.size(), CV_8UC3, cv::Scalar(0,0,0));
        for (int i = 0; i < lane_clusters.size(); ++i) {
            if (i >= 8) {
                continue;
            }
            for (int j = 0; j < lane_clusters[i].size(); ++j) {
                lane_ret.at<cv::Vec3b>(lane_clusters[i][j].y, lane_clusters[i][j].x) = _colors[i];
            }
        }
    }


private:
    void preProcess(const cv::Mat& input_img, cv::Mat& img_cal) {
        cv::Mat img;
        
        cv::resize(input_img, img, _input_size);
        cv::Mat img_float;
        img.convertTo(img_float, CV_32FC3);
        img_cal = cv::Mat(img_float.size(), CV_32FC3);
        cv::subtract(img_float, _mean_mat, img_cal);
    }
    bool LoadGraph(const char * model_path, int gpu_id, const float& gpu_memory_fraction) {
        TF_Buffer* graph_def = read_file(model_path);
    
        // import graph_def into graph
        TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
        TF_GraphImportGraphDef(_graph, graph_def, opts, _status);
        TF_DeleteImportGraphDefOptions(opts);
        if (TF_GetCode(_status) != TF_OK) {
            LOG(ERROR) << "Unable to import graph " << TF_Message(_status);
            return false;
        }
        LOG(INFO) << "Successfully imported graph.";
        TF_DeleteBuffer(graph_def);
    
        // create session
        TF_SessionOptions* session_options = TF_NewSessionOptions();
        TF_SetConfig(session_options, (void*)&(gpu_memory_fraction_mapping.at(gpu_memory_fraction)[0]),
                     gpu_memory_fraction_mapping.at(gpu_memory_fraction).size(), _status);
        //uint8_t config[2] ={0x38, 0x1};
        //TF_SetConfig(session_options,(void*)config,2,_status);
        _session = TF_NewSession(_graph, session_options, _status);
        if (TF_GetCode(_status) != TF_OK) {
            LOG(ERROR) << "Unable to create session " << TF_Message(_status);
            return false;
        }
        LOG(INFO) << "Successfully create session.";
        TF_DeleteSessionOptions(session_options);
        return true;
    }
    
    
    TF_Buffer* read_file(const char* file) {
        FILE *f = fopen(file, "rb");
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);  //same as rewind(f);
    
        void* data = malloc(fsize);
        size_t res = fread(data, fsize, 1, f);
        fclose(f);
    
        TF_Buffer* buf = TF_NewBuffer();
        buf->data = data;
        buf->length = fsize;
        buf->data_deallocator = free_buffer;
        return buf;
    }

    void binaryPostProcess(cv::Mat& binary_seg, int area_thresh = 25) {
        // 首先进行形态学运算
        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(binary_seg, binary_seg, cv::MORPH_CLOSE, element);

        // 然后进行连通区域分析, 去除小联通区
        cv::Mat labels, stats, centroids;
        int num_area = cv::connectedComponentsWithStats(binary_seg, labels, stats, centroids);
        for (int i = 0; i < num_area; ++i) {
            if (stats.at<int>(i, cv::CC_STAT_AREA) < area_thresh) {
                for (int m = 0; m < labels.rows; ++m) {
                    for (int n = 0; n < labels.cols; ++n) {
                        if (labels.at<int>(m, n) == i) {
                            binary_seg.at<uchar>(m, n) = 0;
                        }
                    }
                }
            }
        }
    }
    void getLaneFeatures(const cv::Mat& binary_seg, const cv::Mat& instance_feat, 
                         std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& lane_features, 
                         std::vector<cv::Point2i>& lane_idx) {
        lane_features.resize(binary_seg.rows * binary_seg.cols);
        lane_idx.resize(binary_seg.rows * binary_seg.cols);
        int index = 0;
        for (int i = 0; i < binary_seg.rows; ++i) {
            for (int j = 0; j < binary_seg.cols; ++j) {
                if (binary_seg.at<uchar>(i, j) > 100) {
                    lane_idx[index] = cv::Point2i(j, i);
                    lane_features[index++] =  Eigen::Vector4f(instance_feat.at<cv::Vec4f>(i, j)[0],
                                                              instance_feat.at<cv::Vec4f>(i, j)[1], 
                                                              instance_feat.at<cv::Vec4f>(i, j)[2], 
                                                              instance_feat.at<cv::Vec4f>(i, j)[3]);
                }
            }
        }
        lane_features.resize(index);
        lane_idx.resize(index);
    }

    int meanShift(const std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& features, 
                  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& centers,
                  std::vector<int>& cluster_idx, float band_width = 7, float shift_thresh = 5) {
        cluster_idx = std::vector<int>(features.size(), -1);
        centers.clear();
        int num_cluster = 0;
        while (true) {
            bool finish = true;
            int index_tmp_center = -1;
            for (int i = 0; i < cluster_idx.size(); ++i) {
                if (cluster_idx[i] == -1) {
                    finish = false;
                    index_tmp_center = i;
                    break;
                }
            }
            if (finish) {
                break;
            }
            Eigen::Vector4f center_tmp = features[index_tmp_center];
            while (true) {
                Eigen::Vector4f shift_vec(0.,0.,0.,0.);
                int shift_num = 0;
                for (int i = 0; i < cluster_idx.size(); ++i) {
                    if (cluster_idx[i] != -1) {
                        continue;
                    }
                    Eigen::Vector4f dist_vec = features[i] - center_tmp;
                    float dist = dist_vec.squaredNorm();
                    //LOG(INFO) << "dist: " << dist;
                    if (dist < band_width) {
                        cluster_idx[i] = num_cluster;
                        shift_vec += dist_vec;
                        ++shift_num;
                    }
                }
                if (shift_num > 0) {
                    shift_vec /= float(shift_num);
                }
                //LOG(INFO) << shift_num <<  " shift_vec: " << shift_vec.squaredNorm();
                center_tmp += shift_vec;
                if (shift_vec.squaredNorm() < shift_thresh) {
                    bool merge = false;
                    for (int i = 0; i < centers.size(); ++i) {
                        if ((center_tmp - centers[i]).squaredNorm() < band_width) {
                            for (int j = 0; j < cluster_idx.size(); ++j) {
                                if (cluster_idx[j] == num_cluster) {
                                    cluster_idx[j] = i;
                                }
                            }
                            merge = true;
                            break;
                        }
                    }
                    if (!merge) {
                        num_cluster++;
                        centers.push_back(center_tmp);
                    }
                    break;
                }
            }
        }
        return num_cluster;
    }

    void splitCluster(const std::vector<int>& cluster_idx, const std::vector<cv::Point2i>& pts_pos,
                      std::vector<std::vector<cv::Point2i> >& clusters, 
                      int num_cluster, int num_pt_thresh = 60) {
        clusters.resize(num_cluster);
        for (int i = 0; i < cluster_idx.size(); ++i) {
            clusters[cluster_idx[i]].push_back(pts_pos[i]);
        }
        std::sort(clusters.begin(), clusters.end(), compare_size);
        std::vector<std::vector<cv::Point2i> >::iterator iter;
        for (iter = clusters.begin(); iter != clusters.end();) {
            if ((*iter).size() < num_pt_thresh) {
                iter = clusters.erase(iter);
            } else {
                ++iter;
            }
        }
    }


private:
    TF_Status* _status;
    TF_Graph* _graph;
    TF_Session* _session; 
    std::vector<TF_Output> _inputs, _outputs;
    cv::Mat _mean_mat;
    cv::Size _input_size;
    std::vector<cv::Vec3b> _colors;

    LaneNetDetector* _pthis;
    friend class LaneNetDetector;
};

LaneNetDetector::LaneNetDetector() {
    _ptr = new LaneNetDetectorPrivate(this);
}

LaneNetDetector::~LaneNetDetector() {
    if (_ptr) {
        delete _ptr;
        _ptr = nullptr;
    }
}

bool LaneNetDetector::init(const std::string& model_path, float gpu_memory_fraction, int gpu_id) {
    return _ptr->init(model_path, gpu_id, gpu_memory_fraction);
}

bool LaneNetDetector::detectLane(const cv::Mat& input_img, cv::Mat& lane_ret) {
    cv::Mat binary_seg_ret, instance_seg_ret;
    if(!_ptr->inference(input_img, binary_seg_ret, instance_seg_ret)) {
        LOG(ERROR) << "inference error";
        return false;
    } 
    _ptr->postProcess(binary_seg_ret, instance_seg_ret, lane_ret);
    return true;
}
bool LaneNetDetector::detectLane(const cv::Mat& input_img, std::vector<std::vector<cv::Point2i> >& lane_clusters) {
    cv::Mat binary_seg_ret, instance_seg_ret;
    if(!_ptr->inference(input_img, binary_seg_ret, instance_seg_ret)) {
        LOG(ERROR) << "inference error";
        return false;
    } 
    _ptr->postProcess(binary_seg_ret, instance_seg_ret, lane_clusters);
    return true;
}

bool LaneNetDetector::detectLane(const cv::Mat& input_img, cv::Mat& lane_ret, std::vector<std::vector<cv::Point2i> >& lane_clusters) {
    cv::Mat binary_seg_ret, instance_seg_ret;
    if(!_ptr->inference(input_img, binary_seg_ret, instance_seg_ret)) {
        LOG(ERROR) << "inference error";
        return false;
    } 
    _ptr->postProcess(binary_seg_ret, instance_seg_ret, lane_ret, lane_clusters);
    return true;
}




