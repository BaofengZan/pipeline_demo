﻿//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef YOLOV5_MULTI_H
#define YOLOV5_MULTI_H

#include <string>
#include <queue>
#include <thread>
#include <vector>
#include <mutex>
#include <filesystem> 
// filesystem 属于 c++17

#include "../json.hpp"
#include "../datapipe.hpp"
#include "opencv2/opencv.hpp"


using json = nlohmann::json;

struct YoloV5Box {
    float x, y, width, height;
    float score;
    int class_id;
};

using YoloV5BoxVec = std::vector<YoloV5Box>;

// 视频解码or图片解码
struct DecEle {
    bool is_video;
    // 视频用
    cv::VideoCapture cap;
    int dec_frame_idx;
    int skip_frame_num;
    double time_interval;

    // 图片用
    std::string dir_path;
    std::vector<std::string> image_name_list;
    std::vector<std::string>::iterator image_name_it;
};

// 解码队列数据
struct DataDec
{
    int channel_id;
    int frame_id;
    cv::Mat image;
    std::string image_name;
};

// 预处理和推理队列数据
struct DataInfer {
    std::vector<int> channel_ids;
    std::vector<int> frame_ids;
    //std::vector<bm_tensor_t> tensors;
    std::vector<std::string> tensors;
};

// 后处理队列数据
struct DataPost {
    int channel_id;
    int frame_id;
    YoloV5BoxVec boxes;
};


class YOLOv5
{

public:
    YOLOv5(int dev_id,                            // 设备id
        std::string bmodel_path,                // 模型路径
        std::vector<std::string> input_paths,   // 输入图片或视频路径
        std::vector<bool> is_videos,            // 视频标志位
        std::vector<int> skip_frame_nums,       // 跳帧
        int queue_size,                         // 缓存队列长度
        int num_pre,                            // 预处理线程数
        int num_post,                           // 后处理线程数
        float confThresh,                       // 置信度
        float nmsThresh);                       // nms
    ~YOLOv5();


    // 线程停止返回1，正常取数据返回0
    // box_data为返回的检测结果，origin_image为对应的图片
    int get_post_data(std::shared_ptr<DataPost>& box_data, std::shared_ptr<cv::Mat>& origin_image, std::string& image_name);
    int get_post_data(std::shared_ptr<DataPost>& box_data, std::shared_ptr<cv::Mat>& origin_image);


private:
    int get_frame_count(int channel);

    std::vector<std::thread> m_thread_decodes;
    std::vector<std::thread> m_thread_pres;
    std::thread m_thread_infer;
    //std::vector<std::thread> m_thread_infer_v;
    std::vector<std::thread> m_thread_posts;


    // 不同路共用相同缓冲队列，以channel_id区分
    int m_queue_size;
    // std::queue<std::shared_ptr<DataDec>> m_queue_decode;
    // std::queue<std::shared_ptr<DataInfer>> m_queue_pre;
    // std::queue<std::shared_ptr<DataInfer>> m_queue_infer;
    // std::queue<std::shared_ptr<DataPost>> m_queue_post;

    DataPipe<std::shared_ptr<DataDec>> m_queue_decode;
    DataPipe<std::shared_ptr<DataInfer>> m_queue_pre;
    DataPipe<std::shared_ptr<DataInfer>> m_queue_infer;
    DataPipe<std::shared_ptr<DataPost>> m_queue_post;

    // 原图<channel_id, frame_id, cv::mat>
    std::unordered_map<int, std::unordered_map<int, cv::Mat>> m_origin_image;
    std::mutex m_mutex_map_origin;


    std::mutex m_mutex_queue_decode;
    std::mutex m_mutex_queue_pre;
    std::mutex m_mutex_queue_infer;
    std::mutex m_mutex_queue_post;

    int m_num_decode;
    int m_num_pre;
    int m_num_post;

    // 多线程已停止的数量，以及对应的互斥锁
    int m_stop_decode;
    int m_stop_pre;
    int m_stop_post;

    std::mutex m_mutex_stop_decode;
    std::mutex m_mutex_stop_pre;
    std::mutex m_mutex_stop_post;

    bool m_is_stop_decode;
    bool m_is_stop_pre;
    bool m_is_stop_infer;
    bool m_is_stop_post;

    // 网络用的stage0，名字也需要自己传入才行，见构造函数
    // 一个类一个推理线程，重复的推理线程(模型)在同一张卡没有意义
    // 模型相关
    int m_dev_id;

    int m_batch_size;
    int m_net_w;
    int m_net_h;
    int m_output_num;
    std::vector<float> m_output_scales;


    // config
    float m_confThreshold;
    float m_nmsThreshold;
    std::vector<std::string> m_input_paths;
    // ...

    // decode
    std::vector<std::shared_ptr<DecEle>> m_decode_elements;
    // 流控
    std::vector<std::chrono::high_resolution_clock::time_point> time_counters;
    // 跳帧计数
    std::vector<int> decode_frame_counts;
    // 解码计算frame_id用
    std::vector<int> m_decode_frame_ids;

    // pre
    std::vector<std::vector<cv::Mat>> m_vec_resized_bmimgs;

    // post
    int m_class_num;


    // other
    // 是否是soc模式
    bool can_mmap;
    // 保存图片和json
    std::mutex m_mutex_map_name;
    std::unordered_map<int, std::unordered_map<int, std::string>> m_image_name;
    std::vector<json> results_json;


    // 线程函数
    void worker_decode(int channel_id);
    void worker_pre(int pre_idx);
    void worker_infer();
    void worker_post();
    // ...

    // 处理函数
    void decode(std::shared_ptr<DataDec> data, int channel_id);
    void preprocess(std::vector<std::shared_ptr<DataDec>>& dec_images, std::shared_ptr<DataInfer> pre_data, int pre_idx);
    void inference(std::shared_ptr<DataInfer> input_data, std::shared_ptr<DataInfer> output_data);
    void postprocess(std::shared_ptr<DataInfer> output_data, std::vector<std::shared_ptr<DataPost>>& box_data);
    // ...

    // 计时
    std::chrono::high_resolution_clock::time_point m_start;
    std::chrono::high_resolution_clock::time_point m_end;

};


#endif