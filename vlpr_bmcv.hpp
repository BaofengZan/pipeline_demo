﻿//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef VLPR_H
#define VLPR_H

#include "./yolov5_multi/yolov5.hpp"
#include "./lprnet_multi/lprnet_bmcv.hpp"


// define data struct
struct demo_config {
    int dev_id;
    std::string yolov5_bmodel_path;
    std::string lprnet_bmodel_path;
    std::vector<std::string> input_paths;
    std::vector<bool> is_videos;
    int yolov5_num_pre;
    int yolov5_num_post;
    int lprnet_num_pre;
    int lprnet_num_post;
    int yolov5_queue_size;
    int lprnet_queue_size;
    float yolov5_conf_thresh;
    float yolov5_nms_thresh;
    int frame_sample_interval;
    int in_frame_num;
    int out_frame_num;
    int crop_thread_num;
    int push_data_thread_num;
    // time interval (s) between runtime performance info out
    int perf_out_time_interval;
};

struct rec_single_frame
{
    rec_single_frame() :cur_num(0) {}
    std::unordered_map<std::string, int> rec_res;
    int cur_num;
    int num;
};


class VLPR {
public:
    VLPR(demo_config& config);
    ~VLPR();
    // start threads
    void run();
    // crop image using boxes generated by yolov5, then send to lprnet
    void crop(int process_id);
    // push reconition results
    void push_data(int process_id);
    // pop reconition results, analyze vehicle license plate
    void worker(int channel_id);

private:
    std::mutex m_mutex_num_map;
    // dict, ele(channel id: {frame_id: recognized result})
    std::unordered_map<int, std::unordered_map<int, std::shared_ptr<rec_single_frame>>> m_num_map;
    // array, ele(recognized result: {last_frame_id, continuous_len})
    std::vector<std::unordered_map<std::string, std::pair<int, int>>> statics;

    // threads
    std::mutex m_mutex_crop;
    int crop_activate_thread_num;
    int push_data_activate_thread_num;
    int channel_num;
    std::mutex m_mutex_total_frame_num;
    int total_frame_num;

    // continuous frame num: in action and out action
    int in_frame_num, out_frame_num;

    // objects
    YOLOv5* yolov5 = nullptr;
    LPRNet* lprnet = nullptr;
    demo_config config;

    // output runtime performance info
    std::condition_variable runtime_performance_info_thread_exit_cv;
    bool runtime_performance_info_thread_exit;
    void output_runtime_performance_info();
};


#endif //!VLPR_H