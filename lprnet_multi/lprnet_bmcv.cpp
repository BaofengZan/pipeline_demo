//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "lprnet_bmcv.hpp"
#include<chrono>
#include<thread>
// rec class
static char const* arr_chars[] = {
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
    "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新", "0",  "1",  "2",  "3",  "4",
    "5",  "6",  "7",  "8",  "9",  "A",  "B",  "C",  "D",  "E",  "F",  "G",
    "H",  "J",  "K",  "L",  "M",  "N",  "P",  "Q",  "R",  "S",  "T",  "U",
    "V",  "W",  "X",  "Y",  "Z",  "I",  "O",  "-" };

int argmax(float* data, int num) {
    float max_value = -1e10;
    int max_index = 0;
    for (int i = 0; i < num; ++i) {
        float value = data[i];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
    }
    return max_index;
}

LPRNet::LPRNet(int dev_id, std::string bmodel_path, int pre_thread_num,
    int post_thread_num, int queue_size)
    : dev_id(dev_id),
    pre_thread_num(pre_thread_num),
    post_thread_num(post_thread_num),
    bmodel_path(bmodel_path),
    m_queue_decode("lprnet_decode_queue", queue_size),
    m_queue_pre("lprnet_pre_queue", queue_size),
    m_queue_infer("lprnet_infer_queue", queue_size),
    m_queue_post("lprnet_post_queue", queue_size) {
    // end flag, decode_activate_thread_num is arbitrary number, and need to set
    // bigger 0
    decode_activate_thread_num = 1;
    pre_activate_thread_num = pre_thread_num;
    infer_activate_thread_num = 1;
    post_activate_thread_num = post_thread_num;

    // 这里可以初始化推理库的一些东西！！！！
    batch_size = 1;
}

LPRNet::~LPRNet() {
    // sync thread
    for (std::thread& t : threads) t.join();
    // free dev mem and struct mem
    // 释放分配的一些内存  或者使用智能指针，不用释放
}

void LPRNet::run() {
    threads.clear();
    auto start = std::chrono::steady_clock::now();
    // start threads
    for (int p = 0; p < pre_thread_num; p++)
        threads.emplace_back(&LPRNet::preprocess, this, p);
    threads.emplace_back(&LPRNet::inference, this);
    for (int p = 0; p < post_thread_num; p++)
        threads.emplace_back(&LPRNet::postprocess, this, p);
}

void LPRNet::preprocess(int process_id) {
    while (true) {
        int batch_idx = 0;
        std::shared_ptr<bmtensor> cur_tensor = std::make_shared<bmtensor>();

        // get batch data
        while (true) {
            std::shared_ptr<bmimage> in;
            if (m_queue_decode.pop_front(in) != 0) break;
            // resize
            cv::Mat image_aligned;
            // 前处理 模拟处理时长
            std::this_thread::sleep_for(std::chrono::milliseconds(3)); // 5ms

            batch_idx++;

            cur_tensor->channel_ids.push_back(in->channel_id);
            cur_tensor->frame_ids.push_back(in->frame_id);

            if (batch_idx == batch_size) break;
        }
        // if not have any image, exit thread
        if (batch_idx == 0) break;

        // converto
        // init images

        // init output tensor
        cur_tensor->bmtensor = std::make_shared<bm_tensor_t>();
        for (int i = 0; i < batch_size; ++i)
        {
            auto ss = std::to_string(i) + " img preprocess";
            cur_tensor->bmtensor->imgs.push_back(ss);
        }
        auto threadID = std::this_thread::get_id();
        std::cout << "[[" << threadID << "]]" << " LPR  预处理 。。。。。。。。。。。。." << std::endl;
        // push data
        m_queue_pre.push_back(cur_tensor);
    }

    std::unique_lock<std::mutex> lock(m_mutex_pre_end);
    pre_activate_thread_num--;
    if (pre_activate_thread_num <= 0) m_queue_pre.set_stop_flag(true);
    std::cout << " LPR  预处理 preprocess thread " << process_id << " exit..." << std::endl;
}

void LPRNet::inference() {
    while (true) {
        std::shared_ptr<bmtensor> in;
        if (m_queue_pre.pop_front(in) != 0) break;

        // init output tensor
        std::shared_ptr<bmtensor> cur_tensor = std::make_shared<bmtensor>();
        cur_tensor->channel_ids = in->channel_ids;
        cur_tensor->frame_ids = in->frame_ids;
        cur_tensor->bmtensor = std::make_shared<bm_tensor_t>();

        for (int i = 0; i < batch_size; ++i)
        {
            auto ss = std::to_string(i) + " img infer";
            cur_tensor->bmtensor->imgs.push_back(ss);
        }

        // forward 模拟
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto threadID = std::this_thread::get_id();
        std::cout << "[[" << threadID << "]]" << " LPR 推理 inference.。。。。。。。。。。。。." << std::endl;
        // push data
        m_queue_infer.push_back(cur_tensor);
    }

    std::unique_lock<std::mutex> lock(m_mutex_infer_end);
    infer_activate_thread_num--;
    if (infer_activate_thread_num <= 0) m_queue_infer.set_stop_flag(true);
    std::cout << "LPR 推理 inference thread exit..." << std::endl;
}

void LPRNet::postprocess(int process_id) {
    while (true) {
        std::shared_ptr<bmtensor> in;
        if (m_queue_infer.pop_front(in) != 0) break;

        int valid_batch_size = in->channel_ids.size();
        for (int idx = 0; idx < valid_batch_size; ++idx) {
            std::shared_ptr<rec_data> result(new rec_data());
            result->channel_id = in->channel_ids[idx];
            result->frame_id = in->frame_ids[idx];

            result->rec_res = "postprocess " + std::to_string(idx);
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            // push data
            auto threadID = std::this_thread::get_id();
            std::cout << "[[" << threadID << "]]" "LPR  后处理 postproces。。。。。。。。。。。。。" << std::endl;
            m_queue_post.push_back(result);
        }
    }

    std::unique_lock<std::mutex> lock(m_mutex_post_end);
    post_activate_thread_num--;
    if (post_activate_thread_num <= 0) m_queue_post.set_stop_flag(true);
    std::cout << "LPR  后处理 postprocess thread " << process_id << " exit..." << std::endl;
}


std::string LPRNet::get_res(int pred_num[]) {
    int no_repeat_blank[20];
    int cn_no_repeat_blank = 0;
    int pre_c = pred_num[0];
    if (pre_c != class_num - 1) {
        no_repeat_blank[0] = pre_c;
        cn_no_repeat_blank++;
    }
    for (int i = 0; i < seq_len; i++) {
        if (pred_num[i] == pre_c) continue;
        if (pred_num[i] == class_num - 1) {
            pre_c = pred_num[i];
            continue;
        }
        no_repeat_blank[cn_no_repeat_blank] = pred_num[i];
        pre_c = pred_num[i];
        cn_no_repeat_blank++;
    }

    std::string res = "";
    for (int j = 0; j < cn_no_repeat_blank; j++) {
        res = res + arr_chars[no_repeat_blank[j]];
    }

    return res;
}

void LPRNet::push_m_queue_decode(std::shared_ptr<bmimage> in) {
    m_queue_decode.push_back(in);
}

void LPRNet::set_preprocess_exit() {
    std::unique_lock<std::mutex> lock(m_mutex_decode_end);
    decode_activate_thread_num = 0;
    m_queue_decode.set_stop_flag(true);
}

int LPRNet::pop_m_queue_post(std::shared_ptr<rec_data>& out) {
    if (m_queue_post.pop_front(out) != 0) return -1;
    return 0;
}