//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "yolov5.hpp"
#include <fstream>


#define USE_MULTICLASS_NMS 1  // 后处理nms方式
#define FPS 1                 // 是否计算fps
#define PRESSURE 0            // 压测，循环解码

YOLOv5::YOLOv5(int dev_id,
    std::string bmodel_path,
    std::vector<std::string> input_paths,
    std::vector<bool> is_videos,
    std::vector<int> skip_frame_nums,
    int queue_size,
    int num_pre,
    int num_post,
    float confThresh,
    float nmsThresh
) :
    m_dev_id(dev_id),
    m_queue_size(queue_size),
    m_num_decode(input_paths.size()),
    m_num_pre(num_pre),
    m_num_post(num_post),
    m_stop_decode(0),
    m_stop_pre(0),
    m_stop_post(0),
    m_is_stop_decode(false),
    m_is_stop_pre(false),
    m_is_stop_infer(false),
    m_is_stop_post(false),
    m_confThreshold(confThresh),
    m_nmsThreshold(nmsThresh),
    m_queue_decode("yolov5_decode_queue", m_queue_size, 2),
    m_queue_pre("yolov5_pre_queue", m_queue_size, 2),
    m_queue_infer("yolov5_infer_queue", m_queue_size, 2),
    m_queue_post("yolov5_post_queue", m_queue_size, 2)
{

    //初始化 网络的一些东西 内存等 
    m_batch_size = 1;

    // init decode 
    for (int i = 0; i < m_num_decode; i++) {
        auto decode_element = std::make_shared<DecEle>();
        if (is_videos[i]) {
            decode_element->is_video = true;
            decode_element->cap = cv::VideoCapture(input_paths[i], cv::CAP_FFMPEG);
            if (!decode_element->cap.isOpened()) {
                std::cerr << "Error: open video src failed in channel " << i << std::endl;
                exit(1);
            }
            decode_element->dec_frame_idx = 1;
            decode_element->skip_frame_num = skip_frame_nums[i] + 1;
            decode_element->time_interval = 1 / decode_element->cap.get(cv::CAP_PROP_FPS) * 1e+3;

        }
        else {
            std::vector<std::string> image_paths;
            for (const auto& entry : std::filesystem::directory_iterator(input_paths[i])) {
                if (entry.is_regular_file()) {
                    image_paths.emplace_back(entry.path().filename().string());
                }
            }

            decode_element->is_video = false;
            decode_element->dir_path = input_paths[i];
            decode_element->image_name_list = image_paths;
            decode_element->image_name_it = decode_element->image_name_list.begin();
        }
        m_decode_elements.emplace_back(decode_element);
        m_decode_frame_ids.emplace_back(0);
    }


    m_input_paths = input_paths;

    // init pre
    for (int i = 0; i < m_num_pre; i++) {
        std::vector<cv::Mat> resized_bmimgs(m_batch_size);
        m_vec_resized_bmimgs.emplace_back(resized_bmimgs);
    }

#if FPS
    m_start = std::chrono::high_resolution_clock::now();
#endif

    // init decode worker
    for (int i = 0; i < m_num_decode; i++) {
        m_thread_decodes.emplace_back(&YOLOv5::worker_decode, this, i);
        time_counters.emplace_back(std::chrono::high_resolution_clock::now());
        decode_frame_counts.emplace_back(0);
    }

    // init pre worker
    for (int i = 0; i < m_num_pre; i++) {
        m_thread_pres.emplace_back(&YOLOv5::worker_pre, this, i);
    }

    // init infer worker
    // 这里能不能初始化多份模型？？
    // 一张卡上，初始化多个推理线程无意义？  是因为是GPU资源有限吗？
    m_thread_infer = std::thread(&YOLOv5::worker_infer, this);

    // init post worker
    for (int i = 0; i < m_num_post; i++) {
        m_thread_posts.emplace_back(&YOLOv5::worker_post, this);
    }
}


YOLOv5::~YOLOv5()
{
    for (auto& thread : m_thread_decodes) {
        if (thread.joinable())
            thread.join();
    }

    for (auto& thread : m_thread_pres) {
        if (thread.joinable())
            thread.join();
    }

    if (m_thread_infer.joinable())
        m_thread_infer.join();

    for (auto& thread : m_thread_posts) {
        if (thread.joinable())
            thread.join();
    }

#if FPS
    m_end = std::chrono::high_resolution_clock::now();
    auto duration = m_end - m_start;
    int frame_total = 0;
    for (int i = 0; i < m_num_decode; i++) {
        frame_total += get_frame_count(i);
    }
    std::cout << "yolov5 fps: " << frame_total / (duration.count() * 1e-9) << std::endl;
#endif

    // free decode
    for (auto& ele : m_decode_elements) {
        if (ele->is_video)
            ele->cap.release();
    }

}

int YOLOv5::get_frame_count(int channel_id) {
    return m_decode_frame_ids[channel_id];
}



// -------------------------线程函数----------------------------------
void YOLOv5::worker_decode(int channel_id) {
    while (true) {
        auto data = std::make_shared<DataDec>();
        decode(data, channel_id);
        decode_frame_counts[channel_id] += 1;

        // frame_id为-1时代表读到eof，不进行后续处理
        // 只有可以放入的图片才设置frame id，保证frame id是连续的
        if (data->frame_id != -1) {
            // 流控
            auto time_count = std::chrono::high_resolution_clock::now();
            int sleep_time = int(m_decode_elements[channel_id]->time_interval - (time_count - time_counters[channel_id]).count() * 1e-6);
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
            time_counters[channel_id] = time_count;

            // 跳帧
            if (decode_frame_counts[channel_id] % m_decode_elements[channel_id]->skip_frame_num == 0)
            {
                data->frame_id = m_decode_frame_ids[channel_id];
                m_decode_frame_ids[channel_id] += 1;
                m_queue_decode.push_back(data);
                {
                    std::unique_lock<std::mutex> lock(m_mutex_map_origin);
                    m_origin_image[data->channel_id][data->frame_id] = data->image;
                }

                // 保存图片名称，在输入为图片时使用
                if (!m_decode_elements[channel_id]->is_video) {
                    std::unique_lock<std::mutex> lock(m_mutex_map_name);
                    m_image_name[data->channel_id][data->frame_id] = data->image_name;
                }
            }
        }


#if PRESSURE
        if (data->frame_id == -1) {
            std::cout << "channel " << channel_id << " meets eof" << std::endl;
            auto& cap = m_decode_elements[channel_id]->cap;
            cap.release();
            cap.open(m_input_paths[channel_id]);
            if (!cap.isOpened()) {
                std::cerr << "Failed to reopen the video file." << std::endl;
                exit(1);
            }
        }
#else
        // 如果是eof，解码停止
        if (data->frame_id == -1) {
            std::unique_lock<std::mutex> lock(m_mutex_stop_decode);
            m_stop_decode++;
            // 如果所有路解码停止，向后发送信号
            if (m_stop_decode == m_num_decode) {
                m_is_stop_decode = true;
                m_queue_decode.set_stop_flag(true);
            }

            return;
        }
#endif
    }
}

void YOLOv5::worker_pre(int pre_idx) {
    while (true) {
        std::vector<std::shared_ptr<DataDec>> dec_images;
        auto pre_data = std::make_shared<DataInfer>();
        int ret = 0;
        bool no_data = false;

        // 取一个batch的数据做预处理
        for (int i = 0; i < m_batch_size; i++) {
            std::shared_ptr<DataDec> data;
            ret = m_queue_decode.pop_front(data);
            if (ret == 0) {
                dec_images.emplace_back(data);
            }
            else {
                if (i == 0) {
                    no_data = true;
                }
                break;
            }
        }

        // 解码线程停止并且解码队列为空，可以结束工作线程
        if (no_data) {
            std::unique_lock<std::mutex> lock(m_mutex_stop_pre);
            if (m_is_stop_decode && ret == -1) {
                m_stop_pre++;
                if (m_stop_pre == m_num_pre) {
                    m_is_stop_pre = true;
                    m_queue_pre.set_stop_flag(true);
                }
                return;
            }
        }

        preprocess(dec_images, pre_data, pre_idx);
        m_queue_pre.push_back(pre_data);

    }
}


void YOLOv5::worker_infer() {
    while (true) {
        auto input_data = std::make_shared<DataInfer>();
        auto output_data = std::make_shared<DataInfer>();

        auto ret = m_queue_pre.pop_front(input_data);

        // 预处理线程停止并且预处理队列为空，可以结束工作线程
        if (m_is_stop_pre && ret == -1) {
            m_is_stop_infer = true;
            m_queue_infer.set_stop_flag(true);
            return;
        }

        inference(input_data, output_data);
        m_queue_infer.push_back(output_data);

    }
}


void YOLOv5::worker_post() {
    while (true) {
        auto output_data = std::make_shared<DataInfer>();
        std::vector<std::shared_ptr<DataPost>> box_datas;

        auto ret = m_queue_infer.pop_front(output_data);

        {
            std::unique_lock<std::mutex> lock(m_mutex_stop_post);
            if (m_is_stop_infer && ret == -1) {
                m_stop_post++;
                if (m_stop_post == m_num_post) {
                    m_is_stop_post = true;
                    m_queue_post.set_stop_flag(true);
                }
                return;
            }
        }


        postprocess(output_data, box_datas);

        for (int i = 0; i < box_datas.size(); i++) {
            m_queue_post.push_back(box_datas[i]);
        }

    }
}



// ------------------------------处理函数---------------------------

// 调对应的vectore中的decoder
void YOLOv5::decode(std::shared_ptr<DataDec> data, int channel_id) {
    auto decode_ele = m_decode_elements[channel_id];
    cv::Mat image;

    if (decode_ele->is_video) {
        decode_ele->cap.read(image);

        // eof返回frame_id -1;
        if (image.empty()) {
            data->frame_id = -1;
        }
        else {
            data->image = image;
            data->channel_id = channel_id;
            data->frame_id = 0;
        }

    }
    else {
        if (decode_ele->image_name_it == decode_ele->image_name_list.end()) {
            data->frame_id = -1;
        }
        else {
            std::string name = *decode_ele->image_name_it;
            std::string image_path = decode_ele->dir_path + name;
            image = cv::imread(image_path, cv::IMREAD_COLOR);
            data->image = image;
            data->channel_id = channel_id;
            data->frame_id = 0;
            data->image_name = name;
            decode_ele->image_name_it++;
        }
    }
}

// for循环处理多batch，dec_images.size()代表有效数据的数量
void YOLOv5::preprocess(std::vector<std::shared_ptr<DataDec>>& dec_images,
    std::shared_ptr<DataInfer> pre_data, int idx) {

    auto resized_bmimgs = m_vec_resized_bmimgs[idx];

    auto threadID = std::this_thread::get_id();
    std::cout << "[[" << threadID << "]]" << "yolo 预处理 ..." << std::endl;
    // resize需要单图做，但convertto不需要
    for (int i = 0; i < dec_images.size(); i++) {
        auto dec_image = dec_images[i];

        // 预处理
        std::this_thread::sleep_for(std::chrono::milliseconds(3)); //3ms
    }


    for (int i = 0; i < dec_images.size(); i++) {
        pre_data->channel_ids.emplace_back(dec_images[i]->channel_id);
        pre_data->frame_ids.emplace_back(dec_images[i]->frame_id);

        std::string tensor = "yolo 预处理 " + std::to_string(i);

        pre_data->tensors.emplace_back(tensor);
    }



}



void YOLOv5::inference(std::shared_ptr<DataInfer> input_data, std::shared_ptr<DataInfer> output_data) {

    output_data->channel_ids.assign(input_data->channel_ids.begin(), input_data->channel_ids.end());
    output_data->frame_ids.assign(input_data->frame_ids.begin(), input_data->frame_ids.end());
    auto threadID = std::this_thread::get_id();
    std::cout << "[[" << threadID << "]]"  "【yolo infer================】" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(20)); // 5s
}


void YOLOv5::postprocess(std::shared_ptr<DataInfer> output_infer,
    std::vector<std::shared_ptr<DataPost>>& box_data) {

    auto threadID = std::this_thread::get_id();
    std::cout << "[[" << threadID << "]]" << "yolo 后处理====================================" << std::endl;
    YoloV5BoxVec yolobox_vec;
    std::vector<cv::Rect> bbox_vec;
    auto output_tensors = output_infer->tensors;
    int image_nums = output_infer->channel_ids.size();

    for (int batch_idx = 0; batch_idx < image_nums; ++batch_idx)
    {
        YoloV5Box box;
        box.x = 0;
        box.y = 0;
        box.width = 100;
        box.height = 120;
        yolobox_vec.push_back(box);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::shared_ptr<DataPost> data_post = std::make_shared<DataPost>();
        data_post->channel_id = output_infer->channel_ids[batch_idx];
        data_post->frame_id = output_infer->frame_ids[batch_idx];
        data_post->boxes = yolobox_vec;
        box_data.emplace_back(data_post);

    }

}



int YOLOv5::get_post_data(std::shared_ptr<DataPost>& box_data, std::shared_ptr<cv::Mat>& origin_image, std::string& image_name) {

    auto ret = m_queue_post.pop_front(box_data);

    if (ret == -1) {
        return 1;
    }
    int channel_id = box_data->channel_id;
    int frame_id = box_data->frame_id;
    {
        std::unique_lock<std::mutex> lock(m_mutex_map_origin);
        origin_image = std::make_shared<cv::Mat>(m_origin_image[channel_id][frame_id]);
        m_origin_image[channel_id].erase(frame_id);
    }

    if (m_decode_elements[channel_id]->is_video) {
        image_name = std::to_string(channel_id) + '_' + std::to_string(frame_id) + ".jpg";
    }
    else {
        std::unique_lock<std::mutex> lock(m_mutex_map_name);
        image_name = m_image_name[channel_id][frame_id];
        m_image_name[channel_id].erase(frame_id);
    }

    return 0;
}

int YOLOv5::get_post_data(std::shared_ptr<DataPost>& box_data, std::shared_ptr<cv::Mat>& origin_image) {
    std::string name;
    return get_post_data(box_data, origin_image, name);
}



