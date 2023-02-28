/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <stdio.h>
#include <queue>
#include <stack>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

Estimator estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
queue<sensor_msgs::ImageConstPtr> debugimg_buf;
stack<dvs_msgs::Event> event_buf, event_buf_pre;
std::mutex m_buf, e_buf;

int event_dt = 20, event_sum = 5000;
double DT=0;

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}

void debug_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    debugimg_buf.push(img_msg);
    m_buf.unlock();
}

void event_callback(const dvs_msgs::EventArray::ConstPtr &msg)
{
    e_buf.lock();
    for (const auto &event : msg->events)
    {
        dvs_msgs::Event _event = event;
        _event.ts = msg->header.stamp;
        event_buf.push(_event);
    }
    // int n=msg->events.size();
    // cout<<"event array pushed ["<<n<<"] units"<<endl;
    e_buf.unlock();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}
double dTimeWindow(int event_pre){
    if(event_pre>=event_sum){
        return event_dt*(event_sum/event_pre);
    }
    return event_dt;
}
// extract images with same timestamp from two topics
void sync_process()
{
    while (1)
    {
        if (STEREO)
        {
            cv::Mat image0, image1, image_event;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                // 0.003s sync tolerance
                if (time0 < time1 - 0.003)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if (time0 > time1 + 0.003)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                    // printf("find img0 and img1\n");
                    cv::Size img_size;
                    img_size = image1.size();
                    image_event = cv::Mat::zeros(img_size, CV_16UC1);
                }
            }
            m_buf.unlock();
            if (!image0.empty())
                estimator.inputImage(time, image0, image_event, image1);
        }
        else
        {
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            // event
            cv::Size img_size(event_COL,event_ROW);
            cv::Mat event_img, debug_img;
            m_buf.lock();
            if (!img0_buf.empty())
            {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
                if (!debugimg_buf.empty())
                {
                    debug_img = getImageFromMsg(debugimg_buf.front());
                    debugimg_buf.pop();
                }
                // cout<<"start accumulate event image..."<<endl;
                // img_size = image.size();
                event_img = cv::Mat::zeros(img_size, CV_16UC1);
                int acc_nums = 0, acc_pre_nums = 0;
                // if(event_buf.empty())
                //     cout<<"the event array is empty"<<endl;

                e_buf.lock();
                int t_aft = 0, t_pre = 0;
                int event_n = event_buf.size();
                stack<dvs_msgs::Event> temps, event_pre_temps;
                while (!event_buf.empty())
                {
                    dvs_msgs::Event _event = event_buf.top();
                    //dt (ns)->(ms)
                    double dt =(header.stamp - _event.ts).toNSec() / 1000000;
                    if(abs(dt)>=(event_dt*5)){
                        ROS_ERROR_THROTTLE(1,"Event accumulator: dt = %d , time stamp not aligned",dt);
                        break;
                    }else{
                        ROS_DEBUG_THROTTLE(1,"Event accumulator: adding event into event_frame");
                    }
                    if(dt >=0){
                        t_pre++;
                        //限定叠加总数
                        if (acc_nums <= event_sum)
                        {  
                            //限定叠加时间窗口
                            if (dt <= dTimeWindow(event_n))
                            {
                                acc_nums++;
                                int x = _event.x;
                                int y = _event.y;
                                ++event_img.at<uint16_t>(cv::Point(x, y));
                                event_pre_temps.push(_event);
                            }else if(acc_nums<= 10000){
                                acc_nums++;
                                int x = _event.x;
                                int y = _event.y;
                                ++event_img.at<uint16_t>(cv::Point(x, y));
                                event_pre_temps.push(_event);
                            }else{
                                // do nothing
                            }
                        }
                    }else{
                    // 发现消息队列中经常会收到不少时间戳在图像之后的事件流，把他们保存起来留着下一轮使用可以增加更多用于合成的事件
                        temps.push(_event);
                        t_aft++;
                    }
                    event_buf.pop();
                }
                if(event_n<=(event_sum/2)){
                    //总数过少，则认为运动程度太低，激励不足，则使用上一轮叠加的事件补充进去
                    while (!event_buf_pre.empty()){
                        dvs_msgs::Event _event = event_buf_pre.top();
                        double dt = (header.stamp - _event.ts).toNSec() / 1000000;
                        if (acc_nums <= event_sum)
                        {
                            if (dt <= (event_dt*3)){
                                acc_nums++;
                                acc_pre_nums++;
                                int x = _event.x;
                                int y = _event.y;
                                ++event_img.at<uint16_t>(cv::Point(x, y));
                                event_pre_temps.push(_event);
                            }
                        }
                        event_buf_pre.pop();
                    }
                }
                ROS_DEBUG("Image recived, event accumulated,headerT=%d,t_pre=%d,t_aft=%d , accumulated=%d , pre used=%d", header.stamp.toNSec(), t_pre, t_aft,acc_nums,acc_pre_nums);

                swap(temps, event_buf);
                //把当前轮已经叠加好的数据暂存起来，供下一轮激励不足时使用
                swap(event_pre_temps, event_buf_pre);
                e_buf.unlock();
            }
            m_buf.unlock();
            // if(event_img.data){
            //     // int y=event_img.rows;
            //     // int x=event_img.cols;
            //     // for(int X=221;X<x;++X){
            //     //     for(int Y=0;Y<y;++Y){
            //     //         if(event_img.at<uchar>(cv::Point(X, Y)) > 0  ){
            //     //             ROS_INFO_THROTTLE(0.01,"found a point >220:X=%d,Y=%d ",X,Y);
            //     //         }
            //     //     }
            //     // }
            //     // cv::Mat resized_eventframe;
            //     // float scaleW = 2;
            //     // float scaleH = 1;
            //     // int width_re=static_cast<float>(event_img.cols*scaleW);
            //     // int height_re=static_cast<float>(event_img.rows*scaleH);
            //     // resize(event_img,resized_eventframe,cv::Size(width_re,height_re));
            //     cv::imwrite("./frameevent.jpg",event_img);
            //     // cv::imwrite("./frameevent_resize.jpg",resized_eventframe);
            // }

            cv::normalize(event_img, event_img, 0, 65535, cv::NORM_MINMAX);

            if (!image.empty())
            {
                // cv::imwrite("./frameevent_normalized.jpg",event_img);
                cv::Mat res;
                event_img.convertTo(res, CV_8UC1, 255.0 / 65535);
                // cv::imwrite("./frameevent_cv8.jpg",res);
                // if(!res.empty()){
                //     bool isNull=true;
                //     int y=event_img.rows;
                //     int x=event_img.cols;
                //     for(int X=221;X<x;++X){
                //         for(int Y=0;Y<y;++Y){
                //             if(event_img.at<uint8_t>(cv::Point(X, Y)) > 0  ){
                //                 isNull=false;
                //                 break;
                //             }
                //         }
                //     }
                //     if(isNull){
                //         cout<<" a null event frame contains nothing"<<endl;
                //     }
                // }else{
                //     cout<<"we have published a null event frame"<<endl;
                // }
                // event_img = cv.cvtColor(event_img, cv.COLOR_BGR2GRAY)
                estimator.inputImage(time, image, res);
                // estimator.inputImage(time,res,image);
                // estimator.inputImage(time,image,debug_img);
                // estimator.inputImage(time,debug_img,image);
            }
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    // 没看到使用这个功能
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if (feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            // printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        // ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        // ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        // ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        // ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if (argc != 2)
    {
        printf("please intput: rosrun vins vins_node [config file] \n"
               "for example: rosrun vins vins_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    // string config_file = "/src/VINS-Fusion/config/davis/agilex.yaml";
    printf("config_file: %s\n", argv[1]);

    readParameters(config_file);
    estimator.setParameter();

    if (EVENT_DT != 0 && EVENT_SUM != 0)
    {
        event_dt = EVENT_DT;
        event_sum = EVENT_SUM;
        std::cout << "event_dt = "<<event_dt<<" event_sum = "<<event_sum<<endl; 
    }
    else
    {
        std::cout << "event settings has been set to default 25/5000 , please check config yaml file" << endl;
    }

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu;
    if (USE_IMU)
    {
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
    ros::Subscriber sub_img1;

    if (STEREO)
    {
        sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);
    }
    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);
    ros::Subscriber sub_event = n.subscribe(EVENT_TOPIC, 100, event_callback);
    ros::Subscriber sub_debugimage = n.subscribe(DEBUG_IMAGE_TOPIC, 100, debug_callback);

    std::thread sync_thread{sync_process};
    ros::spin();

    return 0;
}
