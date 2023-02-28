/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator(): f_manager{Rs} ,event_manager{Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState()
{
    mProcess.lock();
    while(!accBuf.empty())
        accBuf.pop();
    while(!gyrBuf.empty())
        gyrBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();
    while(!event_featureBuf.empty())
        event_featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }
    tie[0] = Vector3d::Zero();
    rie[0] = Matrix3d::Identity();

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    VisualInit = ToDoinit;
    EventInit = ToDoinit;
    initial_timestamp = 0;
    all_image_frame.clear();
    all_event_frame.clear();
    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();
    event_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
}

void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
        tie[i]=TIE[i];
        rie[i]=RIE[i];
        cout << " exitrinsic event cam " << i << endl  << rie[i] << endl << tie[i].transpose() << endl;
    }
    // tie[0]=TIE[0];
    // rie[0]=RIE[0];
    f_manager.setRic(ric);
    event_manager.setRic(rie);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);
    event_featureTracker.readIntrinsicParameter(CAM_NAMES);

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
        std::cout << "ProcessThread is running..." << endl;
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if(!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if(USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if(USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }
        
        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if(restart)
    {
        clearState();
        setParameter();
    }
}
/**
 * 输入一帧图像
 * 1、featureTracker，提取当前帧特征点
 * 2、添加一帧特征点，processMeasurements处理
*/
void Estimator::inputImage(double t, const cv::Mat &_img,  const cv::Mat &_event_img , const cv::Mat &_img1)
{
    // pubEventimg(_event_img,t);
     /**
     * 跟踪一帧图像，提取当前帧特征点
     * 1、用前一帧运动估计特征点在当前帧中的位置，如果特征点没有速度，就直接用前一帧该点位置
     * 2、LK光流跟踪前一帧的特征点，正反向，删除跟丢的点；如果是双目，进行左右匹配，只删右目跟丢的特征点
     * 3、对于前后帧用LK光流跟踪到的匹配特征点，计算基础矩阵，用极线约束进一步剔除outlier点（代码注释掉了）
     * 4、如果特征点不够，剩余的用角点来凑；更新特征点跟踪次数
     * 5、计算特征点归一化相机平面坐标，并计算相对与前一帧移动速度
     * 6、保存当前帧特征点数据（归一化相机平面坐标，像素坐标，归一化相机平面移动速度）
     * 7、展示，左图特征点用颜色区分跟踪次数（红色少，蓝色多），画个箭头指向前一帧特征点位置，如果是双目，右图画个绿色点
    */
     inputImageCnt++;
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> event_featureFrame;
    TicToc featureTrackerTime;

    if(_img1.empty()){
        featureFrame = featureTracker.trackImage(t, _img);
        //12.10在取得图像路标点同时获取事件路标
        event_featureFrame = event_featureTracker.trackImage(t, _event_img);
    }else{
        featureFrame = featureTracker.trackImage(t, _img, _img1);
    //printf("featureTracker time: %f\n", featureTrackerTime.toc());
    }

    if (SHOW_TRACK)
    {
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
        cv::Mat EventimgTrack = event_featureTracker.getTrackImage();
        pubEventTrackImage(EventimgTrack, t);
    }
    
    if(MULTIPLE_THREAD)  
    {     
        if(inputImageCnt % 2 == 0)
        {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame));
            event_featureBuf.push(make_pair(t, event_featureFrame));
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        event_featureBuf.push(make_pair(t, event_featureFrame));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
    
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
       // 使用上一时刻的姿态进行快速的imu预积分
        // 用来预测最新P,V,Q的姿态
        // -latest_p,latest_q,latest_v,latest_acc_0,latest_gyr_0 最新时刻的姿态。
        //这个的作用是为了刷新姿态的输出，但是这个值的误差相对会比较大，是未经过非线性优化获取的初始值。
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}
//目前看并没有用这个 2022.12
void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{

    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if(!MULTIPLE_THREAD)
        processMeasurements();
}


bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if(t1 <= accBuf.back().first)
    {
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

bool Estimator::IMUAvailable(double t)
{
    if(!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

void Estimator::processMeasurements()
{
    //处理imu的量测，特征点等的线程
    //主要还是处理图像特征点
    while (1)
    {
        //printf("process measurments\n");
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > event_feature;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        // ROS_INFO_THROTTLE(1,"Estimator::processMeasurements()--> featureBuf.size= %i,event_fBuf.size=%i",featureBuf.size(),event_featureBuf.size());
        if(!featureBuf.empty()&&!event_featureBuf.empty())
        {
            feature = featureBuf.front();
            event_feature = event_featureBuf.front();
            curTime = feature.first + td;
            while(1)
            {
                if ((!USE_IMU  || IMUAvailable(feature.first + td)||IMUAvailable(event_feature.first + td)))
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            mBuf.lock();
            if(USE_IMU)
                getIMUInterval(prevTime, curTime, accVector, gyrVector);

            featureBuf.pop();
            event_featureBuf.pop();
            mBuf.unlock();

            if(USE_IMU)
            {
                if(!initFirstPoseFlag)
                    initFirstIMUPose(accVector);
                for(size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }
            mProcess.lock();
            processImage(feature.second, event_feature.second, feature.first);
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubEventCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            mProcess.unlock();
        }

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}


void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &event_image , const double header)
{
    /* addFeatureCheckParallax
对当前帧与之前帧进行视差比较，如果是当前帧变化很小，就会删去倒数第二帧，如果变化很大，就删去最旧的帧。并把这一帧作为新的关键帧
这样也就保证了划窗内优化的,除了最后一帧可能不是关键帧外,其余的都是关键帧
VINS里为了控制优化计算量，在实时情况下，只对当前帧之前某一部分帧进行优化，而不是全部历史帧。局部优化帧的数量就是窗口大小。
为了维持窗口大小，需要去除旧的帧添加新的帧，也就是边缘化 Marginalization。到底是删去最旧的帧（MARGIN_OLD）还是删去刚
刚进来窗口倒数第二帧(MARGIN_SECOND_NEW)
如果大于最小像素,则返回true */
/**
 * 添加特征点记录，并检查当前帧是否为关键帧
 * @param frame_count   当前帧在滑窗中的索引
 * @param image         当前帧特征（featureId，cameraId，feature）
*/
//12.27暂时先做成事件相机的帧索引与传统相机一样，一起边缘化
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu, event feature points %lu", image.size(),event_image.size());
    bool visual_aFCP, event_aFCP;
    visual_aFCP = f_manager.addFeatureCheckParallax(frame_count, image, td);
    event_aFCP = event_manager.addFeatureCheckParallax(frame_count, event_image, td);
    if (visual_aFCP && event_aFCP)
    // &&event_manager.addFeatureCheckParallax(frame_count, event_image, td))
    {
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d, number of event_feature: %d", f_manager.getFeatureCount(),event_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    ImageFrame eventframe(event_image, header);
    imageframe.pre_integration = tmp_pre_integration;
    eventframe.pre_integration = tmp_pre_integration;

    all_image_frame.insert(make_pair(header, imageframe));
    all_event_frame.insert(make_pair(header, eventframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if(ESTIMATE_EXTRINSIC == 2)
    {//外参没有标定则先标定,这里有Ric，还需要Rc->event或者rie
    
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        //初始化在开始时和出现detectfailure时通过clearstate()触发
        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    result = initialStructure();
                    initial_timestamp = header;   
                }
                if(result)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_WARN("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // // stereo + IMU initilization
        // if(STEREO && USE_IMU)
        // {
        //     f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        //     f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        //     if (frame_count == WINDOW_SIZE)
        //     {
        //         map<double, ImageFrame>::iterator frame_it;
        //         int i = 0;
        //         for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
        //         {
        //             frame_it->second.R = Rs[i];
        //             frame_it->second.T = Ps[i];
        //             i++;
        //         }
        //         solveGyroscopeBias(all_image_frame, Bgs,GyroBiasInit);
        //         for (int i = 0; i <= WINDOW_SIZE; i++)
        //         {
        //             pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
        //         }
        //         optimization();
        //         updateLatestStates();
        //         solver_flag = NON_LINEAR;
        //         slideWindow();
        //         ROS_INFO("Initialization finish!");
        //     }
        // }

        // // stereo only initilization
        // if(STEREO && !USE_IMU)
        // {
        //     f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        //     f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        //     optimization();

        //     if(frame_count == WINDOW_SIZE)
        //     {
        //         optimization();
        //         updateLatestStates();
        //         solver_flag = NON_LINEAR;
        //         slideWindow();
        //         ROS_INFO("Initialization finish!");
        //     }
        // }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {
        //通过初始化后的工作流程，若有传感器没通过初始化则把该传感器单独初始化
        TicToc t_solve;
        // if(!USE_IMU){
        //     f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        // }
        
        RetryInitAlign();
        int initialflag = InitFlagSwitcher();
        switch (initialflag){
            case 1:     
                f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
                event_manager.triangulate(frame_count, Ps, Rs, tie, rie);
                break;
            case 2:
                event_manager.triangulate(frame_count, Ps, Rs, tie, rie);
                break;
            case 3:
                f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
                break;
            default:
            break;
        }

        optimization();
        set<int> removeIndex,event_removeIndex;
        outliersRejection(removeIndex,event_removeIndex);
        switch (initialflag){
            case 1:     
                f_manager.removeOutlier(removeIndex);
                event_manager.removeOutlier(event_removeIndex);
                break;
            case 2:
                event_manager.removeOutlier(event_removeIndex);
                break;
            case 3:
                f_manager.removeOutlier(removeIndex);
                break;
            default:
            break;
        }

        // if (! MULTIPLE_THREAD)
        // {
        //     featureTracker.removeOutliers(removeIndex);
        //     predictPtsInNextFrame();
        // }
            
        ROS_DEBUG("solver costs: %fms, initialflag= %d", t_solve.toc(),initialflag);

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        slideWindow();
        switch (initialflag){
            case 1:     
                f_manager.removeFailures();
                event_manager.removeFailures();
                break;
            case 2:
                event_manager.removeFailures();
                break;
            case 3:
                f_manager.removeFailures();
                break;
            default:
            break;
        }
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }  
}

bool Estimator::initialStructure()
{
    //2022.12.27先做成两个帧一起sfm对齐的
    //前面这些都是针对使用imu和传统帧计算加速度角速度平均值的
    //2023.2.10 两种帧分别按各自流程初始化
    TicToc t_rel,t_sfm,t_esfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // 从第2帧开始累加每帧加速度
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        // 加速度均值
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        // 加速度标准差
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        ROS_WARN("IMU variation %f!", var);
         if(var < 0.25)
        {
            ROS_INFO_THROTTLE(1,"IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    t_rel.tic();
    Quaterniond Q[frame_count + 1], event_Q[frame_count + 1];
    Vector3d T[frame_count + 1], event_T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points, event_sfm_tracked_points;
    vector<SFMFeature> sfm_f, event_sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    for (auto &it_per_id : event_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        event_sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R, event_relative_R;
    Vector3d relative_T, event_relative_T;
    int l,event_l;
    if(relativePose(relative_R, relative_T, l)){
        VisualInit = RelaPass ;
        ROS_INFO("VisualInit = RelaPass ");
    }
    if(event_relativePose(event_relative_R, event_relative_T, event_l)){
        EventInit = RelaPass ;
        ROS_INFO("EventInit = RelaPass ");
    }
    // if (!relativePose(relative_R, relative_T, l)&&!event_relativePose(event_relative_R, event_relative_T, event_l))
    if(VisualInit != RelaPass && EventInit != RelaPass ){
        ROS_INFO("Not enough features or parallax; Move device around");
        VisualInit = ToDoinit;
        EventInit = ToDoinit;
        return false;
    }
    ROS_INFO("relativePose costs: %fms", t_rel.toc());
    t_sfm.tic();
    //2. slove SFM 
    GlobalSFM sfm, event_sfm ;
    if(VisualInit == RelaPass){
        bool res=sfm.construct(frame_count + 1, Q, T, l,relative_R, relative_T,sfm_f, sfm_tracked_points);
        ROS_INFO("Visual SFM costs: %fms", t_sfm.toc());
        if(res){
            VisualInit = SFMPass;
            ROS_INFO("visual SFM passed!");
        }else{
            ROS_INFO("Visual SFM failed!");
            marginalization_flag = MARGIN_OLD;
        }
    }
    // if(!sfm.construct(frame_count + 1, Q, T, l,
    //           relative_R, relative_T,
    //           sfm_f, sfm_tracked_points))
    // {
    //     ROS_INFO("global SFM failed!");
    //     marginalization_flag = MARGIN_OLD;
    //     return false;
    // }
    t_esfm.tic();
    if(EventInit == RelaPass){
        bool res=event_sfm.construct(frame_count + 1, event_Q, event_T, event_l, event_relative_R, event_relative_T, event_sfm_f, event_sfm_tracked_points);
        ROS_INFO("Event SFM costs: %fms", t_esfm.toc());
        if(res){
            EventInit = SFMPass;
            ROS_INFO("Event SFM passed!");
        }else{
            ROS_INFO("Event SFM failed!");
            marginalization_flag = MARGIN_OLD;
        }
    }
    // if(!event_sfm.construct(frame_count + 1, event_Q, event_T, event_l,
    //           event_relative_R, event_relative_T,
    //           event_sfm_f, event_sfm_tracked_points))
    // {
    //     ROS_INFO("global event_SFM failed!");
    //     marginalization_flag = MARGIN_OLD;
    //     return false;
    // }
    //判断两个sfm过程是否符合继续的要求
    if(EventInit != SFMPass && VisualInit != SFMPass){
        VisualInit = ToDoinit;
        EventInit = ToDoinit;
        return false;
    }

    //3. solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it, event_frame_it;
    map<int, Vector3d>::iterator it, event_it;
    if(VisualInit == SFMPass){
        frame_it = all_image_frame.begin( );
        bool visualPNPres=true;
        for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
        {
        // provide initial guess
            cv::Mat r, rvec, t, D, tmp_r;
            if((frame_it->first) == Headers[i])
            {
                frame_it->second.is_key_frame = true;
                frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
                frame_it->second.T = T[i];
                // ROS_DEBUG_STREAM("visual PNP: round "<<i<<endl<<"Q[i].toRotationMatrix(): "<<Q[i].toRotationMatrix()<<endl<<"RIC[0].transpose():"<<RIC[0].transpose()<<endl<<"R:"<<frame_it->second.R<<endl);
                i++;
                continue;
            }
            if((frame_it->first) > Headers[i])
            {
                i++;
            }
            Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
            Vector3d P_inital = - R_inital * T[i];
            cv::eigen2cv(R_inital, tmp_r);
            cv::Rodrigues(tmp_r, rvec);
            cv::eigen2cv(P_inital, t);

            frame_it->second.is_key_frame = false;
            vector<cv::Point3f> pts_3_vector;
            vector<cv::Point2f> pts_2_vector;
            for (auto &id_pts : frame_it->second.points)
            {
                int feature_id = id_pts.first;
                for (auto &i_p : id_pts.second)
                {
                    it = sfm_tracked_points.find(feature_id);
                    if(it != sfm_tracked_points.end())
                    {
                        Vector3d world_pts = it->second;
                        cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                        pts_3_vector.push_back(pts_3);
                        Vector2d img_pts = i_p.second.head<2>();
                        cv::Point2f pts_2(img_pts(0), img_pts(1));
                        pts_2_vector.push_back(pts_2);
                    }
                }
            }
            cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
            if(pts_3_vector.size() < 6)
            {
             cout << "pts_3_vector size " << pts_3_vector.size() << endl;
                ROS_INFO("Visual : Not enough points for solve pnp !");
                visualPNPres = false;
                break;
            // return false;
            }
            if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
                    {
                ROS_INFO("Visual : solve pnp fail!");
                visualPNPres = false;
                // return false;
                break ;
            }
            cv::Rodrigues(rvec, r);
                MatrixXd R_pnp,tmp_R_pnp;
            cv::cv2eigen(r, tmp_R_pnp);
            R_pnp = tmp_R_pnp.transpose();
            MatrixXd T_pnp;
            cv::cv2eigen(t, T_pnp);
            T_pnp = R_pnp * (-T_pnp);
            frame_it->second.R = R_pnp * RIC[0].transpose();
            frame_it->second.T = T_pnp;
            // ROS_DEBUG_STREAM("visual PNP last: round "<<i<<endl<<"R_pnp: "<<R_pnp<<endl<<"RIC[0].transpose():"<<RIC[0].transpose()<<endl<<"R:"<<frame_it->second.R<<endl);
        }
        if(visualPNPres){
            ROS_INFO("Visual PNP passed");
            VisualInit = PNPPass ;
        }else{
            ROS_INFO("Visual PNP failed");
            VisualInit = ToDoinit;
        }
    }
    //solve pnp for all event frame
    if(EventInit == SFMPass){
        bool eventPNPres=true;
        event_frame_it = all_event_frame.begin( );
        for (int i = 0; event_frame_it != all_event_frame.end( ); event_frame_it++)
        {
        // provide initial guess
            cv::Mat r, rvec, t, D, tmp_r;
            if((event_frame_it->first) == Headers[i])
            {
                event_frame_it->second.is_key_frame = true;
                event_frame_it->second.R = event_Q[i].toRotationMatrix() * RIE[0].transpose();
                event_frame_it->second.T = event_T[i];

                // ROS_DEBUG_STREAM("event PNP: round "<<i<<endl
                // <<"event_Q[i].toRotationMatrix(): "<<event_Q[i].toRotationMatrix()<<endl
                // <<"RIE[0].transpose():"<<RIE[0].transpose()<<endl
                // <<"R:"<<event_frame_it->second.R<<endl);

                i++;
                continue;
            }
            if((event_frame_it->first) > Headers[i])
            {
                i++;
            }
            Matrix3d R_inital = (event_Q[i].inverse()).toRotationMatrix();
            Vector3d P_inital = - R_inital * event_T[i];
            cv::eigen2cv(R_inital, tmp_r);
            cv::Rodrigues(tmp_r, rvec);
            cv::eigen2cv(P_inital, t);

            event_frame_it->second.is_key_frame = false;
            vector<cv::Point3f> pts_3_vector;
            vector<cv::Point2f> pts_2_vector;
            for (auto &id_pts : event_frame_it->second.points)
            {
                int feature_id = id_pts.first;
                for (auto &i_p : id_pts.second)
                {
                    event_it = event_sfm_tracked_points.find(feature_id);
                    if(event_it != event_sfm_tracked_points.end())
                    {
                        Vector3d world_pts = event_it->second;
                        cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                        pts_3_vector.push_back(pts_3);
                        Vector2d img_pts = i_p.second.head<2>();
                        cv::Point2f pts_2(img_pts(0), img_pts(1));
                        pts_2_vector.push_back(pts_2);
                    }
                }
            }
            cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
            if(pts_3_vector.size() < 6)
            {
                cout << "pts_3_vector size " << pts_3_vector.size() << endl;
                ROS_INFO("Event : Not enough points for solve pnp !");
                eventPNPres = false;
                break;
                // return false;
            }
            if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
            {
                ROS_INFO("Event : solve pnp fail!");
                eventPNPres = false;
                break;
                // return false;
            }
            cv::Rodrigues(rvec, r);
            MatrixXd R_pnp,tmp_R_pnp;
            cv::cv2eigen(r, tmp_R_pnp);
            R_pnp = tmp_R_pnp.transpose();
            MatrixXd T_pnp;
            cv::cv2eigen(t, T_pnp);
            T_pnp = R_pnp * (-T_pnp);
            event_frame_it->second.R = R_pnp * RIE[0].transpose();
            event_frame_it->second.T = T_pnp;
            // ROS_DEBUG_STREAM("event PNP last: round "<<i<<endl
            // <<"R_pnp: "<<R_pnp<<endl
            // <<"RIE[0].transpose():"<<RIE[0].transpose()<<endl
            // <<"R:"<<event_frame_it->second.R<<endl);
        }
        if(eventPNPres){
            ROS_INFO("Event PNP passed");
            EventInit = PNPPass;
        }else{
            ROS_INFO("Event PNP failed");
            EventInit = ToDoinit;
        }
    }

    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        VisualInit = ToDoinit;
        EventInit = ToDoinit;
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd temp_x_visual, temp_x_event, x;
    Vector3d temp_g_visual, temp_g_event;
    //solve scale
    //2.20这个里面暂时也需要区分，经历了PNP的才能进去solverGYROBIAS
    solveGyroscopeBias(all_image_frame, Bgs, GyroBiasInit);
    bool result_visual=false,result_event=false;
    if(VisualInit == PNPPass){
        result_visual = VisualIMUAlignment(all_image_frame, Bgs, temp_g_visual, temp_x_visual, GyroBiasInit);
    }
    if(EventInit == PNPPass){
        result_event = VisualIMUAlignment(all_event_frame, Bgs, temp_g_event, temp_x_event, GyroBiasInit);
    }
    // bool result_visual = VisualIMUAlignment(all_image_frame, Bgs, g, x, GyroBiasInit);

    //2023.2.15 visual和event分别求解重力方向、尺度信息，若两种方法都可用，则优先相信visual,否则只用成功解算出重力尺度信息的传感器进行初始化。
    if(!result_visual && !result_event)
    {
        ROS_INFO("IMUAlignment : solve g failed!");
        VisualInit = ToDoinit;
        EventInit = ToDoinit;
        return false;
    }else if (result_visual && result_event){
        ROS_INFO("IMUAlignment : visual && event both succeed , using visual result");
        g = temp_g_visual;
        x = temp_x_visual;
    }else if (!result_visual && result_event){
        ROS_INFO("IMUAlignment : only event succeed ,visual has failed");
        g = temp_g_event;
        x = temp_x_event;
        VisualInit = ToDoinit;
    }else if (result_visual && !result_event){
        ROS_INFO("IMUAlignment : only visual succeed ,event has failed");
        g = temp_g_visual;
        x = temp_x_visual;
        EventInit = ToDoinit;
    }else{
        //
    }
    if(VisualInit == PNPPass && result_visual){
        ROS_INFO("Solving alingment by visual");
        for (int i = 0; i <= frame_count; i++)
        {
            Matrix3d Ri = all_image_frame[Headers[i]].R;
            Vector3d Pi = all_image_frame[Headers[i]].T;
            Ps[i] = Pi;
            Rs[i] = Ri;
            all_image_frame[Headers[i]].is_key_frame = true;
        }
        double s = (x.tail<1>())(0);
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
        }
        for (int i = frame_count; i >= 0; i--)
            Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
        int kv = -1;
        map<double, ImageFrame>::iterator frame_i;
        for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
        {
            if(frame_i->second.is_key_frame)
            {
                kv++;
                Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
            }
        }

        Matrix3d R0 = Utility::g2R(g);
        double yaw = Utility::R2ypr(R0 * Rs[0]).x();
        R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
        Matrix3d rot_diff = R0;
        for (int i = 0; i <= frame_count; i++)
        {
            Ps[i] = rot_diff * Ps[i];
            Rs[i] = rot_diff * Rs[i];
            Vs[i] = rot_diff * Vs[i];
        }
        ROS_WARN_STREAM("g0     " << g.transpose());
        ROS_WARN_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 
        f_manager.clearDepth();
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        VisualInit = AlignPass;
        if(EventInit == PNPPass){
            ROS_INFO("visual aligned ,now solving event");
            for (int i = 0; i <= frame_count; i++){
                all_event_frame[Headers[i]].is_key_frame = true;
            }
            event_manager.clearDepth();
            event_manager.triangulate(frame_count, Ps, Rs, tie, rie);
            EventInit = AlignPass;
        }
        return true;

    }else if (EventInit == PNPPass && result_event){
        ROS_INFO("Solving alingment by event");
        for (int i = 0; i <= frame_count; i++)
        {
            Matrix3d Ri = all_event_frame[Headers[i]].R;
            Vector3d Pi = all_event_frame[Headers[i]].T;
            Ps[i] = Pi;
            Rs[i] = Ri;
            all_event_frame[Headers[i]].is_key_frame = true;
        }
        double s = (x.tail<1>())(0);
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
        }
        for (int i = frame_count; i >= 0; i--)
            Ps[i] = s * Ps[i] - Rs[i] * TIE[0] - (s * Ps[0] - Rs[0] * TIE[0]);
        int kv = -1;
        map<double, ImageFrame>::iterator frame_i;
        for (frame_i = all_event_frame.begin(); frame_i != all_event_frame.end(); frame_i++)
        {
            if(frame_i->second.is_key_frame)
            {
                kv++;
                Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
            }
        }

        Matrix3d R0 = Utility::g2R(g);
        double yaw = Utility::R2ypr(R0 * Rs[0]).x();
        R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
        Matrix3d rot_diff = R0;
        for (int i = 0; i <= frame_count; i++)
        {
            Ps[i] = rot_diff * Ps[i];
            Rs[i] = rot_diff * Rs[i];
            Vs[i] = rot_diff * Vs[i];
        }
        ROS_WARN_STREAM("g0     " << g.transpose());
        ROS_WARN_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 
        event_manager.clearDepth();
        event_manager.triangulate(frame_count, Ps, Rs, tie, rie);
        EventInit = AlignPass;
        return true;
    }else{
        ROS_WARN("Error at Initial LinearAlignment");
        return false;
    }


    // change state
    // for (int i = 0; i <= frame_count; i++)
    // {
    //     Matrix3d Ri = all_image_frame[Headers[i]].R;
    //     Vector3d Pi = all_image_frame[Headers[i]].T;
    //     Ps[i] = Pi;
    //     Rs[i] = Ri;
    //     all_image_frame[Headers[i]].is_key_frame = true;
    // }

    // double s = (x.tail<1>())(0);
    // for (int i = 0; i <= WINDOW_SIZE; i++)
    // {
    //     pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    // }
    // for (int i = frame_count; i >= 0; i--)
    //     Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    // int kv = -1;
    // map<double, ImageFrame>::iterator frame_i;
    // for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    // {
    //     if(frame_i->second.is_key_frame)
    //     {
    //         kv++;
    //         Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
    //     }
    // }

    // Matrix3d R0 = Utility::g2R(g);
    // double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    // R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // g = R0 * g;
    // //Matrix3d rot_diff = R0 * Rs[0].transpose();
    // Matrix3d rot_diff = R0;
    // for (int i = 0; i <= frame_count; i++)
    // {
    //     Ps[i] = rot_diff * Ps[i];
    //     Rs[i] = rot_diff * Rs[i];
    //     Vs[i] = rot_diff * Vs[i];
    // }
    // ROS_DEBUG_STREAM("g0     " << g.transpose());
    // ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    // f_manager.clearDepth();
    // f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    // event_manager.clearDepth();
    // event_manager.triangulate(frame_count, Ps, Rs, tie, rie);

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_INFO_THROTTLE(3,"average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    ROS_INFO_THROTTLE(3,"relativePose : Frame average_parallax too small");
    return false;
}
bool Estimator::event_relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = event_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 10)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_INFO_THROTTLE(1,"Event_relativePose: average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }else{
            // cout<<"Event corres size="<<corres.size()<<endl;
        }
    }
    ROS_INFO_THROTTLE(3,"Event_relativePose : Event Frame average_parallax too small");
    return false;
}

void Estimator::vector2double()
{
    if(VisualInit == AlignPass){
    //转换以便输入ceres
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            para_Pose[i][0] = Ps[i].x();
            para_Pose[i][1] = Ps[i].y();
            para_Pose[i][2] = Ps[i].z();
            Quaterniond q{Rs[i]};
            para_Pose[i][3] = q.x();
            para_Pose[i][4] = q.y();
            para_Pose[i][5] = q.z();
            para_Pose[i][6] = q.w();

            if(USE_IMU)
            {
                para_SpeedBias[i][0] = Vs[i].x();
                para_SpeedBias[i][1] = Vs[i].y();
                para_SpeedBias[i][2] = Vs[i].z();

                para_SpeedBias[i][3] = Bas[i].x();
                para_SpeedBias[i][4] = Bas[i].y();
                para_SpeedBias[i][5] = Bas[i].z();

                para_SpeedBias[i][6] = Bgs[i].x();
                para_SpeedBias[i][7] = Bgs[i].y();
                para_SpeedBias[i][8] = Bgs[i].z();
            }
        }

        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            para_Ex_Pose[i][0] = tic[i].x();
            para_Ex_Pose[i][1] = tic[i].y();
            para_Ex_Pose[i][2] = tic[i].z();
            Quaterniond q{ric[i]};
            para_Ex_Pose[i][3] = q.x();
            para_Ex_Pose[i][4] = q.y();
            para_Ex_Pose[i][5] = q.z();
            para_Ex_Pose[i][6] = q.w();
        }


        VectorXd dep = f_manager.getDepthVector();
        for (int i = 0; i < f_manager.getFeatureCount(); i++)
            para_Feature[i][0] = dep(i);

        para_Td[0][0] = td;
    }else{
        return;
    }
}

void Estimator::double2vector()
{
    if(VisualInit == AlignPass){
        Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
        Vector3d origin_P0 = Ps[0];

        if (failure_occur)
        {
            origin_R0 = Utility::R2ypr(last_R0);
            origin_P0 = last_P0;
            failure_occur = 0;
        }

        if(USE_IMU)
        {
            Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                            para_Pose[0][3],
                                                            para_Pose[0][4],
                                                            para_Pose[0][5]).toRotationMatrix());
            double y_diff = origin_R0.x() - origin_R00.x();
            //TODO
            Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
            if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
            {
                ROS_DEBUG("euler singular point!");
                rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                            para_Pose[0][3],
                                            para_Pose[0][4],
                                            para_Pose[0][5]).toRotationMatrix().transpose();
            }

            for (int i = 0; i <= WINDOW_SIZE; i++)
            {

                Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
                
                Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                        para_Pose[i][1] - para_Pose[0][1],
                                        para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                    Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                                para_SpeedBias[i][1],
                                                para_SpeedBias[i][2]);

                    Bas[i] = Vector3d(para_SpeedBias[i][3],
                                    para_SpeedBias[i][4],
                                    para_SpeedBias[i][5]);

                    Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                    para_SpeedBias[i][7],
                                    para_SpeedBias[i][8]);
                
            }
        }
        else
        {
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
                
                Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
            }
        }

        if(USE_IMU)
        {
            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                tic[i] = Vector3d(para_Ex_Pose[i][0],
                                para_Ex_Pose[i][1],
                                para_Ex_Pose[i][2]);
                ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                    para_Ex_Pose[i][3],
                                    para_Ex_Pose[i][4],
                                    para_Ex_Pose[i][5]).normalized().toRotationMatrix();
            }
        }

        VectorXd dep = f_manager.getDepthVector();
        for (int i = 0; i < f_manager.getFeatureCount(); i++)
            dep(i) = para_Feature[i][0];
        f_manager.setDepth(dep);

        if(USE_IMU)
            td = para_Td[0][0];
    }else{
        return;
    }

}

void Estimator::event_vector2double(){
//若两种传感器均初始化成功，则event不影响pose和bias；若event初始化成功但visual初始化失败，则采用event的数据进行位姿转换
    if(EventInit == AlignPass && VisualInit == AlignPass){

        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            event_para_Ex_Pose[i][0] = tie[i].x();
            event_para_Ex_Pose[i][1] = tie[i].y();
            event_para_Ex_Pose[i][2] = tie[i].z();
            Quaterniond q{rie[i]};
            event_para_Ex_Pose[i][3] = q.x();
            event_para_Ex_Pose[i][4] = q.y();
            event_para_Ex_Pose[i][5] = q.z();
            event_para_Ex_Pose[i][6] = q.w();
        }
        VectorXd dep = event_manager.getDepthVector();
        for (int i = 0; i < event_manager.getFeatureCount(); i++)
            event_para_Feature[i][0] = dep(i);
    }else if (EventInit == AlignPass && VisualInit != AlignPass){
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            para_Pose[i][0] = Ps[i].x();
            para_Pose[i][1] = Ps[i].y();
            para_Pose[i][2] = Ps[i].z();
            Quaterniond q{Rs[i]};
            para_Pose[i][3] = q.x();
            para_Pose[i][4] = q.y();
            para_Pose[i][5] = q.z();
            para_Pose[i][6] = q.w();

            if(USE_IMU)
            {
                para_SpeedBias[i][0] = Vs[i].x();
                para_SpeedBias[i][1] = Vs[i].y();
                para_SpeedBias[i][2] = Vs[i].z();

                para_SpeedBias[i][3] = Bas[i].x();
                para_SpeedBias[i][4] = Bas[i].y();
                para_SpeedBias[i][5] = Bas[i].z();

                para_SpeedBias[i][6] = Bgs[i].x();
                para_SpeedBias[i][7] = Bgs[i].y();
                para_SpeedBias[i][8] = Bgs[i].z();
            }
        }

        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            event_para_Ex_Pose[i][0] = tie[i].x();
            event_para_Ex_Pose[i][1] = tie[i].y();
            event_para_Ex_Pose[i][2] = tie[i].z();
            Quaterniond q{rie[i]};
            event_para_Ex_Pose[i][3] = q.x();
            event_para_Ex_Pose[i][4] = q.y();
            event_para_Ex_Pose[i][5] = q.z();
            event_para_Ex_Pose[i][6] = q.w();
        }
        VectorXd dep = event_manager.getDepthVector();
        for (int i = 0; i < event_manager.getFeatureCount(); i++)
            event_para_Feature[i][0] = dep(i);

        para_Td[0][0] = td;

    }else{
        return;
    }
}

void Estimator::event_double2vector()
{
    //若两种传感器均初始化成功，则event不影响pose和bias；若event初始化成功但visual初始化失败，则采用event的数据进行位姿转换
    if(EventInit == AlignPass){
        //如果visual没有通过初始化，则event负责还原PVQ
        if(VisualInit != AlignPass){
            Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
            Vector3d origin_P0 = Ps[0];

            // if (failure_occur)
            // {
            //     origin_R0 = Utility::R2ypr(last_R0);
            //     origin_P0 = last_P0;
            //     failure_occur = 0;
            // }

            if(USE_IMU)
            {
                Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                                para_Pose[0][3],
                                                                para_Pose[0][4],
                                                                para_Pose[0][5]).toRotationMatrix());
                double y_diff = origin_R0.x() - origin_R00.x();
                //TODO
                Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
                if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
                {
                    ROS_DEBUG("euler singular point!");
                    rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                                para_Pose[0][3],
                                                para_Pose[0][4],
                                                para_Pose[0][5]).toRotationMatrix().transpose();
                }

                for (int i = 0; i <= WINDOW_SIZE; i++)
                {

                    Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
                    
                    Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                            para_Pose[i][1] - para_Pose[0][1],
                                            para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                                    para_SpeedBias[i][1],
                                                    para_SpeedBias[i][2]);

                        Bas[i] = Vector3d(para_SpeedBias[i][3],
                                        para_SpeedBias[i][4],
                                        para_SpeedBias[i][5]);

                        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                        para_SpeedBias[i][7],
                                        para_SpeedBias[i][8]);
                    
                }
            }
            else
            {
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
                    
                    Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
                }
            }
        }
        //这部分是只要event通过初始化就一定做的
        if(USE_IMU)
        {
            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                tie[i] = Vector3d(event_para_Ex_Pose[i][0],
                                event_para_Ex_Pose[i][1],
                                event_para_Ex_Pose[i][2]);
                rie[i] = Quaterniond(event_para_Ex_Pose[i][6],
                                    event_para_Ex_Pose[i][3],
                                    event_para_Ex_Pose[i][4],
                                    event_para_Ex_Pose[i][5]).normalized().toRotationMatrix();
            }
        }
        VectorXd dep = event_manager.getDepthVector();
        for (int i = 0; i < event_manager.getFeatureCount(); i++){
            dep(i) = event_para_Feature[i][0];
        }
        event_manager.setDepth(dep);

    }else{
        return;
    }
}

bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

void Estimator::optimization()
{
    //根据初始化结果来决定该把哪个传感器加入残差
    TicToc t_whole, t_prepare;
    vector2double();
    event_vector2double();
    int initialflag=InitFlagSwitcher();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if(!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        switch(initialflag){
	        case 1:         
            problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
            problem.AddParameterBlock(event_para_Ex_Pose[i], SIZE_POSE, local_parameterization);
            break;
            case 2:         
            problem.AddParameterBlock(event_para_Ex_Pose[i], SIZE_POSE, local_parameterization);
            break;
            case 3:
            problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
            break;
            default :
            ROS_WARN_STREAM("Parameter block add failed");
            break;
	    }

        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            // ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            // ROS_INFO("fix extinsic param");
            switch(initialflag){
                case 1:         
                    problem.SetParameterBlockConstant(para_Ex_Pose[i]);
                    problem.SetParameterBlockConstant(event_para_Ex_Pose[i]);
                    break;
                case 2:         
                problem.SetParameterBlockConstant(event_para_Ex_Pose[i]);
                    break;
                case 3:
                problem.SetParameterBlockConstant(para_Ex_Pose[i]);
                    break;
                default :
                ROS_WARN_STREAM("Parameter block set failed");
                    break;
	        }
            // problem.SetParameterBlockConstant(para_Ex_Pose[i]);
            // problem.SetParameterBlockConstant(event_para_Ex_Pose[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }
//添加重投影误差block
//哪个初始化成功了就添加哪个进入block
    if(initialflag==1||initialflag==3){
        int f_m_cnt = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (it_per_id.used_num < 4)
                continue;
    
            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i != imu_j)
                {
                    Vector3d pts_j = it_per_frame.point;
                    ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                }

                // if(STEREO && it_per_frame.is_stereo)
                // {                
                //     Vector3d pts_j_right = it_per_frame.pointRight;
                //     if(imu_i != imu_j)
                //     {
                //         ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                //                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                //         problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                //     }
                //     else
                //     {
                //         ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                //                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                //         problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                //     }
                
                // }
                f_m_cnt++;
            }
        }

        ROS_DEBUG("optimization : add visual RPE , visual measurement count: %d", f_m_cnt);
        //printf("prepare for ceres: %f \n", t_prepare.toc());
    }
//添加event重投影
    if(initialflag==1||initialflag==2){
        int event_f_m_cnt = 0;
        int event_feature_index = -1;
        for (auto &it_per_id : event_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (it_per_id.used_num < 4)
                continue;
    
            ++event_feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i != imu_j)
                {
                    Vector3d pts_j = it_per_frame.point;
                    ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], event_para_Ex_Pose[0], event_para_Feature[event_feature_index], para_Td[0]);
                }
                event_f_m_cnt++;
            }
        }

        ROS_DEBUG("optimization : add visual RPE , event measurement count: %d", event_f_m_cnt);
    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solve ceres problem costs: %f \n", t_solver.toc());

    double2vector();
    event_double2vector();
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)
        return;
    
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        ROS_DEBUG("optimization : marginalization_flag == MARGIN_OLD");
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();
        event_vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        if(initialflag==1||initialflag==3){
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    // if(STEREO && it_per_frame.is_stereo)
                    // {
                    //     Vector3d pts_j_right = it_per_frame.pointRight;
                    //     if(imu_i != imu_j)
                    //     {
                    //         ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                    //                                                       it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    //         ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                    //                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                    //                                                                        vector<int>{0, 4});
                    //         marginalization_info->addResidualBlockInfo(residual_block_info);
                    //     }
                    //     else
                    //     {
                    //         ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                    //                                                       it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    //         ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                    //                                                                        vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                    //                                                                        vector<int>{2});
                    //         marginalization_info->addResidualBlockInfo(residual_block_info);
                    //     }
                    // }
                }
            }
            ROS_DEBUG("optimization : marginalization__ add visual RPE , feature_index count = %d",feature_index);
        }
//event marg
        if(initialflag==1||initialflag==2){
            int event_feature_index = -1;
            for (auto &it_per_id : event_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++event_feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], event_para_Ex_Pose[0], event_para_Feature[event_feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
            ROS_DEBUG("optimization : marginalization__ add event RPE , event_feature_index count = %d",event_feature_index);
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++){
            if(initialflag==1||initialflag==3){
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            }
                        //add event ex pose
            if(initialflag==1||initialflag==2){
                addr_shift[reinterpret_cast<long>(event_para_Ex_Pose[i])] = event_para_Ex_Pose[i];
            }
        }

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            event_vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++){
                if(initialflag==1||initialflag==3){
                    addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
                }
                if(initialflag==1||initialflag==2){
                    addr_shift[reinterpret_cast<long>(event_para_Ex_Pose[i])] = event_para_Ex_Pose[i];
                }
            }

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        ROS_DEBUG("slidewindow: m_old");
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0,event_it_0;
                it_0 = all_image_frame.find(t_0);
                event_it_0 = all_event_frame.find(t_0);
                delete it_0->second.pre_integration;
                int image_pre=all_image_frame.size(),event_pre=all_event_frame.size();
                all_image_frame.erase(all_image_frame.begin(), it_0);
                // delete event_it_0->second.pre_integration;
                all_event_frame.erase(all_event_frame.begin(), event_it_0);
                int image_after=all_image_frame.size(),event_after=all_event_frame.size();
                ROS_DEBUG("slidewindow: image_pre=%d,event_pre=%d,image_after=%d,event_after=%d",image_pre,event_pre,image_after,event_after);
            }
            slideWindowOld();
        }
    }
    else
    {
        ROS_DEBUG("slidewindow: m_2new");
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    ROS_DEBUG("sliding Window New....");
    sum_of_front++;
    f_manager.removeFront(frame_count);
    event_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    ROS_DEBUG("sliding Window Old....");
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
        //event
        Matrix3d event_R0, event_R1;
        Vector3d event_P0, event_P1;
        event_R0 = back_R0 * rie[0];
        event_R1 = Rs[0] * rie[0];
        event_P0 = back_P0 + back_R0 * tie[0];
        event_P1 = Ps[0] + Rs[0] * tie[0];
        event_manager.removeBackShiftDepth(event_R0, event_P0, event_R1, event_P1);
    }
    else
    {
        f_manager.removeBack();
        event_manager.removeBack();
    }
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame()
{
    printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex,set<int> &event_removeIndex)
{
    //return;
    int initialflag=InitFlagSwitcher();
    if(initialflag==1||initialflag==3){
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            double err = 0;
            int errCnt = 0;
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (it_per_id.used_num < 4)
                continue;
            feature_index ++;
            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;
            double depth = it_per_id.estimated_depth;
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i != imu_j)
                {
                    Vector3d pts_j = it_per_frame.point;             
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                        depth, pts_i, pts_j);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                // // need to rewrite projecton factor.........
                // if(STEREO && it_per_frame.is_stereo)
                // {
                    
                //     Vector3d pts_j_right = it_per_frame.pointRight;
                //     if(imu_i != imu_j)
                //     {            
                //         double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                //                                             Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                //                                             depth, pts_i, pts_j_right);
                //         err += tmp_error;
                //         errCnt++;
                //         //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                //     }
                //     else
                //     {
                //         double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                //                                             Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                //                                             depth, pts_i, pts_j_right);
                //         err += tmp_error;
                //         errCnt++;
                //         //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                //     }       
                // }
            }
            double ave_err = err / errCnt;
            if(ave_err * FOCAL_LENGTH > 3)
                removeIndex.insert(it_per_id.feature_id);
        }
    }
    if(initialflag==1||initialflag==2){
        int event_feature_index = -1;
        for (auto &it_per_id : event_manager.feature)
        {
            double err = 0;
            int errCnt = 0;
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (it_per_id.used_num < 4)
                continue;
            event_feature_index ++;
            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;
            double depth = it_per_id.estimated_depth;
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i != imu_j)
                {
                    Vector3d pts_j = it_per_frame.point;             
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], rie[0], tie[0], 
                                                        Rs[imu_j], Ps[imu_j], rie[0], tie[0],
                                                        depth, pts_i, pts_j);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                
            }
            double ave_err = err / errCnt;
            if(ave_err * FOCAL_LENGTH > 3)
                event_removeIndex.insert(it_per_id.feature_id);

        }
    }
    ROS_DEBUG("outliersRejection : remove visual %d , remove event %d",removeIndex.size(),event_removeIndex.size());
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
//1:双传感器初始化同时成功；2.只有event；3.只有visual；4.均失败 
int Estimator::InitFlagSwitcher(){
    if(VisualInit == AlignPass && EventInit == AlignPass) return 1;
    if(VisualInit != AlignPass && EventInit == AlignPass) return 2;
    if(VisualInit == AlignPass && EventInit != AlignPass) return 3;
    if(VisualInit != AlignPass && EventInit != AlignPass) return 4;
    return -1;
}

void Estimator::RetryInitAlign(){
// return;
//用于运行当中对某一相机进行重新初始化和对齐
    int initialflag = InitFlagSwitcher();
    if(initialflag ==1) return;
    if(initialflag ==4 ){
        ROS_ERROR("RetryInitAlign is solving an unexpected situation, systemflag changed to initial");
        solver_flag = INITIAL;
    }
    if(initialflag ==2){
        //重新对visual进行初始化
        for (int i = 0; i <= frame_count; i++){
            all_image_frame[Headers[i]].is_key_frame = true;
        }
        f_manager.clearDepth();
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        VisualInit = AlignPass;

    }
    if(initialflag ==3){
        //重新对event进行初始化
        // if(EventInit == PNPPass){
            for (int i = 0; i <= frame_count; i++){
                all_event_frame[Headers[i]].is_key_frame = true;
            }
            event_manager.clearDepth();
            event_manager.triangulate(frame_count, Ps, Rs, tie, rie);
            EventInit = AlignPass;
        // }
    }
    return;
}
