/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;
std::vector<Eigen::Matrix3d> RIE;
std::vector<Eigen::Vector3d> TIE;
std::vector<double> GyroBiasInit;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
int EQUALIZE;

int USE_V2DT;
double ACC_R;
double GYRO_R;
double AMAGI;
double W_SIZE;
double W_SIZE2;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string CONTRIBUTE_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string PATH_OUTPUT_NAME;
std::string CONTRIBUTE_OUTPUT_NAME;
std::string IMU_TOPIC;
int ROW, COL, event_ROW, event_COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
int EVENT_DT;
int EVENT_SUM;
int EVENT_MIN;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC, EVENT_TOPIC, DEBUG_IMAGE_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
std::vector<std::string> EventCAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;


template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    fsSettings["event_topic"] >> EVENT_TOPIC;
    fsSettings["debug_image_topic"] >> DEBUG_IMAGE_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"];
    EVENT_DT = fsSettings["event_dt"];
    EVENT_SUM = fsSettings["event_sum"];
    EVENT_MIN = fsSettings["event_min"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    fsSettings["path_output_name"] >> PATH_OUTPUT_NAME;
    fsSettings["contribute_output_name"] >> CONTRIBUTE_OUTPUT_NAME;
    VINS_RESULT_PATH = OUTPUT_FOLDER + PATH_OUTPUT_NAME;
    CONTRIBUTE_RESULT_PATH = OUTPUT_FOLDER + CONTRIBUTE_OUTPUT_NAME;
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::cout << "contribute  path " << CONTRIBUTE_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    std::ofstream contr_fout(CONTRIBUTE_RESULT_PATH, std::ios::out);
    fout.close();
    contr_fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));

        cv::Mat event_cv_T;
        fsSettings["body_T_Event"] >> event_cv_T;
        Eigen::Matrix4d event_T;
        cv::cv2eigen(event_cv_T, event_T);
        RIE.push_back(T.block<3, 3>(0, 0));
        TIE.push_back(T.block<3, 1>(0, 3));
        fsSettings["gyrobiasinit"] >> GyroBiasInit;
        cout<<"GyroBiasInit: ";
        for(int iii=0;iii<GyroBiasInit.size();iii++){
            cout<<GyroBiasInit[iii]<<" , ";
        }
        cout<<endl;
    } 
    
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }


    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    
    std::string cam0Calib, camECalib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    fsSettings["camE_calib"] >> camECalib;
    std::string camEPath = configPath + "/" + camECalib;
    EventCAM_NAMES.push_back(camEPath);

    if(NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib; 
        //printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);
        
        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;
    
    EQUALIZE = fsSettings["equalize"];
    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];

    USE_V2DT = fsSettings["usev2dt"];
    ACC_R = fsSettings["acc_ratio"];
    GYRO_R = fsSettings["gyro_ratio"];
    AMAGI = fsSettings["const_amagi"];
    W_SIZE  = fsSettings["window_size"];
    W_SIZE2  = fsSettings["window_size2"];

    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    event_ROW = fsSettings["event_height"];
    event_COL = fsSettings["event_width"];
    ROS_INFO("ROW: %d COL: %d ,event_ROW: %d event_COL: %d", ROW, COL, event_ROW, event_COL);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings.release();
}
