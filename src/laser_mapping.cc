#include <tf/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>
#include <execution>
#include <fstream>
#include "unordered_map"

#include "laser_mapping.h"
#include "use-ikfom.hpp"
#include "utils.h"
#include "Vertical_optimization/vertical_optimization.h"

namespace faster_lio {
/*这里将点云格式改变了，影响右面的吗？*/
template<typename T>
sensor_msgs::PointCloud2 cloud2msg(pcl::PointCloud<T> cloud, uint64_t stamp_nanoseconds, std::string frame_id = "map") {
    sensor_msgs::PointCloud2 cloud_ROS;
    pcl::toROSMsg(cloud, cloud_ROS);
    cloud_ROS.header.stamp.fromNSec(stamp_nanoseconds);
    cloud_ROS.header.frame_id = frame_id;
    return cloud_ROS;
}

bool LaserMapping::InitROS(ros::NodeHandle &nh) {
    LoadParams(nh);
    SubAndPubToROS(nh); //话题设置

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_); //iVOX的实例化

    // esekf init
    std::vector<double> epsi(23, 0.001);
    // 初始化，传入几个参数
    // 1. get_f: 用于根据IMU数据向前推算
    // 2. df_dx: 误差状态模型，（连续时间下）
    // 3. df_dw: 误差状态模型，误差状态对过程噪声求导
    // 4. lambda: 函数类型 std::function<void(state &, dyn_share_datastruct<scalar_type> &)>
    //
    // kf_.init_dyn_share(
    //     get_f, df_dx, df_dw,
    //     [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
    //     options::NUM_MAX_ITERATIONS, epsi.data()); 
    // return true;
    if (is_extract_large_planes_) { //根据是否需要提取大平面，使用不同的观测模型进行初始化
        kf_.init_dyn_share(
            get_f, df_dx, df_dw,
            [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModelOurs(s, ekfom_data); },
            options::NUM_MAX_ITERATIONS, epsi.data());
    } else {
        kf_.init_dyn_share(
            get_f, df_dx, df_dw,
            [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
            options::NUM_MAX_ITERATIONS, epsi.data());
    }

    plane_tracker_.SetIncrementalFitting(is_incremental_fitting_); //设置是否使用增量拟合方法
    plane_tracker_.SetPlaneLeastInliers(plane_least_inliers_); //设置平面拟合中所需的最小内点数量
    return true;
}

bool LaserMapping::InitWithoutROS(const std::string &config_yaml) {
    LOG(INFO) << "init laser mapping from " << config_yaml;
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    std::vector<double> epsi(23, 0.001);
    // kf_.init_dyn_share(
    //     get_f, df_dx, df_dw,
    //     [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
    //     options::NUM_MAX_ITERATIONS, epsi.data());

    // if (std::is_same<IVoxType, IVox<3, IVoxNodeType::PHC, pcl::PointXYZI>>::value == true) {
    //     LOG(INFO) << "using phc ivox";
    // } else if (std::is_same<IVoxType, IVox<3, IVoxNodeType::DEFAULT, pcl::PointXYZI>>::value == true) {
    //     LOG(INFO) << "using default ivox";
    // }

    // return true;
    if (is_extract_large_planes_) {
        kf_.init_dyn_share(
            get_f, df_dx, df_dw,
            [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModelOurs(s, ekfom_data); },
            options::NUM_MAX_ITERATIONS, epsi.data());
    } else {
        kf_.init_dyn_share(
            get_f, df_dx, df_dw,
            [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
            options::NUM_MAX_ITERATIONS, epsi.data());
    }

    if (std::is_same<IVoxType, IVox<3, IVoxNodeType::PHC, pcl::PointXYZI>>()) {
        LOG(INFO) << "using phc ivox"; //如果使用的是PHC类型的 ivox，记录 "using phc ivox" 自己设置的一个曲线
    } else if (std::is_same<IVoxType, IVox<3, IVoxNodeType::DEFAULT, pcl::PointXYZI>>()) {
        LOG(INFO) << "using default ivox";//如果使用的是默认类型的 ivox，记录 "using default ivox"
    }

    plane_tracker_.SetIncrementalFitting(is_incremental_fitting_);
    plane_tracker_.SetPlaneLeastInliers(plane_least_inliers_);
    return true;
}

/**
 * @brief 读取配置文件，初始化IMUProcessor
 * @param[in] nh
 * @return true
 */
bool LaserMapping::LoadParams(ros::NodeHandle &nh) {
    // get params from param server
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double filter_size_surf_min;
    common::V3D lidar_T_wrt_IMU;
    common::M3D lidar_R_wrt_IMU;
    // common::V3D lidar_T_wrt_IMU;
    // common::M3D lidar_R_wrt_IMU;

    nh.param<bool>("path_save_en", path_save_en_, true);
    nh.param<bool>("publish/path_publish_en", path_pub_en_, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en_, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en_, false);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en_, true);
    nh.param<bool>("publish/scan_effect_pub_en", scan_effect_pub_en_, false);
    nh.param<bool>("is_extract_large_planes", is_extract_large_planes_, false);
    nh.param<int>("plane_least_inliers", plane_least_inliers_, 3);
    nh.param<float>("point_to_plane_thresh", point_to_plane_thresh_, 0.2f);
    nh.param<bool>("is_incremental_fitting", is_incremental_fitting_, false);


    nh.param<std::string>("publish/tf_imu_frame", tf_imu_frame_, "body");
    nh.param<std::string>("publish/tf_world_frame", tf_world_frame_, "camera_init");

    nh.param<int>("max_iteration", options::NUM_MAX_ITERATIONS, 4);
    nh.param<float>("esti_plane_threshold", options::ESTI_PLANE_THRESHOLD, 0.1);
    nh.param<std::string>("map_file_path", map_file_path_, "");
    nh.param<bool>("common/time_sync_en", time_sync_en_, false);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min_, 0.0);
    nh.param<double>("cube_side_length", cube_len_, 200);
    nh.param<float>("mapping/det_range", det_range_, 300.f);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", preprocess_->Blind(), 0.01);
    nh.param<float>("preprocess/time_scale", preprocess_->TimeScale(), 1e-3);
    nh.param<int>("preprocess/lidar_type", lidar_type, 1);
    nh.param<int>("preprocess/scan_line", preprocess_->NumScans(), 16);
    nh.param<int>("point_filter_num", preprocess_->PointFilterNum(), 2);
    nh.param<bool>("feature_extract_enable", preprocess_->FeatureEnabled(), false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log_, true);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en_, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en_, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval_, -1);
    nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT_, std::vector<double>());
    nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR_, std::vector<double>());

    nh.param<float>("ivox_grid_resolution", ivox_options_.resolution_, 0.2);
    nh.param<int>("ivox_nearby_type", ivox_nearby_type, 18);


    nh.param<float>("plane_grid_resolution", plane_grid_resolution, 0.5);
    nh.param<float>("z_resolution", z_resolution, 0.5);
    nh.param<int>("max_points_size", max_points_size, 200);
    nh.param<int>("update_size_threshold", update_size_threshold, 10);
    nh.param<int>("voxel_plane_pub_rate", voxel_plane_pub_rate, 100);
    nh.param<float>("range_filter", range_filter, 10.0);
    nh.param<float>("z_filter", z_filter, 0.5);
    nh.param<float>("marker_scale", marker_scale, 0.45);
    nh.param<double>("ranging_cov", ranging_cov, 0.02);
    nh.param<double>("angle_cov", angle_cov, 0.05);
    nh.param<float>("plane_init_threshold", plane_init_threshold, 0.2);
    nh.param<int>("ransac_iteration", ransac_iteration, 50);
    nh.param<float>("ransac_thre", ransac_thre, 0.1);

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = "camera_init";

    voxel_scan_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
    lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);

    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    return true;
}

bool LaserMapping::LoadParamsFromYAML(const std::string &yaml_file) {
    // get params from yaml
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double filter_size_surf_min;
    common::V3D lidar_T_wrt_IMU;
    common::M3D lidar_R_wrt_IMU;

    auto yaml = YAML::LoadFile(yaml_file);
    try {
        path_pub_en_ = yaml["publish"]["path_publish_en"].as<bool>();
        scan_pub_en_ = yaml["publish"]["scan_publish_en"].as<bool>();
        dense_pub_en_ = yaml["publish"]["dense_publish_en"].as<bool>();
        scan_body_pub_en_ = yaml["publish"]["scan_bodyframe_pub_en"].as<bool>();
        scan_effect_pub_en_ = yaml["publish"]["scan_effect_pub_en"].as<bool>();
        tf_imu_frame_ = yaml["publish"]["tf_imu_frame"].as<std::string>("body");
        tf_world_frame_ = yaml["publish"]["tf_world_frame"].as<std::string>("camera_init");
        path_save_en_ = yaml["path_save_en"].as<bool>();

        options::NUM_MAX_ITERATIONS = yaml["max_iteration"].as<int>();
        options::ESTI_PLANE_THRESHOLD = yaml["esti_plane_threshold"].as<float>();
        time_sync_en_ = yaml["common"]["time_sync_en"].as<bool>();

        filter_size_surf_min = yaml["filter_size_surf"].as<float>();
        filter_size_map_min_ = yaml["filter_size_map"].as<float>();
        cube_len_ = yaml["cube_side_length"].as<int>();
        det_range_ = yaml["mapping"]["det_range"].as<float>();
        gyr_cov = yaml["mapping"]["gyr_cov"].as<float>();
        acc_cov = yaml["mapping"]["acc_cov"].as<float>();
        b_gyr_cov = yaml["mapping"]["b_gyr_cov"].as<float>();
        b_acc_cov = yaml["mapping"]["b_acc_cov"].as<float>();
        preprocess_->Blind() = yaml["preprocess"]["blind"].as<double>();
        preprocess_->TimeScale() = yaml["preprocess"]["time_scale"].as<double>();
        lidar_type = yaml["preprocess"]["lidar_type"].as<int>();
        preprocess_->NumScans() = yaml["preprocess"]["scan_line"].as<int>();
        preprocess_->PointFilterNum() = yaml["point_filter_num"].as<int>();
        preprocess_->FeatureEnabled() = yaml["feature_extract_enable"].as<bool>();
        extrinsic_est_en_ = yaml["mapping"]["extrinsic_est_en"].as<bool>();
        pcd_save_en_ = yaml["pcd_save"]["pcd_save_en"].as<bool>();
        pcd_save_interval_ = yaml["pcd_save"]["interval"].as<int>();
        extrinT_ = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
        extrinR_ = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();

        ivox_options_.resolution_ = yaml["ivox_grid_resolution"].as<float>();
        ivox_nearby_type = yaml["ivox_nearby_type"].as<int>();
    } catch (...) {
        LOG(ERROR) << "bad conversion";
        return false;
    }

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    voxel_scan_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
    lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);

    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    run_in_offline_ = true;
    return true;
}


void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh) {
    // ROS subscribe initialization
    std::string lidar_topic, imu_topic;
    nh.param<std::string>("common/lid_topic", lidar_topic, "/livox/lidar");
    nh.param<std::string>("common/imu_topic", imu_topic, "/livox/imu");

    if (preprocess_->GetLidarType() == LidarType::AVIA) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            lidar_topic, 200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
    } else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            lidar_topic, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });

    // ROS publisher init
    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = "camera_init";

    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_body_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    pub_laser_cloud_lidar_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_lidar", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);

    //GroudSeg debug
    pub_laser_cloud_gnd_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_seg_gnd", 100000);
    pub_laser_cloud_none_gnd_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_seg_none_gnd", 100000);
    pub_laser_cloud_origin_ = nh.advertise<sensor_msgs::PointCloud2>("/gnd_world", 100000);
    pub_laser_cloud_debug_ = nh.advertise<sensor_msgs::PointCloud2>("/debug_cloud_outliers", 100000);
    pub_laser_cloud_debug_2 = nh.advertise<sensor_msgs::PointCloud2>("/debug_cloud2_points2add", 100000);
    pub_laser_cloud_debug_3 = nh.advertise<sensor_msgs::PointCloud2>("/debug_cloud3_pndown", 100000);
    voxel_map_pub = nh.advertise<visualization_msgs::MarkerArray>("/plane_maps", 10000);
    // voxel_map_pub = nh.advertise<jsk_recognition_msgs::PolygonArray>("/planes", 10000);
}

LaserMapping::LaserMapping() {
    preprocess_.reset(new PointCloudPreprocess());
    p_imu_.reset(new ImuProcess());
    scan_down_body_ = boost::make_shared<PointCloudType>();
    scan_undistort_ = boost::make_shared<PointCloudType>();
    scan_down_world_ = boost::make_shared<PointCloudType>();
    scan_gnd = boost::make_shared<PointCloudType>();
    scan_none_gnd = boost::make_shared<PointCloudType>();

    nearest_points_.clear();
}

// LaserMapping::~LaserMapping() {
//     // 确保智能指针被正确释放
//     scan_down_body_.reset();
//     scan_undistort_.reset();
//     scan_down_world_.reset();
//     scan_gnd.reset();
//     scan_none_gnd.reset();

//     LOG(INFO) << "laser mapping deconstruct";
// }
//TAG::run函数
void LaserMapping::Run() {
    if (!SyncPackages()) {
        return;
    }

    p_imu_->Process(measures_, kf_, scan_undistort_); // 前20帧点云(imu初始化),去畸变从第21帧才开始,此时scan_undistort_为空  std::shared_ptr<ImuProcess> p_imu_ = nullptr;// imu process

    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        LOG(WARNING) << "No point, skip this scan!";
        return;
    }

    Timer::Evaluate([&, this]() {
        PatchworkppGroundSeg->estimate_ground(*scan_undistort_, *scan_gnd_undistort, *scan_none_gnd, time_taken_Patchworkpp_0);
    },"ground segment before downsample");//这个函数会将 *scan_undistort_ 点云分为两部分：地面点云和非地面点云，分别保存在 *scan_gnd_undistort 和 *scan_none_gnd 中
    // body_n_esti(scan_gnd);
    pub_laser_cloud_gnd_.publish(cloud2msg(*scan_gnd_undistort, scan_undistort_->header.stamp, "lidar"));
    pub_laser_cloud_none_gnd_.publish(cloud2msg(*scan_none_gnd, scan_undistort_->header.stamp, "lidar"));
 
    if (flg_first_scan_) {

        ivox_->AddPoints(scan_undistort_->points); //(origin)向iVOX中添加第一帧点云
        first_lidar_time_ = measures_.lidar_bag_time_; //第一帧的时间为时间同步后数据中的lidar_bag_time_
        flg_first_scan_ = false;

        PointVector scan_gnd_world;
        PointVector scan_gnd_to_add;
        // std::vector<Eigen::Matrix3d> cov_list;
        scan_gnd_world.reserve(scan_gnd_undistort->size());
        // cov_list.reserve(scan_gnd_undistort->size());
        for(size_t i = 0; i < scan_gnd_undistort->size(); ++i){
            PointType point;
            PointBodyToWorld(&(scan_gnd_undistort->points[i]), &(point)); //车体点云转换到世界坐标系下
            scan_gnd_world.emplace_back(point);
            // if(((pow(scan_gnd->points[i].x, 2) + pow(scan_gnd->points[i].y, 2)) < pow(range_filter, 2))){
            if((fabs(scan_gnd_undistort->points[i].x) < range_filter) && (fabs(scan_gnd_undistort->points[i].y) < range_filter)){ //pow算出来的值大量NAN
                PointType point;
                point = scan_gnd_world[i];
                // PointBodyToWorld(&(scan_gnd->points[i]), &point); //车体点云转换到世界坐标系下
                scan_gnd_to_add.emplace_back(point);
            }
        }
        Timer::Evaluate(
                [&, this]() {
            UnionFindNode::BuildVoxelMap(scan_gnd_to_add, plane_map);
            ;}, "Build_planemap");
            // UnionFindNode::pubVoxelMap(plane_map, voxel_map_pub);
        Timer::Evaluate([&, this]() {
            UnionFindNode::pubVoxelMap(plane_map, voxel_map_pub);
        },"pub_planemap");
            // plane_n_estimate(scan_gnd_world, init_voxel_N, init_voxel_d_); //初始地面法向量
            first_voxel_map_ = false;
            // state_ikfom local_state = kf_.get_x(); //初始化位姿
            return;
    }
    
    flg_EKF_inited_ = (measures_.lidar_bag_time_ - first_lidar_time_) >= options::INIT_TIME; //TODO：没太看懂这里的处理，此时，按道理来说，再一次，才会到达这里， flg_EKF_inited_是满足条件的， 为true
    //stp:3 downsample，下采样，将经过去畸变的点云scan_undistort_进行下采样，得到scan_down_body_
    Timer::Evaluate(
        [&, this]() {
            voxel_scan_.setInputCloud(scan_gnd_undistort); //向点云滤波器传递点云
            voxel_scan_.filter(*scan_gnd); // 调用pcl执行点云滤波的下采样(雷达坐标系) 结果存在了这里
        },
        "Downsample gnd PointCloud");

    Timer::Evaluate(
        [&, this]() {
            // voxel_scan_.setInputCloud(scan_undistort_); //向点云滤波器传递点云
            // voxel_scan_.filter(*scan_down_body_); // 调用pcl执行点云滤波的下采样(雷达坐标系)
            // 下采样前进行地面分割    
            voxel_scan_.setInputCloud(scan_none_gnd); //向点云滤波器传递点云
            voxel_scan_.filter(*scan_down_body_); // 调用pcl执行点云滤波的下采样(雷达坐标系)
        },
        "Downsample none_gnd PointCloud");

    int cur_pts = scan_down_body_->size(); //+ scan_gnd->size();//？？？？为什么只考虑地面点云呢？后面的非
    if (cur_pts < 5) { //车体坐标系下的点云数量小于5，直接返回
        LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
        return;
    }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int cnt_gnd = scan_gnd->size();
    int num = cnt_gnd + cur_pts;
    if (is_extract_large_planes_) {
//        D_RECORD_TIME_START;
        if (is_incremental_fitting_) {
            state_ikfom pose_pre_int = kf_.get_x();
            Eigen::Quaternionf q_wl = (pose_pre_int.rot * pose_pre_int.offset_R_L_I).cast<float>(); //计算雷达到世界的旋转四元组？？
            Eigen::Vector3f t_wl = (pose_pre_int.rot * pose_pre_int.offset_T_L_I +
                                    pose_pre_int.pos).cast<float>();
            plane_tracker_.ExtractLargePlanes(scan_down_body_, &t_wl, &q_wl);  //函数提取大平面数据。该函数使用当前的位姿估计 t_wl 和 q_wl 来进行增量平面拟合
        } else {
            plane_tracker_.ExtractLargePlanes(scan_down_body_, (Eigen::Matrix4f *) nullptr); //不增量拟合使用的方法
        }
        scan_down_body_plane_ = plane_tracker_.CurrCloudPlane(); //保存当前提取的大平面点云
        scan_down_body_other_ = plane_tracker_.CurrCloudOther(); //保存不属于大平面的其他点云
//        D_RECORD_TIME_END("ExtractLargePlanes");
    }
    if (is_extract_large_planes_) {
        int n_other = (int) scan_down_body_other_->size(); //提取大平面之后的点云，也就是非平面点云的数量
        scan_down_world_other_->resize(n_other);//重新设置这两个的大小
        nearest_points_.resize(n_other);
    } 
    else {
        // scan_down_world_在MapIncremental()里面写
        scan_down_world_->resize(cur_pts);
        nearest_points_.resize(cur_pts);
    }
    residuals_.resize(num, 0);
    point_selected_surf_.resize(num, true);
    plane_coef_.resize(num, common::V4F::Zero());

//这里的 grav 表示当前状态中的重力方向，get_vect() 方法返回一个三维向量，表示重力加速度的分量（通常是 [0, 0, -9.81] 或类似的值）
    Eigen::Matrix<double, 3, 1> grav_vec;
    grav_vec = state_point_.grav.get_vect();
    ROS_WARN("Before IEKF grav_vec: %f, %f, %f", grav_vec[0], grav_vec[1], grav_vec[2]);
    Timer::Evaluate(
        [&, this]() {
            // iterated state estimation
            double solve_H_time = 0;
            // update the observation model, will call nn and point-to-plane residual computation
            kf_.update_iterated_dyn_share_modified(options::LASER_POINT_COV, solve_H_time);
            // save the state
            state_point_ = kf_.get_x();
            euler_cur_ = SO3ToEuler(state_point_.rot);
            pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;   
        },
        "IEKF Solve and Update");

    grav_vec = state_point_.grav.get_vect();
    ROS_WARN("After IEKF grav_vec: %f, %f, %f", grav_vec[0], grav_vec[1], grav_vec[2]);
    // ROS_INFO("IEKF updated and start updating local map");

    if (is_extract_large_planes_) {
        Timer::Evaluate([&, this]() { MapIncrementalOurs(); }, "    Incremental Mapping Ours");
    } else {
        Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");
    }
   
    PointVector gnd_world;
    PointVector scan_gnd_to_add;
    PointVector gnd_world_outlier;
    // std::vector<Eigen::Matrix3d> cov_list;
    gnd_world.reserve(scan_gnd->size());
    // cov_list.resize(scan_gnd->size());
    for(size_t i = 0; i < scan_gnd->size(); ++i){
        PointType point;
        PointBodyToWorld(&(scan_gnd->points[i]), &(point)); //车体点云转换到世界坐标系下
        gnd_world.emplace_back(point); //FIXME:scan_gnd中的很多点云为nan
        if(((pow(scan_gnd->points[i].x, 2.0) + pow(scan_gnd->points[i].y, 2.0)) < pow(range_filter, 2.0)) && flg_EKF_inited_){ //fixme:这个条件为什么不满足(pow得到的值很多为nan)
        // if((fabs(scan_gnd->points[i].x) < range_filter) && (fabs(scan_gnd->points[i].y) < range_filter) && flg_EKF_inited_){ //fixme:这个条件为什么不满足
            PointType point;
            point = gnd_world[i];
            scan_gnd_to_add.emplace_back(point);//TODO:判断什么样的点需要添加到地图中
       
        }else{
            // ROS_INFO("X Y : %f | %f", scan_gnd->points[i].x, scan_gnd->points[i].y);
            PointType point;
            point = gnd_world[i];
            gnd_world_outlier.emplace_back(point);
        } //不满足的点需要剔除
    }
    CloudPtr gnd_world_ptr(new PointCloudType); //debug
    ROS_INFO("gnd_world | outlier | cnt : %ld | %ld | %ld", gnd_world.size(), gnd_world_outlier.size(),scan_gnd->size());
    gnd_world_ptr->points = gnd_world;
    pub_laser_cloud_origin_.publish(cloud2msg(*gnd_world_ptr, scan_gnd->header.stamp, "camera_init"));
    // gnd_world_ptr->points = gnd_world_outlier;
    gnd_world_ptr->points = scan_gnd_to_add;
    pub_laser_cloud_debug_.publish(cloud2msg(*gnd_world_ptr, scan_gnd->header.stamp, "camera_init"));
    Timer::Evaluate(
        [&, this]() {
        UnionFindNode::UpdateVoxelMap(scan_gnd_to_add, plane_map);
        ;}, "update_planemap");
    if(frame_num_ % voxel_plane_pub_rate == 0){
        ROS_WARN("publish voxelmap");
        Timer::Evaluate([&, this]() {
            UnionFindNode::pubVoxelMap(plane_map, voxel_map_pub);
        },"pub_planemap");
    }
    
    LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " downsamp " << cur_pts
              << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_;

    // publish or save map pcd
    if (run_in_offline_) {
        if (pcd_save_en_) {
            PublishFrameWorld();
        }
        if (path_save_en_) {
            PublishPath(pub_path_);
        }
    } else {
        if (pub_odom_aft_mapped_) {
            PublishOdometry(pub_odom_aft_mapped_);
        }
        if (path_pub_en_ || path_save_en_) {
            PublishPath(pub_path_);
        }
        if (scan_pub_en_ || pcd_save_en_) {
            PublishFrameWorld();
        }
        if (scan_pub_en_ && scan_body_pub_en_) {
            PublishFrameBody(pub_laser_cloud_body_);
        }
        if (scan_pub_en_ && scan_effect_pub_en_) {
            PublishFrameEffectWorld(pub_laser_cloud_effect_world_);
        }
    }

// q_wb * q_bl
    Eigen::Quaternionf q_wl = (state_point_.rot * state_point_.offset_R_L_I).cast<float>();
    Eigen::Vector3f t_wl = (state_point_.rot * state_point_.offset_T_L_I +
                            state_point_.pos).cast<float>();
    plane_tracker_.SetCurrTransWl(t_wl, q_wl);
    // Debug variables
    frame_num_++;
    LOG(INFO) << "frame_num " << frame_num_;
              
}

void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    mtx_buffer_.lock();
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            PointCloudType::Ptr ptr(new PointCloudType());
            preprocess_->Process(msg, ptr);
            lidar_buffer_.push_back(ptr);
            time_buffer_.push_back(msg->header.stamp.toSec());
            last_timestamp_lidar_ = msg->header.stamp.toSec();
        },
        "Preprocess (Standard)");
    mtx_buffer_.unlock();
}

//雷达、IMU回调函数接收到数据存储在buffer中
void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    mtx_buffer_.lock();
    Timer::Evaluate(
        [&, this]() {
            scan_count_++; //雷达扫描计数
            
            //检查时间戳
            if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
                LOG(WARNING) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            //更新当前时间戳
            last_timestamp_lidar_ = msg->header.stamp.toSec();

            // 时间同步关 && 当前imu和雷达时间戳之差大于10（ns） && imu和雷达buffer不为空 
            if (!time_sync_en_ && abs(last_timestamp_imu_ - last_timestamp_lidar_) > 10.0 && !imu_buffer_.empty() &&
                !lidar_buffer_.empty()) {
                LOG(INFO) << "IMU and LiDAR not Synced, IMU time: " << last_timestamp_imu_
                          << ", lidar header time: " << last_timestamp_lidar_;
            }

            //开启了时间同步则计算时间差
            if (time_sync_en_ && !timediff_set_flg_ && abs(last_timestamp_lidar_ - last_timestamp_imu_) > 1 &&
                !imu_buffer_.empty()) {
                timediff_set_flg_ = true;
                timediff_lidar_wrt_imu_ = last_timestamp_lidar_ + 0.1 - last_timestamp_imu_;
                LOG(INFO) << "Self sync IMU and LiDAR, time diff is " << timediff_lidar_wrt_imu_;
            }


            PointCloudType::Ptr ptr(new PointCloudType());
            preprocess_->Process(msg, ptr);
            lidar_buffer_.emplace_back(ptr);
            time_buffer_.emplace_back(last_timestamp_lidar_);
        },
        "Preprocess (Livox)");

    mtx_buffer_.unlock();
}

void LaserMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in) {
    publish_count_++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu_) > 0.1 && time_sync_en_) {
        msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu_ + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer_.lock();
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "imu loop back, clear buffer";
        imu_buffer_.clear();
    }

    last_timestamp_imu_ = timestamp;
    imu_buffer_.emplace_back(msg);
    mtx_buffer_.unlock();
}

bool LaserMapping::SyncPackages() {
    //雷达或imu数据为空则返回错误
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

     // 将激光扫描数据推入系统
    if (!lidar_pushed_) {
        measures_.lidar_ = lidar_buffer_.front(); // 将激光扫描数据存储在measures_.lidar_中
        measures_.lidar_bag_time_ = time_buffer_.front(); // 将激光扫描的时间戳存储在measures_.lidar_bag_time_中

        if (measures_.lidar_->points.size() <= 1) { // 如果输入点云数据点过少
            LOG(WARNING) << "Too few input point cloud!"; // 输出警告信息
            lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_; // 设置激光扫描结束时间为激光扫描的时间戳加上激光扫描平均时间
        } else if (measures_.lidar_->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime_) { // 如果最后一个点的时间小于激光扫描平均时间的一半
            lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_; // 设置激光扫描结束时间为激光扫描的时间戳加上激光扫描平均时间
        } else {
            scan_num_++; // 扫描次数加一
            lidar_end_time_ = measures_.lidar_bag_time_ + measures_.lidar_->points.back().curvature / double(1000); // 设置激光扫描结束时间为激光扫描的时间戳加上最后一个点的时间
            lidar_mean_scantime_ +=
                (measures_.lidar_->points.back().curvature / double(1000) - lidar_mean_scantime_) / scan_num_; // 更新激光扫描平均时间
        }

        measures_.lidar_end_time_ = lidar_end_time_; // 将激光扫描结束时间存储在measures_.lidar_end_time_中
        lidar_pushed_ = true; // 设置激光扫描已推送标志为true
    }

    if (last_timestamp_imu_ < lidar_end_time_) { // 如果最近的IMU时间戳小于激光扫描结束时间
        return false; // 返回false
    }

    /*** push imu_ data, and pop from imu_ buffer ***/
    //stp:1 时间同步，传感器数据从buffer转移到measures_
     // 获取imu_buffer_队列中首部元素的时间戳
    double imu_time = imu_buffer_.front()->header.stamp.toSec();
    // 清空measures_.imu_向量
    measures_.imu_.clear();
    
    // 当imu_buffer_不为空且当前时间戳小于lidar_end_time_时执行循环
    while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
        // 更新时间戳为imu_buffer_队列中首部元素的时间戳
        imu_time = imu_buffer_.front()->header.stamp.toSec();
        // 如果时间戳大于lidar_end_time_，则跳出循环
        if (imu_time > lidar_end_time_) break;
        // 将首部元素添加到measures_.imu_向量中
        measures_.imu_.push_back(imu_buffer_.front());
        // 移除队列中的首部元素
        imu_buffer_.pop_front();
    }
    // 移除lidar_buffer_队列中的首部元素
    lidar_buffer_.pop_front();
    // 移除time_buffer_队列中的首部元素
    time_buffer_.pop_front();
    // 将lidar_pushed_置为false
    lidar_pushed_ = false;
    // 返回true
    return true;
}

void LaserMapping::PrintState(const state_ikfom &s) {
    LOG(INFO) << "state r: " << s.rot.coeffs().transpose() << ", t: " << s.pos.transpose()
              << ", off r: " << s.offset_R_L_I.coeffs().transpose() << ", t: " << s.offset_T_L_I.transpose();
}


void LaserMapping::MapIncrementalOurs() {
    PointVector points_to_add;
    PointVector point_no_need_downsample;
    
    int cur_pts = (int) scan_down_body_other_->size();  // only non-skeleton points 当前非大平面的点
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }
    std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) { //unseq表示无序并行处理每个点
        /* transform to world frame */
        PointBodyToWorld(&(scan_down_body_other_->points[i]), //将lidar坐标系转换到世界坐标系中
                         &(scan_down_world_other_->points[i]));

        /* decide if need add to map */
        PointType &point_world = scan_down_world_other_->points[i];
        if (!nearest_points_[i].empty() && flg_EKF_inited_) { //如果该点有最近邻点并且扩展卡尔曼滤波器（EKF）已初始化   是否要把点添加到地图
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_; //计算栅格中的中心点位

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;//计算第一个最近邻点到中心的距离

            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                point_no_need_downsample.emplace_back(point_world); //如果第一个最近邻点已经足够远离栅格中心点，则有理由相信 point_world 是一个在该区
                                                                    //域内具有代表性的独立点，继续检查其他最近邻点的距离可能是冗余的
                return;
            }//怎么修改栅格距离呢？

            bool need_add = true;
            float dist = common::calc_dist(point_world.getVector3fMap(), center);
            if (points_near.size() >= options::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < options::NUM_MATCH_POINTS; readd_i++) {
                    if (common::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;//把更接近的点用着
                    }
                }
            }
            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    });
    CloudPtr world_cloud_ptr(new PointCloudType); //debug
    world_cloud_ptr->points = points_to_add;
    pub_laser_cloud_debug_2.publish(cloud2msg(*world_cloud_ptr, scan_down_body_->header.stamp, "camera_init"));
    world_cloud_ptr->points = point_no_need_downsample;
    pub_laser_cloud_debug_3.publish(cloud2msg(*world_cloud_ptr, scan_down_body_->header.stamp, "camera_init"));
    Timer::Evaluate([&, this]() {
            ivox_->AddPoints(points_to_add);
            ivox_->AddPoints(point_no_need_downsample);
        }, "    IVox Add Points"); //使用 Timer::Evaluate 计时工具来评估点添加过程的时间
}

//Key_更新(局部)地图
void LaserMapping::MapIncremental() {
    PointVector points_to_add;
    PointVector point_no_need_downsample;

    // std::vector 的 reserve 函数用于预留一定数量的存储空间，但并不改变实际的元素个数。这个函数的主要目的是提前分配足够的内存，以避免因为频繁的动态内存分配而导致的性能开销。
    
    int cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts); //reserve()函数用于预留一定数量的存储空间，但并不改变实际的元素个数。这个函数的主要目的是提前分配足够的内存，以避免因为频繁的动态内存分配而导致的性能开销。
    
    //size_t 提供了一个无符号整数类型，其大小足够大以适应系统中可能的最大对象大小。
    //size_t 通常被用于数组索引和循环计数。size_t 在 32 位系统上通常是一个无符号整数，而在 64 位系统上通常是一个无符号长整数。
    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }
    // 处理点云 std::execution::unseq表示并行执行，执行过程写在了lamda表达式中，这里的index是一个vector，里面存储了点云的索引
    std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) {
        /* transform to world frame */

        PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i])); //车体点云转换到世界坐标系下

        /* decide if need add to map */
        PointType &point_world = scan_down_world_->points[i]; //PointType pcl::PointCloud<pcl::PointXYZINormal>
        if (!nearest_points_[i].empty() && flg_EKF_inited_) { //近邻点不为空(在Obsmodel中获取的),且EKF初始化完成
            const PointVector &points_near = nearest_points_[i]; //

            /*计算点云中心点,center表示该点所在点云网格的中心点的坐标
            这个位置是通过将世界坐标系下的点坐标除以地图的最小滤波器大小（`filter_size_map_min_`），然后向下取整并加上0.5，最后再乘以地图的最小滤波器大小来计算的。
            这样可以确保`center`总是指向一个网格单元的中心。这个计算过程的目的是为了确定一个点是否需要添加到地图中。*/
            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_; //filter_size_map_min_ 局部地图分辨率

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;
            /*如果最近邻点距离网格中心的距离大于网格大小的一半，那么这个点就不需要进行下采样，直接添加到地图中。
            如果一最近邻点距离网格中心的距离小于网格大小的一半，并且在同一个网格中没有其他更接近中心的点，那么这个点就需要添加到地图中。*/
            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) { //这个条件说明当前点离网格中心近
                point_no_need_downsample.emplace_back(point_world);
                return;
            }

            bool need_add = true;
            float dist = common::calc_dist(point_world.getVector3fMap(), center); //dist是当前点到网格中心的距离
            if (points_near.size() >= options::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < options::NUM_MATCH_POINTS; readd_i++) {
                    if (common::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) { //(这个条件说明当前点离网格中心近)如果当前点到网格中心的距离小于最近邻点到网格中心的距离+1e-6，那么这个点就不需要添加到地图中
                        need_add = false;
                        break;}
                }
            }
            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);//当近邻点云为空或者EKF未初始化完成时，将点云添加到地图中
        }
    });

    CloudPtr world_cloud_ptr(new PointCloudType); //debug
    world_cloud_ptr->points = points_to_add;
    pub_laser_cloud_debug_2.publish(cloud2msg(*world_cloud_ptr, scan_down_body_->header.stamp, "camera_init"));
    world_cloud_ptr->points = point_no_need_downsample;
    pub_laser_cloud_debug_3.publish(cloud2msg(*world_cloud_ptr, scan_down_body_->header.stamp, "camera_init"));

    Timer::Evaluate(
        [&, this]() {
            ivox_->AddPoints(points_to_add);
            ivox_->AddPoints(point_no_need_downsample);
        },
        "    IVox Add Points");
}

/**
 * Lidar point cloud registration
 * will be called by the eskf custom observation model
 * compute point-to-plane residual here
 * @param s kf state
 * @param ekfom_data H matrix
 */
void LaserMapping::ObsModelOurs(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    int cnt_gnd = scan_gnd->size();
    int plane_n_pts = (int) scan_down_body_plane_->size();//获取下采样后的平面点云的点数。
    int other_n_pts = (int) scan_down_body_other_->size();
    int total_n_pts = plane_n_pts + other_n_pts + cnt_gnd;
    std::vector<size_t> index_all(total_n_pts);
    for (size_t i = 0; i < total_n_pts; ++i) {
        index_all[i] = i;
    }
    std::vector<size_t> index_plane(index_all.begin(), index_all.begin() + plane_n_pts);
    std::vector<size_t> index_other(index_all.begin(), index_all.begin() + other_n_pts);
    std::vector<size_t> index_gnd(index_all.begin(), index_all.begin() + cnt_gnd);
    
    Timer::Evaluate([&, this]() {
            const int n_planes = plane_tracker_.NumPlanes();
            auto &last_planes = plane_tracker_.LastPlanes();
            
            // R_wb * R_bl
            auto R_wl = (s.rot * s.offset_R_L_I).cast<float>();
            auto t_wl = (s.rot * s.offset_T_L_I + s.pos).cast<float>();

            std::for_each(std::execution::par_unseq,
                          index_plane.begin(), index_plane.end(), [&](const size_t &i) {
                PointType &point_body = scan_down_body_plane_->points[i];
                PointType point_world;

                /* transform to world frame */
                common::V3F p_body = point_body.getVector3fMap();
                point_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_world.intensity = point_body.intensity;

                if (ekfom_data.converge) {
                    float min_dist = std::numeric_limits<float>::max();
                    for (auto &candidate_plane : last_planes) {
                        float dist = Point2PlaneDist(point_world, candidate_plane.coef);
                        if (dist < point_to_plane_thresh_ && dist < min_dist) {
                            min_dist = dist;
                            point_selected_surf_[i] = true;
                            plane_coef_[i] = candidate_plane.coef;
                        }
                    }
                }
                if (point_selected_surf_[i]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[i].dot(temp);
                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        residuals_[i] = pd2;
                    }
                }
            });
            
            std::for_each(std::execution::par_unseq,
                          index_other.begin(), index_other.end(), [&](const size_t &i) {
                PointType &point_body = scan_down_body_other_->points[i];
                PointType point_world;
                
                /* transform to world frame */
                common::V3F p_body = point_body.getVector3fMap();
                point_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_world.intensity = point_body.intensity;

                int idx = i + plane_n_pts;
                auto &points_near = nearest_points_[i];
                if (ekfom_data.converge) {
                    /** Find the closest surfaces in the map **/
                    ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS); 
                    point_selected_surf_[idx] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;
                    if (point_selected_surf_[idx]) {
                        point_selected_surf_[idx] =
                            common::esti_plane(plane_coef_[idx], points_near, options::ESTI_PLANE_THRESHOLD);
                    }
                }
                if (point_selected_surf_[idx]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[idx].dot(temp);
                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        residuals_[idx] = pd2;
//                    } else {
//                        point_selected_surf_[idx] = false;
                    }
                }
            });
            std::for_each(std::execution::par_unseq,
                          index_gnd.begin(), index_gnd.end(), [&](const size_t &i) {

                // debug4++;
                PointType point_gnd = scan_gnd->points[i];
                PointType point_gnd_world;
                
                /* transform to world frame */
                common::V3F p_body = point_gnd.getVector3fMap();
                point_gnd_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_gnd_world.intensity = point_gnd.intensity;

                int idx = i + plane_n_pts + other_n_pts;
                // auto &points_near = nearest_points_[i];
                if (ekfom_data.converge) {
                    /** Find the closest surfaces in the map **/
                    // ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS); 
                    // point_selected_surf_[idx] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;
                    Eigen::Matrix<int, 2, 1> position; //fixme::这里的position是的x异常变大
                    position << static_cast<int>(point_gnd_world.x >= 0 ? point_gnd_world.x * plane_grid_resolution : (point_gnd_world.x * plane_grid_resolution) - 1.0),
                                static_cast<int>(point_gnd_world.y >= 0 ? point_gnd_world.y * plane_grid_resolution : (point_gnd_world.y * plane_grid_resolution) - 1.0);
                                // static_cast<int>(point_gnd_world.z >= 0 ? point_gnd_world.z * z_resolution : (point_gnd_world.z * z_resolution) - 1.0);
                    // ROS_INFO("match postion: %d, %d origin:%f %f", position[0], position[1], point_gnd_world.x, point_gnd_world.y);
                    
                    auto iter = plane_map.find(position);
                    if (iter != plane_map.end() && iter->second->plane_ptr_->is_plane && !iter->second->plane_ptr_->isRootPlane) {
                        plane_coef_[idx] = iter->second->plane_ptr_->n_vec;
                        point_selected_surf_[idx] = true;
                    } else {
                        point_selected_surf_[idx] = false;
                        // residuals_[cnt_pts + j] = 0;
                        // plane_coef_[cnt_pts + j] << 0, 0, 0, 0;
                        
                        // if(!(iter !=plane_map.end())){
                        //     //fixme:在过程当中为什么这里在plane map中找不到节点
                        //     debug1++;
                        // }else if(!(iter->second->plane_ptr_->is_plane)){
                        //     debug2++;
                        // }else if(iter->second->plane_ptr_->isRootPlane){
                        //     debug3++;
                        // }
                    }
                }   

                    // if (point_selected_surf_[idx]) {
                    //     point_selected_surf_[idx] =
                    //         common::esti_plane(plane_coef_[idx], points_near, options::ESTI_PLANE_THRESHOLD);
                    // }
                if (point_selected_surf_[idx]) {
                    auto temp = point_gnd_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[idx].dot(temp);
                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        residuals_[idx] = pd2;
//                    } else {
//                        point_selected_surf_[idx] = false;
                    }
                }
            });
        }, "    ObsModelOurs (Lidar Match)");

    effect_feat_num_ = 0;
    corr_pts_.resize(total_n_pts);
    corr_norm_.resize(total_n_pts);
    for (int i = 0; i < total_n_pts; i++) {
        if (point_selected_surf_[i]) {
            corr_norm_[effect_feat_num_] = plane_coef_[i];
            if (i < plane_n_pts) {
                corr_pts_[effect_feat_num_] = scan_down_body_plane_->points[i].getVector4fMap();
                // count_++;
            } 
            else if (i >= plane_n_pts && i < other_n_pts)
            {
                corr_pts_[effect_feat_num_] =
                    scan_down_body_other_->points[i - plane_n_pts].getVector4fMap();
                    // count_++
            }
            else{
                corr_pts_[effect_feat_num_] =
                    scan_gnd->points[i - plane_n_pts - other_n_pts].getVector4fMap();
                    // count_gnd_++
            }

            corr_pts_[effect_feat_num_][3] = residuals_[i];
            effect_feat_num_++;
        }
    }

    // ROS_INFO("effect_feat_num_: %d, count_: %d, effect_gnd: %d", effect_feat_num_, count_, count_gnd_);
    corr_pts_.resize(effect_feat_num_);
    corr_norm_.resize(effect_feat_num_);

    if (effect_feat_num_ < 1) {
        ekfom_data.valid = false;
        LOG(WARNING) << "No Effective Points!";
        return;
    }

    Timer::Evaluate(
        [&, this]() {
            /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
            ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
            ekfom_data.h.resize(effect_feat_num_);

            index_all.resize(effect_feat_num_);
            const common::M3F off_R = s.offset_R_L_I.toRotationMatrix().cast<float>();
            const common::V3F off_t = s.offset_T_L_I.cast<float>();
            const common::M3F Rt = s.rot.toRotationMatrix().transpose().cast<float>();

            std::for_each(std::execution::par_unseq,
                          index_all.begin(), index_all.end(), [&](const size_t &i) {
                common::V3F point_this_be = corr_pts_[i].head<3>();
                common::M3F point_be_crossmat = SKEW_SYM_MATRIX(point_this_be);
                common::V3F point_this = off_R * point_this_be + off_t;
                common::M3F point_crossmat = SKEW_SYM_MATRIX(point_this);

                /*** get the normal vector of closest surface/corner ***/
                common::V3F norm_vec = corr_norm_[i].head<3>();

                /*** calculate the Measurement Jacobian matrix H ***/
                common::V3F C(Rt * norm_vec);
                common::V3F A(point_crossmat * C);

                if (extrinsic_est_en_) {
                    common::V3F B(point_be_crossmat * off_R.transpose() * C);
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0],
                        B[1], B[2], C[0], C[1], C[2];
                } else {
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0;
                }
                /*** Measurement: distance to the closest surface/corner ***/
                ekfom_data.h(i) = -corr_pts_[i][3];
            });
        }, "    ObsModelOurs (IEKF Build Jacobian)");
}

void LaserMapping::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    int cnt_pts = scan_down_body_->size();
    int cnt_gnd = scan_gnd->size();
    // if(first_voxel_map_) cnt_gnd = 0;
    int num = cnt_pts + cnt_gnd;

    std::vector<size_t> index(cnt_pts); //创建了一个vector，里面存储了点云的索引，大小等于下采样的点数
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }
    std::vector<size_t> index_(cnt_gnd); //创建了一个vector，里面存储了点云的索引，大小等于下采样的点数
    for (size_t i = 0; i < index_.size(); ++i) {
        index_[i] = i;
    }
    int count_up = 0;
    int count_gnd = 0;
    int debug1 = 0;
    int debug2 = 0;
    int debug3 = 0;
    int debug4 = 0;
    Timer::Evaluate(
        [&, this]() {
            auto R_wl = (s.rot * s.offset_R_L_I).cast<float>(); //R_wl = R_wi * R_il
            auto t_wl = (s.rot * s.offset_T_L_I + s.pos).cast<float>();

            /** closest surface search and residual computation 
             * 最近表面搜索和残差计算
             * 计算从体框架到世界框架的旋转（R_wl）和平移（t_wl）
             * 处理每个点，将其转换到世界框架并进行最近邻搜索和残差计算**/
            //这里对标的是非地面点云
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                // if(i<cnt_pts){
                    //雷达坐标系和世界坐标系下的点云
                    PointType &point_body = scan_down_body_->points[i];
                    PointType &point_world = scan_down_world_->points[i];

                    /* transform to world frame */
                    common::V3F p_body = point_body.getVector3fMap(); //雷达坐标系下的点
                    point_world.getVector3fMap() = R_wl * p_body + t_wl;
                    point_world.intensity = point_body.intensity;

                    auto &points_near = nearest_points_[i]; //最近邻点
                    if (ekfom_data.converge) {
                        /** Find the closest surfaces in the map
                         * 寻找最近的平面 **/
                        ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS);
                        
                        point_selected_surf_[i] = points_near.size() >= options::MIN_NUM_MATCH_POINTS; //满足成平面的最小点数
                        if (point_selected_surf_[i]) {
                            point_selected_surf_[i] =
                                common::esti_plane(plane_coef_[i], points_near, options::ESTI_PLANE_THRESHOLD); //满足点要组成平面
                        }
                    }

                    if (point_selected_surf_[i]) {
                        auto temp = point_world.getVector4fMap();
                        temp[3] = 1.0;
                        float pd2 = plane_coef_[i].dot(temp);//点到平面的距离

                        bool valid_corr = p_body.norm() > 81 * pd2 * pd2;//p_body.norm()即点到雷达的距离，pd2是点到平面的距离，如果点到平面的距离小于阈值，则认为该点有效
                        if (valid_corr) {
                            point_selected_surf_[i] = true; //满足距离平面在一定距离的点
                            residuals_[i] = pd2;
                        }
                    count_up ++;

                    }
                });

            ROS_INFO(ekfom_data.converge ? "true" : "false");

            // for (size_t j = 0; j < cnt_gnd; ++j) {
            std::for_each(std::execution::par_unseq, index_.begin(), index_.end(), [&](const size_t &j) {

                debug4++;
                PointType point_gnd = scan_gnd->points[j]; //引用会改变scan_gnd的值!!!!所以呢？
                PointType point_gnd_world = scan_gnd->points[j];
                common::V3F p_body = point_gnd.getVector3fMap();  // 雷达坐标系下的点
                point_gnd_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_gnd_world.intensity = point_gnd.intensity;
                if (ekfom_data.converge) {
                    Eigen::Matrix<int, 2, 1> position; //fixme::这里的position是的x异常变大
                    position << static_cast<int>(point_gnd_world.x >= 0 ? point_gnd_world.x * plane_grid_resolution : (point_gnd_world.x * plane_grid_resolution) - 1.0),
                                static_cast<int>(point_gnd_world.y >= 0 ? point_gnd_world.y * plane_grid_resolution : (point_gnd_world.y * plane_grid_resolution) - 1.0);
                                // static_cast<int>(point_gnd_world.z >= 0 ? point_gnd_world.z * z_resolution : (point_gnd_world.z * z_resolution) - 1.0);
                    // ROS_INFO("match postion: %d, %d origin:%f %f", position[0], position[1], point_gnd_world.x, point_gnd_world.y);
                    
                    auto iter = plane_map.find(position);
                    if (iter != plane_map.end() && iter->second->plane_ptr_->is_plane && !iter->second->plane_ptr_->isRootPlane) {
                        plane_coef_[cnt_pts + j] = iter->second->plane_ptr_->n_vec;
                        point_selected_surf_[cnt_pts + j] = true;
                    } else {
                        point_selected_surf_[cnt_pts + j] = false;
                        // residuals_[cnt_pts + j] = 0;
                        // plane_coef_[cnt_pts + j] << 0, 0, 0, 0;
                        
                        if(!(iter !=plane_map.end())){
                            //fixme:在过程当中为什么这里在plane map中找不到节点
                            debug1++;
                        }else if(!(iter->second->plane_ptr_->is_plane)){
                            debug2++;
                        }else if(iter->second->plane_ptr_->isRootPlane){
                            debug3++;
                        }
                    }
                }   
                if(point_selected_surf_[cnt_pts + j]){
                    auto temp = point_gnd_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[cnt_pts + j].dot(temp);//点到平面的距离
                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;  // p_body.norm()即点到雷达的距离，pd2是点到平面的距离，如果点到平面的距离小于阈值，则认为该点有效
                    if (valid_corr) {
                        point_selected_surf_[cnt_pts + j] = true;  // 满足距离平面在一定距离的点
                        residuals_[cnt_pts + j] = pd2;
                    }
                    point_selected_surf_[cnt_pts + j] = true;
                    count_gnd++;
                }
            });
        },
        "    ObsModel (Lidar Match)");

    ROS_INFO("count_up: %d, count_gnd: %d", count_up, count_gnd);
    ROS_INFO("node: %d, plane: %d, root: %d, sum / loop: %d/%d", debug1, debug2, debug3, debug1+debug2+debug3, debug4);

    effect_feat_num_ = 0;

    corr_pts_.resize(num); //相关点的buffer
    corr_norm_.resize(num); //相关点的法向量buffer

    int count_ = 0;
    int count_gnd_ = 0;
    for (int i = 0; i < num; i++) {
        if (point_selected_surf_[i]) {
            corr_norm_[effect_feat_num_] = plane_coef_[i];
            if(i<cnt_pts){
                corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
                count_++;
            }else{
                corr_pts_[effect_feat_num_] = scan_gnd->points[i-cnt_pts].getVector4fMap();
                count_gnd_++;
            }
            corr_pts_[effect_feat_num_][3] = residuals_[i];
            effect_feat_num_++;
    //         if(i<cnt_pts){
    //             corr_norm_[effect_feat_num_] = plane_coef_[i];
    //             corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
    //             corr_pts_[effect_feat_num_][3] = residuals_[i]; //强度信息为实际为离最近的平面的距离
    //             count_++;
    //         }
    //         else{
    //             corr_norm_[effect_feat_num_] = plane_coef_[i];
    //             corr_pts_[effect_feat_num_] = scan_gnd->points[i-cnt_pts].getVector4fMap();
    //             corr_pts_[effect_feat_num_][3] = residuals_[i]; //强度信息为实际为离最近的平面的距离
    //             count_gnd_++;
    //         }
    //         effect_feat_num_++;
        }
    }

    ROS_INFO("effect_feat_num_: %d, count_: %d, effect_gnd: %d", effect_feat_num_, count_, count_gnd_);

    corr_pts_.resize(effect_feat_num_);
    corr_norm_.resize(effect_feat_num_);

    if (effect_feat_num_ < 1) {
        ekfom_data.valid = false;
        LOG(WARNING) << "No Effective Points!";
        return;}

    Timer::Evaluate(
        [&, this]() {
            /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
            //***测量雅可比矩阵 H 和测量向量的计算***/
            ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
            ekfom_data.h.resize(effect_feat_num_);

            index.resize(effect_feat_num_);
            const common::M3F off_R = s.offset_R_L_I.toRotationMatrix().cast<float>();
            const common::V3F off_t = s.offset_T_L_I.cast<float>();
            const common::M3F Rt = s.rot.toRotationMatrix().transpose().cast<float>();

            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                // if (i < count_){
                    common::V3F point_this_be = corr_pts_[i].head<3>(); //上面算过残差的点的坐标(body坐标系)
                    common::M3F point_be_crossmat = SKEW_SYM_MATRIX(point_this_be);
                    common::V3F point_this = off_R * point_this_be + off_t; //上面算过残差的点的坐标(IMU坐标系)
                    common::M3F point_crossmat = SKEW_SYM_MATRIX(point_this);

                    /*** get the normal vector of closest surface/corner ***/
                    /***获取最近表面/角点的法向量***/
                    common::V3F norm_vec = corr_norm_[i].head<3>();

                    /*** calculate the Measurement Jacobian matrix H ***/
                    /***计算测量雅可比矩阵 H***/
                    common::V3F C(Rt * norm_vec); //C是旋转矩阵Rt和法向量norm_vec的乘积
                    common::V3F A(point_crossmat * C); //A是点乘叉乘的结果

                    if (extrinsic_est_en_) {
                        common::V3F B(point_be_crossmat * off_R.transpose() * C);
                        ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0],
                            B[1], B[2], C[0], C[1], C[2]; //第i行12列数据,是构建雅可比矩阵的一行
                    } else {
                        ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0;
                    }

                    /*** Measurement: distance to the closest surface/corner ***/
                        /***测量：到最近表面/角落的距离***/
                    ekfom_data.h(i) = -corr_pts_[i][3];
                // }
                // else{
                //     common::V3F point_this_be = corr_pts_[i].head<3>(); //上面算过残差的点的坐标(body坐标系)
                //     common::M3F point_be_crossmat = SKEW_SYM_MATRIX(point_this_be);
                //     common::V3F point_this = off_R * point_this_be + off_t; //上面算过残差的点的坐标(IMU坐标系)
                //     common::M3F point_crossmat = SKEW_SYM_MATRIX(point_this);
                //     /*** get the normal vector of closest surface/corner ***/
                //     /***获取最近表面/角点的法向量***/
                //     common::V3F norm_vec = corr_norm_[i].head<3>();
                //     // ROS_INFO("gnd calc norm_vec: %f %f %f", norm_vec[0], norm_vec[1], norm_vec[2]);
                //     /*** calculate the Measurement Jacobian matrix H ***/
                //     /***计算测量雅可比矩阵 H***/
                //     common::V3F C(Rt * norm_vec); //C是旋转矩阵Rt和法向量norm_vec的乘积
                //     common::V3F A(point_crossmat * C); //A是点乘叉乘的结果
                //     if (extrinsic_est_en_) {
                //         common::V3F B(point_be_crossmat * off_R.transpose() * C);
                //         ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0],
                //              B[1],  B[2],  C[0],  C[1],  C[2]; //第i行12列数据,是构建雅可比矩阵的一行
                //     } else {
                //         ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0,
                //             0.0, 0.0, 0.0, 0.0, 0.0;
                //     }

                //     /*** Measurement: distance to the closest surface/corner ***/
                //         /***测量：到最近表面/角落的距离***/
                //     ekfom_data.h(i) = -corr_pts_[i][3];
                    
                // }
            });
        },
        "    ObsModel (IEKF Build Jacobian)");
}

/////////////////////////////////////  debug save / show /////////////////////////////////////////////////////

void LaserMapping::PublishPath(const ros::Publisher pub_path) {
    SetPosestamp(msg_body_pose_);
    msg_body_pose_.header.stamp = ros::Time().fromSec(lidar_end_time_);
    msg_body_pose_.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    path_.poses.push_back(msg_body_pose_);
    if (run_in_offline_ == false) {
        pub_path.publish(path_);
    }
}

void LaserMapping::PublishOdometry(const ros::Publisher &pub_odom_aft_mapped) {
    odom_aft_mapped_.header.frame_id = "camera_init";
    odom_aft_mapped_.child_frame_id = "body";
    odom_aft_mapped_.header.stamp = ros::Time().fromSec(lidar_end_time_);  // ros::Time().fromSec(lidar_end_time_);
    SetPosestamp(odom_aft_mapped_.pose);
    pub_odom_aft_mapped.publish(odom_aft_mapped_);
    auto P = kf_.get_P();
    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        odom_aft_mapped_.pose.covariance[i * 6 + 0] = P(k, 3);
        odom_aft_mapped_.pose.covariance[i * 6 + 1] = P(k, 4);
        odom_aft_mapped_.pose.covariance[i * 6 + 2] = P(k, 5);
        odom_aft_mapped_.pose.covariance[i * 6 + 3] = P(k, 0);
        odom_aft_mapped_.pose.covariance[i * 6 + 4] = P(k, 1);
        odom_aft_mapped_.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odom_aft_mapped_.pose.pose.position.x, odom_aft_mapped_.pose.pose.position.y,
                                    odom_aft_mapped_.pose.pose.position.z));
    q.setW(odom_aft_mapped_.pose.pose.orientation.w);
    q.setX(odom_aft_mapped_.pose.pose.orientation.x);
    q.setY(odom_aft_mapped_.pose.pose.orientation.y);
    q.setZ(odom_aft_mapped_.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odom_aft_mapped_.header.stamp, tf_world_frame_, tf_imu_frame_));
}

void LaserMapping::PublishFrameWorld() {
    if (!(run_in_offline_ == false && scan_pub_en_) && !pcd_save_en_) {
        return;
    }

    PointCloudType::Ptr laserCloudWorld;
    if (dense_pub_en_) {
        PointCloudType::Ptr laserCloudFullRes(scan_undistort_);
        int size = laserCloudFullRes->points.size();
        laserCloudWorld.reset(new PointCloudType(size, 1));
        for (int i = 0; i < size; i++) {
            PointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        }
    } else {
        if (is_extract_large_planes_) {
            laserCloudWorld = scan_down_world_other_;
            int n_plane_pts = (int) scan_down_body_plane_->size();
            scan_down_world_plane_->resize(n_plane_pts);
            std::vector<size_t> index(n_plane_pts);
            for (size_t i = 0; i < n_plane_pts; ++i) {
                index[i] = i;
            }
            std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) {
                PointBodyToWorld(&(scan_down_body_plane_->points[i]), &(scan_down_world_plane_->points[i]));
            });
            *laserCloudWorld += *scan_down_world_plane_;
        } else {
            laserCloudWorld = scan_down_world_;
        }
        // laserCloudWorld = scan_down_world_;
    }

    if (run_in_offline_ == false && scan_pub_en_) {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
        laserCloudmsg.header.frame_id = "camera_init";
        pub_laser_cloud_world_.publish(laserCloudmsg);
        publish_count_ -= options::PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en_) {
        *pcl_wait_save_ += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save_->size() > 0 && pcd_save_interval_ > 0 && scan_wait_num >= pcd_save_interval_) {
            pcd_index_++;
            std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/scans_") + std::to_string(pcd_index_) +
                                       std::string(".pcd"));
            pcl::PCDWriter pcd_writer;
            LOG(INFO) << "current scan saved to /PCD/" << all_points_dir;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
            pcl_wait_save_->clear();
            scan_wait_num = 0;
        }
    }
}

void LaserMapping::PublishFrameBody(const ros::Publisher &pub_laser_cloud_body) {
    int size = scan_undistort_->points.size();
    PointCloudType::Ptr laser_cloud_imu_body(new PointCloudType(size, 1));

    for (int i = 0; i < size; i++) {
        PointBodyLidarToIMU(&scan_undistort_->points[i], &laser_cloud_imu_body->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudmsg_lidar;
    pcl::toROSMsg(*scan_undistort_, laserCloudmsg_lidar);
    laserCloudmsg_lidar.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg_lidar.header.frame_id = "lidar";
    pub_laser_cloud_lidar_.publish(laserCloudmsg_lidar);
    
    common::M3D Lidar_R_wrt_IMU_; //LiDAR相对于IMU的旋转矩阵
    common::V3D Lidar_T_wrt_IMU_;//表示 LiDAR 相对于 IMU 的平移向量。
    Lidar_T_wrt_IMU_ = common::VecFromArray<double>(extrinT_);//从一个存储外参平移信息的数组（extrinT_）中提取出平移向量
    Lidar_R_wrt_IMU_ = common::MatFromArray<double>(extrinR_);
    tf::Matrix3x3 tf3d;//将Lidar_R_wrt_IMU_的旋转矩阵转换为 ROS的tf::Matrix3x3 格式
    tf3d.setValue(static_cast<double>(Lidar_R_wrt_IMU_(0, 0)), static_cast<double>(Lidar_R_wrt_IMU_(0, 1)), static_cast<double>(Lidar_R_wrt_IMU_(0, 2)),
                  static_cast<double>(Lidar_R_wrt_IMU_(1, 0)), static_cast<double>(Lidar_R_wrt_IMU_(1, 1)), static_cast<double>(Lidar_R_wrt_IMU_(1, 2)),
                  static_cast<double>(Lidar_R_wrt_IMU_(2, 0)), static_cast<double>(Lidar_R_wrt_IMU_(2, 1)), static_cast<double>(Lidar_R_wrt_IMU_(2, 2)));
    tf::Quaternion q;
    tf3d.getRotation(q);//将 tf3d 旋转矩阵转换为四元数 q

    tf::Vector3 origin;
    origin.setValue(static_cast<double>(Lidar_T_wrt_IMU_(0)), static_cast<double>(Lidar_T_wrt_IMU_(1)), static_cast<double>(Lidar_T_wrt_IMU_(2)));
    
    static tf::TransformBroadcaster br;
    tf::Transform transform;//创建一个 tf::Transform 对象，将旋转（q）和平移（origin）设定到该对象上，表示 LiDAR 到 IMU 的坐标转换

    transform.setOrigin(origin);
    transform.setRotation(q);
    /*
    sendTransform：通过 tf::TransformBroadcaster 将 transform 变换广播出去。
    laserCloudmsg_lidar.header.stamp：指定变换的时间戳。
    tf_imu_frame_：表示 IMU 坐标系的名称。
    "lidar"：表示 LiDAR 坐标系的名称*/
    br.sendTransform(tf::StampedTransform(transform, laserCloudmsg_lidar.header.stamp, tf_imu_frame_, "lidar"));

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laser_cloud_imu_body, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = "body";
    pub_laser_cloud_body.publish(laserCloudmsg);
    publish_count_ -= options::PUBFRAME_PERIOD;
}

void LaserMapping::PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world) {
    int size = corr_pts_.size();
    PointCloudType::Ptr laser_cloud(new PointCloudType(size, 1));

    for (int i = 0; i < size; i++) {
        PointBodyToWorld(corr_pts_[i].head<3>(), &laser_cloud->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laser_cloud, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = "camera_init";
    pub_laser_cloud_effect_world.publish(laserCloudmsg);
    publish_count_ -= options::PUBFRAME_PERIOD;
}

void LaserMapping::Savetrajectory(const std::string &traj_file) {
    std::ofstream ofs;
    ofs.open(traj_file, std::ios::out);
    if (!ofs.is_open()) {
        LOG(ERROR) << "Failed to open traj_file: " << traj_file;
        return;
    }

    ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
    if (path_.poses.empty()) {
        LOG(WARNING) << "no pose to save!!";
        return;
    }
    double start_time = path_.poses[0].header.stamp.toSec();
    for (const auto &p : path_.poses) {
        ofs << std::fixed << std::setprecision(6) << p.header.stamp.toSec() << " " << std::setprecision(15)
            << p.pose.position.x << " " << p.pose.position.y << " " << p.pose.position.z << " " << p.pose.orientation.x
            << " " << p.pose.orientation.y << " " << p.pose.orientation.z << " " << p.pose.orientation.w << std::endl;
    }

    ofs.close();
}

///////////////////////////  private method /////////////////////////////////////////////////////////////////////
template <typename T>
void LaserMapping::SetPosestamp(T &out) {
    out.pose.position.x = state_point_.pos(0);
    out.pose.position.y = state_point_.pos(1);
    out.pose.position.z = state_point_.pos(2);
    out.pose.orientation.x = state_point_.rot.coeffs()[0];
    out.pose.orientation.y = state_point_.rot.coeffs()[1];
    out.pose.orientation.z = state_point_.rot.coeffs()[2];
    out.pose.orientation.w = state_point_.rot.coeffs()[3];
}

void LaserMapping::PointBodyToWorld(const PointType *pi, PointType *const po) {
    common::V3D p_body(pi->x, pi->y, pi->z);
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LaserMapping::PointBodyToWorld(const common::V3F &pi, PointType *const po) {
    common::V3D p_body(pi.x(), pi.y(), pi.z());
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = std::abs(po->z);
}

void LaserMapping::PointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
    common::V3D p_body_lidar(pi->x, pi->y, pi->z);
    common::V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void LaserMapping::PointBodyLidarToIMU(const common::V3F &pi, PointType *const po) {
    common::V3D p_body_lidar(pi.x(), pi.y(), pi.z());
    common::V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = std::abs(po->z);
}

void LaserMapping::Finish() {
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save_->size() > 0 && pcd_save_en_) {
        std::string file_name = std::string("scans.pcd");
        std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        LOG(INFO) << "current scan saved to /PCD/" << file_name;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
    }

    LOG(INFO) << "finish done";
}

}  // namespace faster_lio