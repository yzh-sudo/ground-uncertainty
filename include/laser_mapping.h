#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H

#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <condition_variable>
#include <thread>
#include <Eigen/Dense>

#include "imu_processing.hpp"
#include "ivox3d/ivox3d.h"
#include "options.h"
#include "pointcloud_preprocess.h"
#include "plane_tracker.h"

#include "patchworkpp/patchworkpp.hpp"
#include "patchworkpp/utils.hpp"
#include "common_lib.h"
#include "Vertical_optimization/vertical_optimization.h"
#include "visualization_msgs/MarkerArray.h"

namespace faster_lio {

class LaserMapping {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

#ifdef IVOX_NODE_TYPE_PHC
    using IVoxType = IVox<3, IVoxNodeType::PHC, PointType>;
#else
    using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;
#endif

    LaserMapping();
    ~LaserMapping() {
        scan_down_body_ = nullptr;
        scan_undistort_ = nullptr;
        scan_down_world_ = nullptr;
        scan_gnd = nullptr;
        scan_none_gnd = nullptr;
        LOG(INFO) << "laser mapping deconstruct";
    }

    /// init with ros
    bool InitROS(ros::NodeHandle &nh);

    /// init without ros
    bool InitWithoutROS(const std::string &config_yaml);

    void Run();

    // callbacks of lidar and imu
    void StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in);

    // sync lidar with imu
    bool SyncPackages();

    /// interface of mtk, customized obseravtion model
    void ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);
    void ObsModelOurs(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);

    ////////////////////////////// debug save / show ////////////////////////////////////////////////////////////////
    void PublishPath(const ros::Publisher pub_path);
    void PublishOdometry(const ros::Publisher &pub_odom_aft_mapped);
    void PublishFrameWorld();
    void PublishFrameBody(const ros::Publisher &pub_laser_cloud_body);
    void PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world);
    void Savetrajectory(const std::string &traj_file);

    void Finish();

   private:
    template <typename T>
    void SetPosestamp(T &out);

    void PointBodyToWorld(PointType const *pi, PointType *const po);
    void PointBodyToWorld(const common::V3F &pi, PointType *const po);
    void PointBodyLidarToIMU(PointType const *const pi, PointType *const po);
    void PointBodyLidarToIMU(const common::V3F &pi, PointType *const po);

    void MapIncremental();
    void MapIncrementalOurs();

    void SubAndPubToROS(ros::NodeHandle &nh);

    bool LoadParams(ros::NodeHandle &nh);
    bool LoadParamsFromYAML(const std::string &yaml);

    void PrintState(const state_ikfom &s);

   private:
    /// modules
    IVoxType::Options ivox_options_;
    std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
    //TAG::Imu预处理指针
    std::shared_ptr<ImuProcess> p_imu_ = nullptr;                 // imu process
    PlaneTracker<PointType> plane_tracker_;
    bool is_extract_large_planes_;
    int plane_least_inliers_;
    float point_to_plane_thresh_;
    bool is_incremental_fitting_;

    // local map related
    // 局部地图相关
    float det_range_ = 300.0f;
    double cube_len_ = 0;
    double filter_size_map_min_ = 0;
    bool localmap_initialized_ = false;

    // params
    // 雷达和IMU之间的外参
    std::vector<double> extrinT_{3, 0.0};  // lidar-imu translation
    std::vector<double> extrinR_{9, 0.0};  // lidar-imu rotation
    std::string map_file_path_;

    // point clouds data
    // 点云的数据指针
    CloudPtr scan_undistort_{new PointCloudType()};   // scan after undistortion，去畸变后的点云
    CloudPtr scan_down_body_plane_{new PointCloudType()};
    CloudPtr scan_down_body_other_{new PointCloudType()};
    CloudPtr scan_down_body_{new PointCloudType()};   // downsampled scan in body，车体坐标系下的点云
    CloudPtr scan_down_world_{new PointCloudType()};  // downsampled scan in world，世界坐标系下的点云
    CloudPtr scan_down_world_plane_{new PointCloudType()};
    CloudPtr scan_down_world_other_{new PointCloudType()};
    // 当前扫描的最近点，点云序列的序列
    std::vector<PointVector, Eigen::aligned_allocator<PointVector>> nearest_points_;         // nearest points of current scan，当前扫描的最近点，点云序列的序列
    common::VV4F corr_pts_;                           // inlier pts，平面上的点
    common::VV4F corr_norm_;                          // inlier plane norms，平面法向量

    // 当前扫描的体素滤波器，pcl::VoxelGrid是一个体素滤波器，用于降采样
    pcl::VoxelGrid<PointType> voxel_scan_;            // voxel filter for current scan，当前扫描的体素滤波器
    std::vector<float> residuals_;                    // point-to-plane residuals，点到平面的残差
    std::vector<bool> point_selected_surf_;           // selected points，选中的点
    common::VV4F plane_coef_;                         // plane coeffs，平面系数

    /// ros pub and sub stuffs
    // ros收发的容器
    ros::Subscriber sub_pcl_;
    ros::Subscriber sub_imu_;
    ros::Publisher pub_laser_cloud_world_;
    ros::Publisher pub_laser_cloud_body_;
    ros::Publisher pub_laser_cloud_effect_world_;
    ros::Publisher pub_odom_aft_mapped_;
    ros::Publisher pub_path_;
    std::string tf_imu_frame_;
    std::string tf_world_frame_;

    std::mutex mtx_buffer_;
    std::deque<double> time_buffer_; //储存每帧点云的时间戳rosmsg.header.stamp，会被认为是点云扫描起始时间
    std::deque<PointCloudType::Ptr> lidar_buffer_; //存储的是一帧帧点云的指针
    std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer_;
    nav_msgs::Odometry odom_aft_mapped_;

    /// options
    bool time_sync_en_ = false;
    double timediff_lidar_wrt_imu_ = 0.0;
    double last_timestamp_lidar_ = 0;
    double lidar_end_time_ = 0;
    double last_timestamp_imu_ = -1.0;
    double first_lidar_time_ = 0.0;
    //标记measures_中雷达数据是否已填入
    bool lidar_pushed_ = false; 
    bool first_voxel_map_ = true;
    /// statistics and flags ///
    int scan_count_ = 0;
    int publish_count_ = 0;
    bool flg_first_scan_ = true;
    bool flg_EKF_inited_ = false;
    int pcd_index_ = 0;
    double lidar_mean_scantime_ = 0.0;
    int scan_num_ = 0;
    bool timediff_set_flg_ = false;
    int effect_feat_num_ = 0, frame_num_ = 0;

    ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
    common::MeasureGroup measures_;                    // sync IMU and lidar scan,同步imu和lidar后的数据
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;  // esekf,误差状态扩展卡尔曼滤波器 //key_卡尔曼滤波类的实例化
    state_ikfom state_point_;                          // ekf current state.扩展卡尔曼滤波的当前状态
    vect3 pos_lidar_;                                  // lidar position after eskf update,误差状态卡尔曼滤波更新后的雷达位姿
    common::V3D euler_cur_ = common::V3D::Zero();      // rotation in euler angles,用欧拉角表示的旋转
    bool extrinsic_est_en_ = true;

    /////////////////////////  debug show / save ///////////////////////////////////////
    bool run_in_offline_ = false;
    bool path_pub_en_ = true;
    bool scan_pub_en_ = false;
    bool dense_pub_en_ = false;
    bool scan_body_pub_en_ = false;
    bool scan_effect_pub_en_ = false;
    bool pcd_save_en_ = false;
    bool runtime_pos_log_ = true;
    int pcd_save_interval_ = -1;
    bool path_save_en_ = false;
    std::string dataset_;

    PointCloudType::Ptr pcl_wait_save_{new PointCloudType()};  // debug save
    nav_msgs::Path path_;
    geometry_msgs::PoseStamped msg_body_pose_;
    
    
    public:
        //执行地面分割相关和垂直角度校正的变量
        // PatchWorkpp::PatchWorkpp<PointType>::Ptr ground_remover_;
        bool dense_map_en_ = false;
        boost::shared_ptr<PatchWorkpp<PointType>> PatchworkppGroundSeg; //TODO:patchwork++ 地面分割器
        
        PointCloudType::Ptr scan_gnd{new PointCloudType()};
        PointCloudType::Ptr scan_gnd_undistort{new PointCloudType()};
        PointCloudType::Ptr scan_none_gnd{new PointCloudType()};
        // CloudPtr scan_gnd_world{new PointCloudType()};

        double time_taken_Patchworkpp_0 = 0.0;

        ros::Publisher pub_laser_cloud_lidar_;
        ros::Publisher pub_laser_cloud_origin_;
        ros::Publisher pub_laser_cloud_debug_;
        ros::Publisher pub_laser_cloud_debug_2;
        ros::Publisher pub_laser_cloud_debug_3;
        ros::Publisher pub_laser_cloud_gnd_;
        ros::Publisher pub_laser_cloud_none_gnd_;

        ros::Publisher voxel_map_pub;
        std::unordered_map<plane_key, UnionFindNode *, hash_vec<2>> plane_map;

        common::V3F voxel_N;
        common::V3F init_voxel_N;
        float init_voxel_d_;

            
};

}  // namespace faster_lio

#endif  // FASTER_LIO_LASER_MAPPING_H