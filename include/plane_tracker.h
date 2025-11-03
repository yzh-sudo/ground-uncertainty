#pragma once

#include <vector>
#include <list>
#include <glog/logging.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Point32.h>
#include <ros/node_handle.h>
#include "sac_model_plane.h"
#include "ransac.h"
//#include "timer.h"



template<class PointType>
class PlaneTracker {
  public:
    
    using PointCloudPtr = typename pcl::PointCloud<PointType>::Ptr;
    
    explicit PlaneTracker(int n_max_planes = 5);
    //用于从给定的点云中提取大的平面
    void ExtractLargePlanes(PointCloudPtr, float *rpyxyz_wl_init_guess = nullptr);//第一个版本接受一个点云指针和一个初始估计的 rpyxyz_wl 数组

    void ExtractLargePlanes(PointCloudPtr, Eigen::Vector3f *t_wl_init_guess = nullptr,
                            Eigen::Quaternionf *q_wl_init_guess = nullptr);//第二个版本接受一个点云指针、一个初始估计的平移向量和四元数

    void ExtractLargePlanes(PointCloudPtr, Eigen::Matrix4f *T_wl_init_guess = nullptr);//第三个版本接受一个点云指针和一个初始估计的4x4变换矩阵

    //设置当前的变换矩阵
    void SetCurrTransWl(float *rpyxyz_wl);//第一个版本使用一个 rpyxyz_wl 数组来设置当前变换
    
    void SetCurrTransWl(Eigen::Vector3f &t_wl, Eigen::Quaternionf &q_wl);//第二个版本使用平移向量和四元数来设置当前变换
    
    void TransCurrPlanes(Eigen::Matrix4f &trans, Eigen::Matrix4f &trans_inv);//给定的变换矩阵和其逆矩阵到当前的平面，trans_inv是变换矩阵的逆矩阵

    void SetPlaneLeastInliers(int n_least_inliers); //提取平面的最小内点数
    
    inline void SetIncrementalFitting(bool is_incremental_fitting) {//是否使用增量拟合
        is_incremental_fitting_ = is_incremental_fitting;
    }

    inline int NumPlanes() const { return all_planes_world_.size(); }//获取当前提取的平面数量
    
    inline std::vector<PlaneWithCentroid> &LastPlanes() { return all_planes_world_; }//获取所有提取的平面
    
    inline PointCloudPtr CurrCloudPlane() { return curr_cloud_plane_; }//获取当前的平面点云
    
    inline PointCloudPtr CurrCloudOther() { return curr_cloud_other_; }//获取当前的非平面点云
  
  private:
    
    int n_least_points_on_a_plane_;
    int n_max_planes_;
    bool is_incremental_fitting_;
    
    PointCloudPtr curr_cloud_;
    PointCloudPtr curr_cloud_plane_;
    PointCloudPtr curr_cloud_other_;
    
    std::vector<PlaneWithCentroid> curr_planes_local_;
    std::vector<PlaneWithCentroid> all_planes_world_;
    
    static const double kAngleEpsilon;
};


template<class PointType>
const double PlaneTracker<PointType>::kAngleEpsilon = std::sin(3 * M_PI / 180); //计算角度为3的正弦值


template<class PointType>
PlaneTracker<PointType>::PlaneTracker(int n_max_planes /* = 5*/)
        : n_least_points_on_a_plane_(-1)  //初始化
        , n_max_planes_(n_max_planes)
        , is_incremental_fitting_(false)
        , curr_cloud_plane_(nullptr)  //设置为空指针
        , curr_cloud_other_(nullptr) {
 
    curr_cloud_.reset(new pcl::PointCloud<PointType>());  //指针管理新的点云
}

template<class PointType>//函数重载，允许作用域定义多个函数，只要参数不一样即可，下面的函数定义一样的原因 //这好像是非增量拟合使用的东西
void PlaneTracker<PointType>::ExtractLargePlanes(PointCloudPtr pc,
                        float *rpyxyz_wl_init_guess/* = nullptr*/) {
    if (!is_incremental_fitting_) {
        ExtractLargePlanes(pc, (Eigen::Matrix4f *) nullptr);//如果没有启用增量拟合,启用模式3
        return;
    }
    LOG_ASSERT(rpyxyz_wl_init_guess != nullptr) << //断言句，如果rpyxyz_wl_init_guess是空的话就输出错误
             "Please provide initial guess if using incremental fitting.";
    Eigen::Affine3f T_w_lnp1;//三维仿射类型，f表示float类型
    pcl::getTransformation(rpyxyz_wl_init_guess[3], rpyxyz_wl_init_guess[4], rpyxyz_wl_init_guess[5], rpyxyz_wl_init_guess[0],
                       rpyxyz_wl_init_guess[1], rpyxyz_wl_init_guess[2], T_w_lnp1);  //给定的平移，俯仰角等参数进行构造变换矩阵   指针指向浮点数数组
    ExtractLargePlanes(pc, &(T_w_lnp1.matrix()));   //不都是进行模式3？？？
}


template<class PointType>  //看前面的情况这好像是增量拟合的时候使用的平面拟合
void PlaneTracker<PointType>::ExtractLargePlanes(PointCloudPtr pc,
                        Eigen::Vector3f *t_wl_init_guess/* = nullptr*/,       //初始的平移向量
                        Eigen::Quaternionf *q_wl_init_guess/* = nullptr*/) {  //四元数，表示旋转，方向和角度
    if (!is_incremental_fitting_) {   //这里面还需要判断吗？感觉不需要啊，在外面运行的时候不是已经判断了？
        ExtractLargePlanes(pc, (Eigen::Matrix4f *) nullptr);
        return;
    }
    LOG_ASSERT(t_wl_init_guess != nullptr && q_wl_init_guess != nullptr) <<
             "Please provide initial guess if using incremental fitting.";
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity(); //定义了一个变换矩阵
    trans.template block<3, 3>(0, 0) = q_wl_init_guess->toRotationMatrix(); //将q_wl_init_guess赋值给 trans 的左上角 3x3 块
    trans.template block<3, 1>(0, 3) = *t_wl_init_guess;//右3角
    ExtractLargePlanes(pc, &trans); //模式1和模式2的目的都是为了进到模式3
}

template<class PointType>
void PlaneTracker<PointType>::ExtractLargePlanes(PointCloudPtr pc,
                        Eigen::Matrix4f *T_wl_init_guess/* = nullptr*/) {  //输入一个变换矩阵

    LOG_ASSERT(n_least_points_on_a_plane_ > 0) <<
         "illegal n_least_points_on_a_plane_: " << n_least_points_on_a_plane_;

    // LOG_ASSERT(T_wl_init_guess != nullptr) <<
    //      "illegal n_least_points_on_a_plane_: " << n_least_points_on_a_plane_;

    pcl::copyPointCloud(*pc, *curr_cloud_); //复制点云
    size_t original = curr_cloud_->size(); //当前点云的大小

    std::vector<PlaneWithCentroid> planes_init_guess;
    if (is_incremental_fitting_ && !LastPlanes().empty()) { //检查使用增量拟合且有以前拟合的平面
        LOG_ASSERT(T_wl_init_guess != nullptr) <<
             "Please provide initial guess if using incremental fitting.";
        // compute plane initial guess here
        for (auto &plane_world : all_planes_world_) {
            PlaneWithCentroid plane_lnp1;
            for (int i = 0; i < 4; ++i) { //坐标转换后的参数向量 （a,b,c,d）
                plane_lnp1.coef[i] = plane_world.coef[0] * (*T_wl_init_guess)(0, i) + plane_world.coef[1] * (*T_wl_init_guess)(1, i) +
                                     plane_world.coef[2] * (*T_wl_init_guess)(2, i) + plane_world.coef[3] * (*T_wl_init_guess)(3, i);
            }
            planes_init_guess.template emplace_back(plane_lnp1);//将转换后的平面 plane_lnp1 添加到 planes_init_guess 向量中,矩阵的每一行去乘以拟合的参数向量
        }
    }

    constexpr int kSampleSize = 250;//表示每次 RANSAC 迭代时从点云中随机选取 3 个点来拟合一个平面。3个点是拟合平面所需的最小数量
    using RansacModel = SampleConsensusModelPlane<PointType, kSampleSize>;//定义了一个别名 RansacModel，它是基于 SampleConsensusModelPlane 模型的类型
    typename RansacModel::Ptr model_plane(new RansacModel(curr_cloud_)); //指向 RansacModel 对象的智能指针，使用当前点云 curr_cloud_ 初始化
    //这是一个 RansacWithPca 类型的对象，用于运行 RANSAC 算法并提取平面
    //指向 RANSAC 平面模型的智能指针，这个模型包含了平面拟合所需的点云数据和算法
    //表示最大提取的平面数量。这个参数限制了算法最多提取的平面数
    //表示拟合平面所需的最小点数量。如果一个平面上点的数量少于该值，则忽略该平面
    RansacWithPca<RansacModel> ransac(model_plane, n_max_planes_, n_least_points_on_a_plane_);
    
    ransac.SetDistanceThreshold(0.05); //设置RANSAC的距离阈值为0.1，用于判断是否属于平面
    
    if (kSampleSize > 3) {
        typename pcl::search::Search<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
        tree->setInputCloud(curr_cloud_);//当前点云为K-D树的输入数据
        //设置 RANSAC 采样点之间的最大距离为 10，使用 K-D 树加速查找。如果两个采样点之间的距离超过 10，则不会将它们作为同一个平面的一部分
        ransac.GetSampleConsensusModel()->SetSamplesMaxDist(10, tree);//设置样本的最大距离为 10，使用 K-D 树进行搜索
        ransac.setMaxIterations(5);  //如果kSampleSize 大于 3，使用 k-D 树 (KdTree) 作为搜索树，并将样本的最大距离设置为 10，最大迭代次数设置为 20
    } 
    else {                          //todo 看看能不能实现比K-D树更好的数据结构
        ransac.setMaxIterations(20);//
    }  //这一段怎么看怎么都不知道啥意思。。
    ransac.ComputeModel(planes_init_guess);//传入的初始平面估计值，可能来自于增量拟合的结果，但是为什么这里没有定义呢？
    
    curr_planes_local_ = ransac.AllModelsCoef(); //找到所有的平面参数
    curr_cloud_plane_ = ransac.GetSampleConsensusModel()->GetInliersCloud(); //获取符合模型的内点
    // curr_cloud_ is now the remaining points
    curr_cloud_other_ = curr_cloud_;//包含不符合平面模型的点（外点）   在哪个地方对curr_cloud_进行的操作呢
}


template <class PointType>
void PlaneTracker<PointType>::SetCurrTransWl(float *rpyxyz_wl) { //前三个元素表示旋转(roll, pitch, yaw)，后三个表示平移(x, y, z)
    Eigen::Affine3f trans;
    pcl::getTransformation(rpyxyz_wl[3], rpyxyz_wl[4], rpyxyz_wl[5], rpyxyz_wl[0],
                           rpyxyz_wl[1], rpyxyz_wl[2], trans);

    TransCurrPlanes(trans.matrix(), trans.inverse().matrix());//将变换矩阵和其逆矩阵应用到当前的平面上
}

template <class PointType>
void PlaneTracker<PointType>::SetCurrTransWl(Eigen::Vector3f &t_wl,
                                             Eigen::Quaternionf &q_wl) {
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.template block<3, 3>(0, 0) = q_wl.toRotationMatrix(); //类似上面的，左上角和右上角
    trans.template block<3, 1>(0, 3) = t_wl;

    Eigen::Matrix3f R_inv = q_wl.conjugate().toRotationMatrix();//计算 q_wl 的共轭四元数对应的旋转矩阵 R_inv，表示逆旋转
    Eigen::Vector3f t_inv = -(R_inv * t_wl);//计算逆平移 t_inv，通过将逆旋转矩阵 R_inv 乘以平移向量 t_wl 并取负值得到
    Eigen::Matrix4f trans_inv = Eigen::Matrix4f::Identity();
    trans_inv.template block<3, 3>(0, 0) = R_inv;
    trans_inv.template block<3, 1>(0, 3) = t_inv;  //计算变换矩阵的逆

    TransCurrPlanes(trans, trans_inv);
}

template <class PointType>
void PlaneTracker<PointType>::TransCurrPlanes(Eigen::Matrix4f &trans, Eigen::Matrix4f &trans_inv) {
    if (is_incremental_fitting_) {
        // If using incremental fitting, some new planes are obtained by checking and
        // refining from old planes, therefore, clear old planes to avoid duplication.
        all_planes_world_.clear();
    }
    std::vector<PlaneWithCentroid> new_planes; //用于存储转换后的全局平面

    for (auto &local_plane : curr_planes_local_) {
        PlaneWithCentroid world_plane;
        for (int i = 0; i < 4; ++i) {
            world_plane.coef[i] = local_plane.coef[0] * trans_inv(0, i) + local_plane.coef[1] * trans_inv(1, i) +
                                  local_plane.coef[2] * trans_inv(2, i) + local_plane.coef[3] * trans_inv(3, i);
        }//通过逆变换矩阵 trans_inv 将局部平面的参数（系数）转换为全局坐标系中的参数
        // TODO: use homogenous coordinate
        for (int i = 0; i < 3; ++i) {
            world_plane.centroid[i] = trans(i, 0) * local_plane.centroid[0] + trans(i, 1) * local_plane.centroid[1] +
                                      trans(i, 2) * local_plane.centroid[2] + trans(i, 3);//通过变换矩阵 trans 将局部平面的质心坐标转换为全局坐标系中的质心
        }
        new_planes.template emplace_back(world_plane); //
    }
    
    // This does not affect performance too much as there wouldn't be too many planes.
    all_planes_world_.insert(all_planes_world_.begin(),
            new_planes.begin(), new_planes.end());//将新转换的平面插入到all_planes_world_ 容器的起始位置

    while (all_planes_world_.size() > n_max_planes_) {
        all_planes_world_.pop_back();//当平面的个数大于最大平面的个数时，从容器末尾移除平面
    }
}

template <class PointType>
void PlaneTracker<PointType>::SetPlaneLeastInliers(int n_least_inliers) {
    LOG_ASSERT(n_least_inliers > 0);
    n_least_points_on_a_plane_ = n_least_inliers;//设置最少的内点数量
}

