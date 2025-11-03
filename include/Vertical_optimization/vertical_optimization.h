#ifndef VERTICAL_OPTIMIZATION_H
#define VERTICAL_OPTIMIZATION_H

#include <common_lib.h>
#include <Eigen/Dense>
#include <execution>
#include <list>
#include <thread>
#include <random>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <functional>  // For std::equal_to
#include <iostream>
#include <ivox3d/eigen_types.h>
#include "visualization_msgs/MarkerArray.h"
#include "ivox3d/ivox3d.h"
#include "patchworkpp/patchworkpp.hpp"
#include "sophus/se2.hpp"
#include "sophus/se3.hpp"
#include "laser_mapping.h"
#include "cmath"
#include "cstdlib"
#include "options.h"
#include "use-ikfom.hpp"

using plane_key = Eigen::Matrix<int, 2, 1>;

#define SKEW_SYM_MATRX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0
namespace faster_lio {

// common parameters
static int plane_id = 0;
static int max_points_size = 100;
static float plane_grid_resolution = 0.5;//ivox_options_.resolution_ options_.inv_resolution_
static float z_resolution = 0.5;
static int update_size_threshold = 10; //估计平面的最低点数要求
float plane_init_threshold = 0.2;
common::V4F local_norm_vector = common::V4F::Zero();
common::V3D local_center = common::V3D::Zero();
// common::V3F local_norm_vector = common::V3F::Zero();
// float local_d = 0.0;
common::V3F n_this = common::V3F::Zero();
common::V3F n_last = common::V3F::Zero();

//voxel
int voxel_plane_pub_rate = 100;
float range_filter = 10.0; //这个参数在laser_mapping.cc中是判断点距离的阈值
float z_filter = 0.5;
float marker_scale = 0.45;
double ranging_cov = 0.02;
double angle_cov = 0.1;
//icp
int max_iteration_ = 20;                // 最大迭代次数
double max_nn_distance_ = 1.0;          // 点到点最近邻查找时阈值
double max_plane_distance_ = 0.05;      // 平面最近邻查找时阈值
double max_line_distance_ = 0.5;        // 点线最近邻查找时阈值
int min_effective_pts_ = 10;            // 最近邻点数阈值
double eps_ = 1e-2;                     // 收敛判定条件
bool use_initial_translation_ = false;  // 是否使用初始位姿中的平移估计

//ransac
float ransac_thre = 0.1;
float plane_thre = 2.5;
int ransac_iteration = 15;

typedef struct pointWithCov {
    common::V3D point;
    common::V3F point_world;
    Eigen::Matrix3d cov;
} pointWithCov;

typedef struct Plane {
    /*** Update Flag ***/
    bool is_plane = false;
    bool is_init = false;
    
    /*** Plane Param ***/
    int main_direction = 0; //表示平面的主要方向,数字表示不同的方程 0:ax+by+z+d=0;  1:ax+y+bz+d=0;  2:x+ay+bz+d=0;
    common::V4F n_vec;  //平面法向量

    // float d_ = 0.0; //平面方程的d
    bool isRootPlane = true;
    int rgb[3] = {0, 0, 0};
    common::V3D sum_points;
    common::M3D sum_squared_points;
    int points_size = 0;   //平面上点的数量?

    common::V3D center;
    common::V3D normal;
    Eigen::Matrix<double, 6, 6> plane_cov; //平面协方差矩阵

    // double xx = 0.0;
    // double yy = 0.0;
    // double zz = 0.0;
    // double xy = 0.0;
    // double xz = 0.0;
    // double yz = 0.0;
    // double x = 0.0;
    // double y = 0.0;
    // double z = 0.0;
    common::V3D center_cov;
    Eigen::Matrix3d covariance = common::M3D::Zero(); //协方差矩阵
} Plane;
typedef std::shared_ptr<Plane> PlanePtr;
typedef const std::shared_ptr<Plane> PlaneConstPtr;


inline Vec3f ToVec3f(const PointType& pt) { return pt.getVector3fMap(); }
inline Vec3d ToVec3d(const PointType& pt) { return pt.getVector3fMap().cast<double>(); }
inline Sophus::SO3d so3_exp(const Vec3d& vec, const double& scale = 1) {
    double norm2 = vec.squaredNorm();
    std::pair<double, double> cos_sinc = MTK::cos_sinc_sqrt(scale * scale * norm2);
    double mult = cos_sinc.second * scale;
    common::V3D result = mult * vec;
    return Sophus::SO3d(Eigen::Quaterniond(cos_sinc.first, result[0], result[1], result[2]));
}

plane_key get_plane_key(PointType pt){
    Eigen::Matrix<int, 2 ,1> result;
    // result << static_cast<int>(pt.x >= 0 ? pt.x * 0.5 : (pt.x * 0.5) - 1.0),
    //           static_cast<int>(pt.y >= 0 ? pt.y * 0.5 : (pt.y * 0.5) - 1.0);
        result(0) = static_cast<int>(pt.x >= 0 ? pt.x * plane_grid_resolution : (pt.x * plane_grid_resolution) - 1.0);
        result(1) = static_cast<int>(pt.y >= 0 ? pt.y * plane_grid_resolution : (pt.y * plane_grid_resolution) - 1.0);
        // result(1) = static_cast<int>(pt.z >= 0 ? pt.z * z_resolution : (pt.z * z_resolution) - 1.0);
    return result; 
}

void CalcVectQuaternion(const Plane &single_plane, geometry_msgs::Quaternion &q) {
    //int main_direction = 0; //0:ax+by+z+d=0;  1:ax+y+bz+d=0;  2:x+ay+bz+d=0;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    if (single_plane.main_direction == 0) {
        a = single_plane.n_vec[0];
        b = single_plane.n_vec[1];
        c = 1;
    } else if (single_plane.main_direction == 1) {
        a = single_plane.n_vec[0];
        b = 1.0;
        c = single_plane.n_vec[1];
    } else if (single_plane.main_direction == 2) {
        a = 1;
        b = single_plane.n_vec[0];
        c = single_plane.n_vec[1];
    }
    double t1 = sqrt(a * a + b * b + c * c);
    a = a / t1;
    b = b / t1;
    c = c / t1;
    double theta_half = acos(c) / 2;
    double t2 = sqrt(a * a + b * b);
    b = b / t2;
    a = a / t2;
    q.w = cos(theta_half);
    q.x = b * sin(theta_half);
    q.y = -1 * a * sin(theta_half);
    q.z = 0.0;
}

/*RANSAC（Random Sample Consensus）：
鲁棒性：RANSAC 对于噪声和局外点相对较好的鲁棒性，因为它通过随机采样和拟合模型来识别内点，并将不符合模型的点视为噪声或局外点。通过迭代的方式，RANSAC可以在一定程度上抵抗数据中的噪声。
适用性：RANSAC 在估计平面模型时适用于数据集中存在局外点或噪声的情况，但需要设置好合适的阈值和迭代次数，同时也需要考虑计算成本和效率。

PCA（Principal Component Analysis）：
鲁棒性：PCA 在平面估计中通常用于计算数据集的主要方向或主要成分，对于数据中的噪声相对较敏感，因为它倾向于找到数据集的主要方向，而噪声可能会对主要方向的计算产生较大影响。
适用性：PCA 在处理较少噪声或局外点的数据集时比较适用，可以有效地找到数据集的主要方向，但在噪声较多的情况下可能会导致估计不准确。

SVD（Singular Value Decomposition）：
鲁棒性：SVD 是一种数学上较为稳健的方法，可以对数据集进行数学建模，求解最小二乘问题，因此对于噪声的鲁棒性较好。然而，它仍然可能受到数据中的极端值或离群点的影响。
适用性：SVD 在点云平面估计中通常能够提供较准确的结果，尤其适用于数据中噪声相对较少的情况。但需要注意的是，在数据集中存在大量噪声或局外点时，SVD 可能会导致估计不准确。
综上所述，RANSAC 在噪声较多或存在局外点的情况下具有较好的鲁棒性，适用于各种数据情况；PCA 适用于较少噪声或局外点的情况下，能够找到数据集的主要方向；
而 SVD 在点云平面估计中通常提供较准确的结果，尤其适用于数据中噪声相对较少的情况。在实际应用中，应根据数据集的特点和需求选择合适的算法。
*/


template <typename T>   //原先的却未使用平面估计
inline bool ODC_esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point) {
    // 检查点的数量是否足够
    if (point.size() < 3) {
        return false;
    }

    // 计算质心
    Eigen::Matrix<T, 3, 1> centroid(0, 0, 0);
    for (const auto& p : point) {
        centroid += p.getVector3fMap().template cast<T>();
    }
    centroid /= static_cast<T>(point.size());

    // 计算协方差矩阵
    Eigen::Matrix<T, 3, 3> covariance_matrix = Eigen::Matrix<T, 3, 3>::Zero();
    for (const auto& p : point) {
        Eigen::Matrix<T, 3, 1> centered_point = p.getVector3fMap().template cast<T>() - centroid;
        covariance_matrix += centered_point * centered_point.transpose();
    }
    covariance_matrix /= static_cast<T>(point.size());

    // 进行特征值分解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> eigen_solver(covariance_matrix);
    Eigen::Matrix<T, 3, 1> normal_vector = eigen_solver.eigenvectors().col(0);

    // 获取平面参数
    pca_result.template head<3>() = normal_vector;
    pca_result(3) = -normal_vector.dot(centroid);

    // 验证点云与平面的距离
    for (const auto& p : point) {
        Eigen::Matrix<T, 4, 1> point_homogeneous;
        point_homogeneous.template head<3>() = p.getVector3fMap().template cast<T>();
        point_homogeneous(3) = 1.0;
        if (std::abs(pca_result.dot(point_homogeneous)) > 0.1) {
            return false;
        }
    }
    
    return true;
}

// template <typename T>
// inline bool ODC_esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point) { //后面的阈值感觉都可以不需要了啊
//     if (point.size() < options::MIN_NUM_MATCH_POINTS) { //3
//         return false;
//     }
//     else {
//         PointVector cur_points = point; //创建一个局部副本
//         do{  
//             Eigen::Matrix<T, 3, 1> normvec;
//             Eigen::Matrix<T, Eigen::Dynamic, 3> A(cur_points.size(), 3);
//             Eigen::Matrix<T, Eigen::Dynamic, 1> b(cur_points.size(), 1);

//             A.setZero();
//             for (size_t i = 0; i < cur_points.size(); ++i)
//             { 
//                 A(i, 0) = static_cast<T>(cur_points[i].x);
//                 A(i, 1) = static_cast<T>(cur_points[i].y);
//             }

//             Eigen::Matrix<T, 3, 3> covarianceMatrix = (A.transpose() * A) / static_cast<T>(cur_points.size()); // 计算数据的协方差
//             Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> eigenSolver(covarianceMatrix); // 进行特征值分解
//             normvec = eigenSolver.eigenvectors().col(0).template cast<T>(); // 拟合平面的法向量，对应最小特征值的那个向量


//             Eigen::Matrix<T, 3, 1> mean = A.colwise().mean(); // 计算所有点的均值向量
//             T d = -normvec.dot(mean); // 计算参数d

//             //获取平面的参数
//             pca_result.template head<3>() = normvec;
//             pca_result(3) = d;

//             // midnorm.template head<3>() = normvec;
//             // midnorm[0] = normvec[0];
//             // midnorm[1] = normvec[1];
//             // midnorm[2] = normvec[2];
//             // midnorm[3] = d;
//             T n = pca_result.norm();

//             Eigen::Matrix<T, 4, 1> midnorm;
//             midnorm(0) = normvec(0) / n;
//             midnorm(1) = normvec(1) / n; // 归一化，方便下面进行点到平面的距离计算
//             midnorm(2) = normvec(2) / n;
//             midnorm(3) = d / n; // 这一段是在进行平面估计，然后计算估计出来的平面的法向量

//             std::vector<T> distances(cur_points.size());
//             for (size_t i = 0; i < cur_points.size(); ++i) {
//                 Eigen::Matrix<T, 4, 1> temp = cur_points[i].getVector4fMap().template cast<T>(); // 将坐标设成一个四维数组，并进行类型转换
//                 temp[3] = 1.0; // 将最后一个数设成1
//                 T distance = fabs(midnorm.dot(temp)); // 计算每个点到平面的距离
//                 distances[i] = distance; // 将距离存储到数组中
//             }

//             std::vector<T> absDeviations(distances.size());
//             size_t size = distances.size(); // 点到平面距离的数量就是点的数量
//             std::sort(distances.begin(), distances.end()); // 进行排序

//             T median = 0.0;
//             T mad = 0.0;
//             if (size % 2 == 0) { // 用来求中位数
//                 median = 0.5 * (distances[size / 2] + distances[size / 2 + 1]);
//             } else {
//                 median = distances[(size + 1) / 2];
//             }

//             for (size_t i = 0; i < distances.size(); ++i) {
//                 absDeviations[i] = std::abs(distances[i] - median);
//             }
//             std::sort(absDeviations.begin(), absDeviations.end());

//             if (size % 2 == 0) {
//                 mad = 0.5 * (absDeviations[size / 2] + absDeviations[size / 2 + 1]) * 1.4826;
//             } else {
//                 mad = absDeviations[(size + 1) / 2] * 1.4826;
//             }

//             //计算鲁棒值
//             std::vector<T> robustZScores(distances.size());
//             for (size_t i = 0; i < distances.size(); ++i) {
//                 robustZScores[i] = (distances[i] - median) / mad;
//             }

//             std::vector<bool> outliers(robustZScores.size()); // 判断离群点
//             for (size_t i = 0; i < robustZScores.size(); ++i) {
//                 outliers[i] = std::abs(robustZScores[i]) >= 2.5; // 计算robustZScores的绝对值，再跟阈值进行比较，outliers 是 bool 类型，大于或者等于阈值就是 true
//             }
 
//             PointVector inliers;
//             for (size_t i = 0; i < cur_points.size(); ++i) {
//                 if (!outliers[i]) {
//                     inliers.push_back(cur_points[i]); // 将未剔除的点加入到 inliers 中
//                 }
//             }

//             if (inliers.size() == cur_points.size()) { // 发现所有点都是局内点
//                 return true;
//             }

//             cur_points = inliers;
//         } while(cur_points.size() >= options::MIN_NUM_MATCH_POINTS); 
//         // if (cur_points.size() < options::MIN_NUM_MATCH_POINTS){return false;}    
//     }   
//     return false;
// }  //这里是我自己新改的

class UnionFindNode {
public:
    // std::vector<pointWithCov> temp_points_; // all points in an octo tree
    PointVector temp_points_; // all points
    // std::vector<Eigen::Matrix3d> cov_list;
    PlanePtr plane_ptr_;    // 指向平面的指针
    // double voxel_center_[3]{}; // 中心坐标x, y, z
    // double voxel_center_[2]{}; // 中心坐标x, y, z
    // float voxel_center_[2]{}; // 中心坐标x, y, z
    int all_points_num_;
    int new_points_num_;
    
    bool init_node_; // 是否初始化
    bool update_enable_; // 是否可以更新
    bool is_plane; //平面标志位
    int id; // 节点标识符
    UnionFindNode *rootNode; //根节点

    UnionFindNode(){
        temp_points_.clear();
        // cov_list.clear();
        new_points_num_ = 0;
        all_points_num_ = 0;
        init_node_ = false;
        update_enable_ = true;
        plane_ptr_ = std::make_shared<Plane>();
        // cov_list = std::vector<Eigen::Matrix3d>();
        /*** Visualization Set RootNode Color ***/
        //初始化颜色
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 255);
        plane_ptr_->rgb[0] = dis(gen);
        plane_ptr_->rgb[1] = dis(gen);
        plane_ptr_->rgb[2] = dis(gen);
        rootNode = this;
    }

    
    
    void plane_init(const PointVector &points, const PlanePtr& plane, UnionFindNode* node) {
        //TODO:找到更好的平面法向量计算方法和平面判断方法
        // plane->is_plane = true; //fixme::这里默认是平面,缺乏对奇怪平面的判断
        // common::esti_normvector(plane->n_vec, points, 0.01f, points.size());
        // plane_n_estimate(points, plane->n_vec, plane->d_);
        // plane->is_plane = ransac_plane_estimate(plane->n_vec, plane->center_cov, points, options::ESTI_PLANE_THRESHOLD); //满足点要组成平面
        plane->is_plane = ODC_esti_plane(plane->n_vec, points); //满足点要组成平面
        //fixme:似乎没有这个必要
        // common::V4F abd_bias = (plane->n_vec - local_norm_vector).cwiseAbs();
        // if ((abd_bias[0] < plane_init_threshold && abd_bias[1] < plane_init_threshold)) {
        //     plane->is_plane = true; //如果与全局法向量差别不大,则认为是平面
        // }
        if(plane->is_plane){
            // clc_plane_cov(points, node->cov_list, plane); 
            node->is_plane = true;
            if (!plane->is_init) { //如果是一个平面,且没有初始化
                node->id = plane_id;
                plane_id++;
                plane->is_init = true;
            }
        }
        else{
            if (!plane->is_init) { //如果不是平面,且没有初始化
                node->id = plane_id;
                plane_id++;
                plane->is_init = true; //所以所谓平面的初始化就是给id赋值?
            }
            plane->is_plane = false;
            node->is_plane = false;
        }
    }



    void InitUnionFindNode() {
        if (temp_points_.size() > update_size_threshold) {
            //init_plane(temp_points_, plane_ptr_);
            plane_init(temp_points_, plane_ptr_, this);
            if (plane_ptr_->is_plane) {
                if (temp_points_.size() > max_points_size) {
                    update_enable_ = false; //初始化时默认要更新，如果点数大于100，则不更新
                    float mean_z = 0;
                    for(auto & iter : temp_points_) {
                        mean_z += iter.z;
                        }
                    plane_ptr_->center[2] = mean_z / temp_points_.size(); //在清空点云之前计算z

                }
            }
            init_node_ = true;
            new_points_num_ = 0;
            //      temp_points_.clear();
        }
    }

    static void BuildVoxelMap(const PointVector &input_points, 
                    std::unordered_map<plane_key, UnionFindNode *, hash_vec<2>> &feat_map) {
        // 调试相关,看看当前帧的               
        // common::esti_normvector(local_norm_vector, input_points, 0.01f, input_points.size());
        // plane_n_estimate(input_points, local_norm_vector, local_d);
        // ransac_plane_estimate(local_norm_vector, local_center, input_points, options::ESTI_PLANE_THRESHOLD); 
        ODC_esti_plane(local_norm_vector, input_points); 

        // ROS_INFO("local_norm_vector: %f, %f, %f", local_norm_vector[0], local_norm_vector[1], local_norm_vector[2]);
        
        uint plsize = input_points.size();

        for (uint i = 0; i < plsize; i++) { //遍历点云
            const PointType p_v = input_points[i];
            // auto position = get_plane_key(p_v);
            Eigen::Matrix<int, 2, 1> position;
            position << static_cast<int>(p_v.x >= 0 ? p_v.x * plane_grid_resolution : (p_v.x * plane_grid_resolution) - 1.0),
                        static_cast<int>(p_v.y >= 0 ? p_v.y * plane_grid_resolution : (p_v.y * plane_grid_resolution) - 1.0);
                        
            // ROS_INFO("build:position: %d, %d origin: %f %f", position[0], position[1], p_v.x, p_v.y);
            //iter是键值对
            auto iter = feat_map.find(position); //FIXME::只用x,y的话,当出现二层楼/地图向上翘起时(解决办法:仅保存最新的10个点,对其进行拟合,如果平面参数离当前平面差距过大则判断是否回环,如果没有发生回环则在当前节点构建新的平面)
                                                    //或者参考ivox删除较长时间未更新的节点

            if (iter != feat_map.end()) { //判断
                feat_map[position]->temp_points_.push_back(p_v);
                feat_map[position]->new_points_num_++;
                
                // float mean_z = 0;
                // for(auto & iter :feat_map[position]->temp_points_) {
                //     mean_z += iter.z;
                // }               
                // feat_map[position]->plane_ptr_->center[2] = mean_z / feat_map[position]->temp_points_.size();
            } else {
                auto *octo_tree = new UnionFindNode();
                feat_map[position] = octo_tree;
                feat_map[position]->plane_ptr_->center[0] = (0.5 + static_cast<double>(position[0])) * (1.0 / plane_grid_resolution);
                feat_map[position]->plane_ptr_->center[1] = (0.5 + static_cast<double>(position[1])) * (1.0 / plane_grid_resolution);
                // feat_map[position]->plane_ptr_->center[2] = (0.5 + static_cast<double>(position[2])) * (1.0 / z_resolution);
                feat_map[position]->plane_ptr_->center[2] = p_v.z; 
                feat_map[position]->temp_points_.push_back(p_v);
                feat_map[position]->new_points_num_++;

                feat_map[position]->plane_ptr_->main_direction = 0; //TODO:缺乏对平面的判断
            }
        }
        for (auto & iter :feat_map) {
            iter.second->InitUnionFindNode();
        }

    }

    static void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub,
                    const std::string& plane_ns, const Plane &single_plane,
                    const float alpha, const common::V3D& rgb, int id) {
        visualization_msgs::Marker plane;
        plane.header.frame_id = "camera_init";
        plane.header.stamp = ros::Time();
        plane.ns = plane_ns;
        plane.id = id;
        if (single_plane.isRootPlane) {
            plane.type = visualization_msgs::Marker::CYLINDER; //圆柱体为根节点
        } else {
            plane.type = visualization_msgs::Marker::CUBE;
        }

        plane.action = visualization_msgs::Marker::ADD;
        // todo:重新计算参数

        plane.pose.position.x = single_plane.center[0];
        plane.pose.position.y = single_plane.center[1];
        plane.pose.position.z = single_plane.center[2];
        geometry_msgs::Quaternion q;
        CalcVectQuaternion(single_plane, q);
        plane.pose.orientation = q;
        plane.scale.x = 0.95 / plane_grid_resolution;
        plane.scale.y = 0.95 / plane_grid_resolution;
        plane.scale.z = 0.01;
        plane.color.a = alpha;
        plane.color.r = static_cast<float>(rgb[0]);
        plane.color.g = static_cast<float>(rgb[1]);
        plane.color.b = static_cast<float>(rgb[2]);
        plane.lifetime = ros::Duration();
        plane_pub.markers.push_back(plane);
    }

    static void pubVoxelMap(const std::unordered_map<plane_key, UnionFindNode *, hash_vec<2>> &voxel_map,
                 const ros::Publisher &plane_map_pub) {
        // ros::Rate loop(500);
        float use_alpha = 1;
        visualization_msgs::MarkerArray voxel_plane;
        // jsk_recognition_msgs::PolygonArray voxel_plane;
        voxel_plane.markers.reserve(1000000);
        std::vector<UnionFindNode *> pub_node_list;
        int cnt = 0;
        int updated_ = 0;
        for (const auto & iter : voxel_map) {
            if (!iter.second->update_enable_) { //update_enable_默认为真,也就是说这里只发布停止更新的节点?
                pub_node_list.emplace_back(iter.second);
                // ROS_INFO("center.z %f", iter.second->plane_ptr_->center[2]);
                updated_++;
            }
            cnt++;
        }

        ROS_INFO("pub updated_node / total : %d / %d", updated_, cnt);

        for (auto & node : pub_node_list) {
            UnionFindNode *curRootNode = node;
            while (curRootNode->rootNode != curRootNode) {
                curRootNode = curRootNode->rootNode;
            }

            common::V3D plane_rgb(curRootNode->plane_ptr_->rgb[0] / 256.0,
                        curRootNode->plane_ptr_->rgb[1] / 256.0,
                        curRootNode->plane_ptr_->rgb[2] / 256.0);
            float alpha;
            if (curRootNode->plane_ptr_->is_plane) {
                alpha = use_alpha;

                Plane newP;
                if(node == curRootNode) {
                    newP.isRootPlane = true;
                }else{
                    newP.isRootPlane = false;
                }
                newP.n_vec = curRootNode->plane_ptr_->n_vec;
                newP.center[0] = node->plane_ptr_->center[0];
                newP.center[1] = node->plane_ptr_->center[1];
                newP.center[2] = node->plane_ptr_->center[2]; //这里相当于重新构造一个平面，这里的center是可视化的中心

                newP.main_direction = curRootNode->plane_ptr_->main_direction;
                pubSinglePlane(voxel_plane, "plane", newP, alpha, plane_rgb, node->id);
            } else {
                alpha = 0;
            }
    }
    // std::cout << "voxel_plane size:" << voxel_plane.markers.size() << std::endl;
    plane_map_pub.publish(voxel_plane);
    // loop.sleep();
    }   
    
    void UpdatePlane(const PointType &pv,
                    //  const Eigen::Matrix3d &cov,
                     plane_key &position, std::unordered_map<plane_key, UnionFindNode *, hash_vec<2>> &feat_map) {
        //fixme:update没有更新平面的cov
        if (!init_node_) {
            // cov_list.push_back(cov);
            new_points_num_++;
            all_points_num_++;
            temp_points_.push_back(pv);
            if (temp_points_.size() > update_size_threshold) { //满足点数要求即可初始化节点
                InitUnionFindNode();
            }
        } 
        else { //已经初始化
            if (plane_ptr_->is_plane) { //如果是平面
                if (update_enable_) {
                    
                    temp_points_.push_back(pv);
                    // cov_list.push_back(cov);
                    // clc_plane_cov(temp_points_, cov_list, plane_ptr_); //更新平面的cov
                    new_points_num_++;
                    all_points_num_++;

                    // if (new_points_num_ > update_size_threshold) { //当新增点数大于拟合新平面的阈值时
                    //     if (update_enable_) {
                    //         plane_init(temp_points_, plane_ptr_, this);
                    //     }
                    //     new_points_num_ = 0;
                    // }
                    if (all_points_num_ > max_points_size) {
                        update_enable_ = false;
                        // std::vector<PointType>().swap(temp_points_);
                        float mean_z = 0;
                        for(auto & iter : temp_points_) {
                            mean_z += iter.z;
                        }
                        plane_ptr_->center[2] = mean_z / temp_points_.size(); //在清空点云之前计算z
                        // plane_n_estimate(temp_points_, plane_ptr_->n_vec, plane_ptr_->d_);
                        // ransac_plane_estimate(plane_ptr_->n_vec, plane_ptr_->center_cov, temp_points_, options::ESTI_PLANE_THRESHOLD); 
                        ODC_esti_plane(plane_ptr_->n_vec, temp_points_);
                        // plane_ptr_->is_plane = 
                        temp_points_.clear();
                        // cov_list.clear();
                    }
                } 
                else { 
                    // 点数已经足够(更新停止,update_enable_置为false)，可实行融合
                    // nowRealRootNode: realRootNode
                    UnionFindNode *nowRealRootNode = this;
                    // 寻找当前节点的根节点
                    // 本while也许可以直接删除
                    while (nowRealRootNode != nowRealRootNode->rootNode) { //如果当前节点的根节点不是自己,则一直找到根节点
                        nowRealRootNode = nowRealRootNode->rootNode;
                    }

                    for (int k = 0; k < 4; k++) {
                        // 检查周围6个节点 明明就只检查了二维哈希表的前后左右四个
                        switch (k) {
                            case 0:
                                position[0] -= 1;
                                break;
                            case 1:
                                position[0] += 2; //+2是因为上一步改变了position的值
                                break;
                            case 2:
                                position[0] -= 1;
                                position[1] -= 1;
                                break;
                            case 3:
                                position[1] += 2;
                                break;

                            default:
                                break;
                        }
                        auto iter = feat_map.find(position);
                        if (iter != feat_map.end()) { //如果邻居节点存在
                            //neighbor_plane所在的octotree可能不是root,所以要找到它的根节点
                            UnionFindNode *neighRealRootNode = iter->second;
                            //找邻居的根节点
                            while (neighRealRootNode != neighRealRootNode->rootNode) {
                                neighRealRootNode = neighRealRootNode->rootNode; //找到邻居的根节点
                            }
                            //邻居与当前平面可能是相同root或点数不够
                            if (neighRealRootNode == nowRealRootNode || neighRealRootNode->update_enable_) {
                                continue;
                            }
                            PlanePtr neighbor_plane = neighRealRootNode->plane_ptr_; //邻居根结点的平面指针
                            PlanePtr now_plane = nowRealRootNode->plane_ptr_;
                            /*** Plane Merging ***/
                            if (neighbor_plane->is_plane) {
                                if (neighbor_plane->main_direction == now_plane->main_direction) {
                                    common::V4F abd_bias = (neighbor_plane->n_vec - now_plane->n_vec).cwiseAbs(); //绝对值,cwiseAbs()是对应元素的绝对值

                                    if ((abd_bias[0] < 0.05 && abd_bias[1] < 0.05)) {
                                        // nowRealRootNode->plane_ptr_->n_vec = (nowRealRootNode->plane_ptr_->n_vec + neighRealRootNode->plane_ptr_->n_vec) / 2;
                                        float p1 = local_norm_vector.dot(nowRealRootNode->plane_ptr_->n_vec);
                                        float p2 = local_norm_vector.dot(neighRealRootNode->plane_ptr_->n_vec);
                                        if(std::abs(p1) < std::abs(p2)){
                                            neighRealRootNode->rootNode = nowRealRootNode;
                                            neighbor_plane->isRootPlane = false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    /*return;*/
                }
            } 
            else { //初始化过节点但不是平面//TODO:可以在这里加入异常值剔除的逻辑
                if (update_enable_) {
                    new_points_num_++;
                    all_points_num_++;

                    temp_points_.push_back(pv);
                    // cov_list.push_back(cov);
                    if (all_points_num_ >= max_points_size) {
                        // plane_init(temp_points_, plane_ptr_, this); ///
                        update_enable_ = false;
                        // std::vector<PointType>().swap(temp_points_);
                        temp_points_.clear();
                        // cov_list.clear();
                    }
                }
            }
        }
    }

    static void UpdateVoxelMap(const PointVector &input_points, 
                    std::unordered_map<plane_key, UnionFindNode *, hash_vec<2>> &feat_map) {
                    // const std::vector<Eigen::Matrix3d> &cov_list) {
        // common::esti_normvector(local_norm_vector, input_points, 0.01f, input_points.size());
        // plane_n_estimate(input_points, local_norm_vector, local_d);
        // ransac_plane_estimate(local_norm_vector, local_center, input_points, options::ESTI_PLANE_THRESHOLD);
        ODC_esti_plane(local_norm_vector, input_points);
         
        ROS_INFO("local_norm_vector: %f, %f, %f, %f", local_norm_vector[0], local_norm_vector[1], local_norm_vector[2], local_norm_vector[3]);
        
        if(local_norm_vector[0] == NAN || local_norm_vector[1] == NAN || local_norm_vector[2] == NAN || local_norm_vector[3] == NAN){
            for(auto & iter : input_points){
                ROS_INFO("input_points: %f, %f, %f", iter.x, iter.y, iter.z);
            }
        }
        uint plsize = input_points.size();
        for (uint i = 0; i < plsize; i++) {
            const PointType& p_v = input_points[i];

            // auto position = get_plane_key(p_v);
            Eigen::Matrix<int, 2, 1> position;
            position << static_cast<int>(p_v.x >= 0 ? p_v.x * plane_grid_resolution : (p_v.x * plane_grid_resolution) - 1.0),
                        static_cast<int>(p_v.y >= 0 ? p_v.y * plane_grid_resolution : (p_v.y * plane_grid_resolution) - 1.0);
                        // static_cast<int>(p_v.z >= 0 ? p_v.z * z_resolution : (p_v.z * z_resolution) - 1.0);
            // ROS_INFO("update:position: %d, %d origin: %f %f", position[0], position[1], p_v.x, p_v.y);
            auto iter = feat_map.find(position);
            if (iter != feat_map.end()) { //这个是第0层
                feat_map[position]->UpdatePlane(p_v, position, feat_map); //TODO:更新时能否按帧更新voxelmap,每个点计算一次updateplane有点费时间
            } else {
                auto *node = new UnionFindNode();
                feat_map[position] = node;
                feat_map[position]->plane_ptr_->center[0] = (0.5 + static_cast<double>(position[0])) * (1.0 / plane_grid_resolution);
                feat_map[position]->plane_ptr_->center[1] = (0.5 + static_cast<double>(position[1])) * (1.0 / plane_grid_resolution);
                // feat_map[position]->plane_ptr_->center[2] = (0.5 + static_cast<double>(position[2])) * (1.0 / z_resolution);
                feat_map[position]->plane_ptr_->center[2] = p_v.z;
                feat_map[position]->UpdatePlane(p_v, position, feat_map);
            }
        }
    };

};

//输入应该为当前帧IMU坐标系下的点云,init_pose为faster-lio估计得到的位姿
template <class scalar>
inline std::pair<scalar, scalar> cos_sinc_(const scalar& x2) {
    using std::cos;
    using std::sin;
    using std::sqrt;
    static scalar const taylor_0_bound = boost::math::tools::epsilon<scalar>();
    static scalar const taylor_2_bound = sqrt(taylor_0_bound);
    static scalar const taylor_n_bound = sqrt(taylor_2_bound);

    if (x2 < 0) {
        ROS_WARN("argument must be non-negative: %f", x2);
    }
    // assert(x2 >= 0 && "argument must be non-negative");

    // FIXME check if bigger bounds are possible
    if (x2 >= taylor_n_bound) {
        // slow fall-back solution
        scalar x = sqrt(x2);
        return std::make_pair(cos(x), sin(x) / x);  // x is greater than 0.
    }

    // FIXME Replace by Horner-Scheme (4 instead of 5 FLOP/term, numerically more stable, theoretically cos and sinc can
    // be calculated in parallel using SSE2 mulpd/addpd)
    // TODO Find optimal coefficients using Remez algorithm
    static scalar const inv[] = {1 / 3., 1 / 4., 1 / 5., 1 / 6., 1 / 7., 1 / 8., 1 / 9.};
    scalar cosi = 1., sinc = 1;
    scalar term = -1 / 2. * x2;
    for (int i = 0; i < 3; ++i) {
        cosi += term;
        term *= inv[2 * i];
        sinc += term;
        term *= -inv[2 * i + 1] * x2;
    }

    return std::make_pair(cosi, sinc);
}
inline MTK::SO3<double> EXP_(const common::V3D& vec, const double& scale = 1) {
    double norm2 = vec.squaredNorm();
    std::pair<double, double> cos_sinc = cos_sinc_(scale * scale * norm2);
    double mult = cos_sinc.second * scale;
    Vec3d result = mult * vec;
    return MTK::SO3<double>(Eigen::Quaterniond(cos_sinc.first, result[0], result[1], result[2]));
}



void calcBodyCov(common::V3D &pb, const float range_inc,
                 const float degree_inc, Eigen::Matrix3d &cov) {
    double range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]); //点到原点的距离
    double range_var = range_inc * range_inc; // range_inc (velodyne) = 0.02, 计算方差
    Eigen::Matrix2d direction_var;
    direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
            pow(sin(DEG2RAD(degree_inc)), 2); //
    common::V3D direction(pb);
    // 防止NAN
    if (direction(2) == 0) {
        direction(2) = 1e-6;
    }
    direction.normalize();
    Eigen::Matrix3d direction_hat;
    direction_hat << 0, -direction(2), direction(1), direction(2), 0,
            -direction(0), -direction(1), direction(0), 0; //SE(3)的斜对称矩阵
    common::V3D base_vector1(1, 1,
                     -(direction(0) + direction(1)) / direction(2));
    base_vector1.normalize();
    common::V3D base_vector2 = base_vector1.cross(direction);
    base_vector2.normalize();
    Eigen::Matrix<double, 3, 2> N;
    N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
            base_vector1(2), base_vector2(2);
    Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
    cov = direction * range_var * direction.transpose() +
          A * direction_var * A.transpose();
}

} // namespace faster_lio
#endif