#ifndef FASTER_LIO_IMU_PROCESSING_H
#define FASTER_LIO_IMU_PROCESSING_H

#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <cmath>
#include <deque>
#include <fstream>

#include "common_lib.h"
#include "so3_math.h"
#include "use-ikfom.hpp"
#include "utils.h"

namespace faster_lio {

constexpr int MAX_INI_COUNT = 20;

bool time_list(const PointType &x, const PointType &y) { return (x.curvature < y.curvature); };

/// IMU Process and undistortion
class ImuProcess {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();
    ~ImuProcess();

    void Reset();
    void SetExtrinsic(const common::V3D &transl, const common::M3D &rot);
    void SetGyrCov(const common::V3D &scaler);
    void SetAccCov(const common::V3D &scaler);
    void SetGyrBiasCov(const common::V3D &b_g);
    void SetAccBiasCov(const common::V3D &b_a);
    void Process(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                 PointCloudType::Ptr pcl_un_);

    std::ofstream fout_imu_;
    Eigen::Matrix<double, 12, 12> Q_;
    common::V3D cov_acc_;
    common::V3D cov_gyr_;
    common::V3D cov_acc_scale_;
    common::V3D cov_gyr_scale_;
    common::V3D cov_bias_gyr_;
    common::V3D cov_bias_acc_;

   private:
    void IMUInit(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
    void UndistortPcl(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                      PointCloudType &pcl_out);

    PointCloudType::Ptr cur_pcl_un_;
    sensor_msgs::ImuConstPtr last_imu_;
    std::deque<sensor_msgs::ImuConstPtr> v_imu_;
    //imu的姿态
    std::vector<common::Pose6D> IMUpose_;
    std::vector<common::M3D> v_rot_pcl_;
    common::M3D Lidar_R_wrt_IMU_;
    common::V3D Lidar_T_wrt_IMU_;
    common::V3D mean_acc_; //mean_acc_ 和 mean_gyr_（分别表示加速度计和陀螺仪的平均值）
    common::V3D mean_gyr_;
    common::V3D angvel_last_;
    common::V3D acc_s_last_;
    double last_lidar_end_time_ = 0;
    // 用作一个计数器，它记录了处理的 IMU 测量次数，在IMU初始化时使用（N）
    int init_iter_num_ = 1; 
    bool b_first_frame_ = true;
    bool imu_need_init_ = true;
};

/**
 * @brief imu预处理类的构造函数，初始化了imu的噪声协方差矩阵，重力加速度，陀螺仪偏置，加速度偏置，加速度和陀螺仪的噪声协方差
*/
ImuProcess::ImuProcess() : b_first_frame_(true), imu_need_init_(true) {
    init_iter_num_ = 1;
    Q_ = process_noise_cov();
    cov_acc_ = common::V3D(0.1, 0.1, 0.1);
    cov_gyr_ = common::V3D(0.1, 0.1, 0.1);
    cov_bias_gyr_ = common::V3D(0.0001, 0.0001, 0.0001);
    cov_bias_acc_ = common::V3D(0.0001, 0.0001, 0.0001);
    mean_acc_ = common::V3D(0, 0, -1.0);
    mean_gyr_ = common::V3D(0, 0, 0);
    angvel_last_ = common::Zero3d;
    Lidar_T_wrt_IMU_ = common::Zero3d;
    Lidar_R_wrt_IMU_ = common::Eye3d;
    last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() {
    mean_acc_ = common::V3D(0, 0, -1.0);
    mean_gyr_ = common::V3D(0, 0, 0);
    angvel_last_ = common::Zero3d;
    imu_need_init_ = true;
    init_iter_num_ = 1;
    v_imu_.clear();
    IMUpose_.clear();
    last_imu_.reset(new sensor_msgs::Imu());
    cur_pcl_un_.reset(new PointCloudType());
}

void ImuProcess::SetExtrinsic(const common::V3D &transl, const common::M3D &rot) {
    Lidar_T_wrt_IMU_ = transl;
    Lidar_R_wrt_IMU_ = rot;
}

void ImuProcess::SetGyrCov(const common::V3D &scaler) { cov_gyr_scale_ = scaler; }

void ImuProcess::SetAccCov(const common::V3D &scaler) { cov_acc_scale_ = scaler; }

void ImuProcess::SetGyrBiasCov(const common::V3D &b_g) { cov_bias_gyr_ = b_g; }

void ImuProcess::SetAccBiasCov(const common::V3D &b_a) { cov_bias_acc_ = b_a; }

void ImuProcess::IMUInit(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                         int &N) {
    /** 1. initializing the gravity_, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurenments to unit gravity_ **/
    //1. 初始化重力,陀螺仪偏置,加速度和陀螺仪的协方差
    //2. 将加速度测量归一化为单位重力

    common::V3D cur_acc, cur_gyr;
    //第一帧imu测量
    if (b_first_frame_) {
        Reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu_.front()->linear_acceleration;
        const auto &gyr_acc = meas.imu_.front()->angular_velocity;
        
        //对于每个 IMU 测量，mean_acc_ 和 mean_gyr_（分别表示加速度计和陀螺仪的平均值）根据新的测量值更新：
        mean_acc_ << imu_acc.x, imu_acc.y, imu_acc.z;
        mean_gyr_ << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    }

    for (const auto &imu : meas.imu_) {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        mean_acc_ += (cur_acc - mean_acc_) / N;
        mean_gyr_ += (cur_gyr - mean_gyr_) / N;
        
        // 协方差的更新也使用了 N，以确保随着更多数据的累积，协方差计算能够反映所有测量数据的变化情况：
        // 协方差公式: 
        cov_acc_ =
            cov_acc_ * (N - 1.0) / N + (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);
            //其实算的是方差,可以看作是协方差的一种特殊情况，是用来衡量随机变量与其自身的变异程度。
            /*计算加速度的协方差的:cov_acc_` 是一个表示协方差的变量，它在每次有新的数据进来时都会被更新。
            协方差（Covariance）是统计学和概率论中用于衡量两个随机变量之间线性关系的一种度量。它描述了两个变量的联合变化程度，即它们相互关联的程度
            `N` 是一个表示数据的数量的变量。
            `(cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_)` 这部分代码计算的是当前加速度 `cur_acc` 与平均加速度 `mean_acc_` 的差的平方，也就是方差。
            这里使用的是 `cwiseProduct` 函数，它可以进行元素级别的乘法运算。
            `(N - 1.0) / N` 和 `(N - 1.0) / (N * N)` 是权重，它们决定了旧的协方差和新的方差在更新协方差时的权重。
            所以，`cov_acc_ = cov_acc_ * (N - 1.0) / N + (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);` 
            这行代码的作用是在线更新加速度的协方差。*/
        cov_gyr_ =
            cov_gyr_ * (N - 1.0) / N + (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) * (N - 1.0) / (N * N);
        //处理完每帧率IMU测量后，将N加1
        N++;
    }

    state_ikfom init_state = kf_state.get_x();
    init_state.grav = S2(-mean_acc_ / mean_acc_.norm() * common::G_m_s2); //根据平均加速度计算重力方向

    init_state.bg = mean_gyr_; //陀螺仪零偏
    init_state.offset_T_L_I = Lidar_T_wrt_IMU_; //
    init_state.offset_R_L_I = Lidar_R_wrt_IMU_;

    kf_state.change_x(init_state); //将初始化的结果作为卡尔曼滤波的初始状态,对应书中前20次imu静态测量获取bg的部分 P91

    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P(); //初始的协方差
    init_P.setIdentity();
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
    init_P(21, 21) = init_P(22, 22) = 0.00001;
    kf_state.change_P(init_P);
    last_imu_ = meas.imu_.back();
}

/**
 * @brief 点云去畸变模块
*/
void ImuProcess::UndistortPcl(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                              PointCloudType &pcl_out) {
    /*** add the imu_ of the last frame-tail to the of current frame-head ***/
    // 添加上一帧尾部的imu_到当前帧头部
    auto v_imu = meas.imu_;
    v_imu.push_front(last_imu_);
    const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
    const double &imu_end_time = v_imu.back()->header.stamp.toSec();
    const double &pcl_beg_time = meas.lidar_bag_time_;
    const double &pcl_end_time = meas.lidar_end_time_;

    /*** sort point clouds by offset time ***/
    // 按偏移时间对点云进行排序
    pcl_out = *(meas.lidar_);
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

    /*** Initialize IMU pose ***/
    // 初始化IMU姿态
    state_ikfom imu_state = kf_state.get_x();
    IMUpose_.clear();
    IMUpose_.push_back(common::set_pose6d(0.0, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                                          imu_state.rot.toRotationMatrix())); //imu的预积分姿态(全局坐标),gyr为机身坐标
    
    /*** forward propagation at each imu_ point ***/
    // 在每个imu_点上进行前向传播，计算出每个点的位置和姿态
    common::V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    common::M3D R_imu;

    double dt = 0;

    input_ikfom in;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) { //it_imu :deque双端队列
        auto &&head = *(it_imu); //&&引用
        auto &&tail = *(it_imu + 1);

        if (tail->header.stamp.toSec() < last_lidar_end_time_) {
            continue;
        }

        angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
            0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
            0.5 * (head->angular_velocity.z + tail->angular_velocity.z); //连续两帧imu的角速度均值

        acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
            0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
            0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);//连续两帧imu的加速度均值

        acc_avr = acc_avr * common::G_m_s2 / mean_acc_.norm();  // - state_inout.ba;

        if (head->header.stamp.toSec() < last_lidar_end_time_) {
            dt = tail->header.stamp.toSec() - last_lidar_end_time_;
        } else {
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        }

        in.acc = acc_avr;
        in.gyro = angvel_avr;
        Q_.block<3, 3>(0, 0).diagonal() = cov_gyr_; //Q:过程噪声协方差矩阵
        Q_.block<3, 3>(3, 3).diagonal() = cov_acc_;
        Q_.block<3, 3>(6, 6).diagonal() = cov_bias_gyr_;
        Q_.block<3, 3>(9, 9).diagonal() = cov_bias_acc_;
        kf_state.predict(dt, Q_, in); //key_:卡尔曼滤波的预测过程,其实质是根据前k-1时刻的观测和输入,加上K时刻的imu的测量值(即输入)更新状态量

        /* save the poses at each IMU measurements */
        // 保存每个IMU测量的姿态
        imu_state = kf_state.get_x(); //根据预测值
        angvel_last_ = angvel_avr - imu_state.bg;
        acc_s_last_ = imu_state.rot * (acc_avr - imu_state.ba); //全局坐标系
        for (int i = 0; i < 3; i++) {
            acc_s_last_[i] += imu_state.grav[i];
        }

        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
        IMUpose_.emplace_back(common::set_pose6d(offs_t, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                                                 imu_state.rot.toRotationMatrix())); //得到每个IMU测量的姿态,用于点云的去畸变
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    // 计算帧末的位置和姿态预测
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    kf_state.predict(dt, Q_, in); //key_:卡尔曼滤波的预测过程 计算帧末的位置和姿态预测

    imu_state = kf_state.get_x(); //取出更新后的卡尔曼滤波姿态
    last_imu_ = meas.imu_.back();
    last_lidar_end_time_ = pcl_end_time;

    /*** undistort each lidar point (backward propagation) ***/
    // 对每个激光点进行去畸变（反向传播）
    // key_:反向传播
    if (pcl_out.points.empty()) {
        return;
    }
    auto it_pcl = pcl_out.points.end() - 1;
    for (auto it_kp = IMUpose_.end() - 1; it_kp != IMUpose_.begin(); it_kp--) {
        auto head = it_kp - 1;
        auto tail = it_kp;
        R_imu = common::MatFromArray(head->rot);
        vel_imu = common::VecFromArray(head->vel);
        pos_imu = common::VecFromArray(head->pos);
        acc_imu = common::VecFromArray(tail->acc);
        angvel_avr = common::VecFromArray(tail->gyr);
        // 从IMU测量中提取姿态（旋转矩阵 R_imu）、速度（vel_imu）、位置（pos_imu）、加速度（acc_imu）、角速度（angvel_avr）等信息

        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
            dt = it_pcl->curvature / double(1000) - head->offset_time; //dt = it_pcl->curvature / double(1000)时间,单位为ms

            /* Transform to the 'end' frame, using only the rotation
             * Note: Compensation direction is INVERSE of Frame's moving direction
             * So if we want to compensate a point at timestamp-i to the frame-e
             * p_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
            /* 将坐标系转换到 'end' 帧，仅使用旋转信息
             * 注意：补偿的方向与坐标系移动的方向相反
             * 因此，如果我们要将时间戳 i 处的点补偿到 'end' 帧
             * p_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)，其中 T_ei 在全局坐标系中表示 */

            common::M3D R_i(R_imu * Exp(angvel_avr, dt)); //(泊松方程)R_i` 是当前IMU姿态在时间 `dt` 后的旋转矩阵，
            // 它是通过当前的IMU旋转矩阵 `R_imu` 和平均角速度 `angvel_avr` 在时间 `dt` 内的旋转得到的。

            common::V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z); 
            common::V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);//P_i` 是当前点云数据的位置，`T_ei` 是在时间 `dt` 内的平移向量，
            // 它是通过当前的IMU位置 `pos_imu`，速度 `vel_imu`，加速度 `acc_imu` 和IMU状态的位置 `imu_state.pos` 计算得到的。

            common::V3D p_compensate =
                imu_state.offset_R_L_I.conjugate() * //conjugate()函数是求共轭
                (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) -
                 imu_state.offset_T_L_I);  // not accurate! 接着，`p_compensate` 是补偿后的点云数据的位置。
                //  它是通过IMU状态的旋转 `imu_state.rot` 和偏移 `imu_state.offset_R_L_I`，以及 `R_i`，`P_i`，`T_ei` 和 `imu_state.offset_T_L_I` 进行坐标变换得到的。
                //  去畸变后的点仍然在雷达坐标系
            // save Undistorted points and their rotation
            // 储存去畸变后的点云和它们的旋转
            it_pcl->x = p_compensate(0);
            it_pcl->y = p_compensate(1);
            it_pcl->z = p_compensate(2);

            if (it_pcl == pcl_out.points.begin()) {
                break; //从后往前遍历修改
            }
        }
    }
}


/**
 * @brief //TAG:IMU预处理
 * @param meas MeasureGroup类，包含了 时间同步以后的 imu_和lidar_的数据
 * @param kf_state esekf<state_ikfom, 12, input_ikfom>类型的变量,实际上是kf_,传入的是整个卡尔曼滤波器
 * @param cur_pcl_un_ PointCloudType::Ptr类型的指针,实际上是scan_undistort
*/
void ImuProcess::Process(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                         PointCloudType::Ptr cur_pcl_un_) {
    if (meas.imu_.empty()) {
        return;
    }

    ROS_ASSERT(meas.lidar_ != nullptr);

    //对第一帧点云的处理
    if (imu_need_init_) {
        /// The very first lidar frame
        IMUInit(meas, kf_state, init_iter_num_);//传入时间同步以后的测量值,卡尔曼滤波器,和 IMU 测量次数

        imu_need_init_ = true; //在满足20次测量之前，imu需要初始化

        last_imu_ = meas.imu_.back();

        state_ikfom imu_state = kf_state.get_x();

        //init_iter_num_是imu的测量，MAX_INI_COUNT=20，含义是初始化imu需要的测量次数
        if (init_iter_num_ > MAX_INI_COUNT) {
            cov_acc_ *= pow(common::G_m_s2 / mean_acc_.norm(), 2); //(Gra/mea_acc)**2
            imu_need_init_ = false; //imu接收20次测量数据后不需要再初始化

            cov_acc_ = cov_acc_scale_;//在初始化类时存在的极小值
            cov_gyr_ = cov_gyr_scale_;
            LOG(INFO) << "IMU Initial Done";
            fout_imu_.open(common::DEBUG_FILE_DIR("imu_.txt"), std::ios::out);
        }

        return;
    }
    // 执行点云去畸变的步骤
    Timer::Evaluate([&, this]() { UndistortPcl(meas, kf_state, *cur_pcl_un_); }, "Undistort Pcl");
}
}  // namespace faster_lio

#endif
