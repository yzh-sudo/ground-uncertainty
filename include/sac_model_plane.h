#pragma once

#include <type_traits>
#include <omp.h>
#include <glog/logging.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/concatenate.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/io.h>
#include "sac_model.h"
#include "sac_model_plane.h"
#include "reducible_vector.h"
//#include "timer.h"


#define Point2PlaneDist(p, plane_coef) fabsf(Point2PlaneDistSigned((p), (plane_coef))) //定义了两个宏，计算绝对值

#define Point2PlaneDistSigned(p, plane_coef) ((plane_coef)[0] * (p).x + (plane_coef)[1] * (p).y + \
    (plane_coef)[2] * (p).z + (plane_coef)[3])//计算两点的距离


class PlaneWithCentroid {
  public:
    // ax + by + cz + d = 0
    Eigen::Vector4f coef;//平面参数
    // centroid from the least square fitting.
    // note that d can be calculated from centroid, but not vice versa.
    Eigen::Vector3f centroid;//质心
};


template<class PointT, int kSampleSize = 6, class PlaneModelT = PlaneWithCentroid>
class SampleConsensusModelPlane : public SampleConsensusModel<
        PointT, kSampleSize, PlaneModelT> {
  public:
    
    static_assert(kSampleSize >= 3, "cannot fit a plane with less than 3 points!"); //编译的时候要确保至少需要3个点
    
    static constexpr bool kIsWithPca = kSampleSize > 3; //点的个数大于3时，就用PCA
    static constexpr bool kIsWithCentroid = std::is_same<PlaneModelT, PlaneWithCentroid>();

    using PointCloud = typename SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::PointCloud;
    using PointCloudPtr = typename SampleConsensusModel<
            PointT, kSampleSize, PlaneModelT>::PointCloudPtr;
    using PointCloudConstPtr = typename SampleConsensusModel<
            PointT, kSampleSize, PlaneModelT>::PointCloudConstPtr;
    using Ptr = std::shared_ptr<SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>>;
    
    
    explicit SampleConsensusModelPlane(const PointCloudPtr &cloud,
                                       bool random = false, int num_parallel = 4); //并行计算时的线程
    
    virtual ~SampleConsensusModelPlane() = default;
     
    bool ComputeModelCoef(const std::vector<int> &samples,
                          PlaneModelT &model_coef) override;
    
    bool FitPlaneFrom3points(const std::vector<int> &samples,
                             PlaneModelT &plane_coef);
    bool FitPlaneFromNPoints(const std::vector<int> &samples,
                             PlaneModelT &plane_coef);
    
    void getDistancesToModel(const PlaneModelT &plane,
                             std::vector<double> &distances) override;
    
    int CountWithinDistance(const PlaneModelT &plane,
                            float dist_thresh) override;
    
    void SelectWithinDistance(const PlaneModelT &plane,
                              float dist_thresh, std::vector<int> &inliers) override;
    
    int RemoveWithinDistance(const PlaneModelT &plane, float dist_thresh,
                             pcl::IndicesPtr inliers_idx, bool is_append_all_inliers) override;
    
    void OptimizeModel(const std::vector<int> &inliers,
                       PlaneModelT &optimized_plane) override;
    
    void projectPoints(const std::vector<int> &inliers,
                       const PlaneModelT &plane,
                       PointCloud &projected_points,
                       bool copy_data_fields) override;
    
    bool doSamplesVerifyModel(const std::set<int> &indices,
                              const PlaneModelT &plane,
                              double threshold) override;
    
  private:
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::model_name_;
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::input_;
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::all_inliers_;
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::indices_;
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::error_sqr_dists_;
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::samples_radius_search_;
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::samples_radius_;
    int num_threads_;
};


template<class PointT, int kSampleSize, class PlaneModelT>
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::SampleConsensusModelPlane(
        const PointCloudPtr &cloud,
        bool random /* = false*/,
        int num_parallel/* = 4*/)
            : SampleConsensusModel<PointT, kSampleSize, PlaneModelT>(cloud, random)
            , num_threads_(num_parallel) {
    model_name_ = "SampleConsensusModelPlane";
}


template<class PointT, int kSampleSize, class PlaneModelT>
bool
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::ComputeModelCoef(
        const std::vector<int> &samples, PlaneModelT &model_coef) {
    DLOG_ASSERT(samples.size() == kSampleSize); //为什么会有这个断言？？？ 确保计算模型参数时使用正确的数量样本点？
//    if constexpr(kSampleSize > 3) {  // this is a c++17 extension  //奇怪的是这个kSampleSize如果是自己规定的话，为什么还需要判断3个还是多个
    if (kIsWithPca) { //看看使用哪个平面拟合的方式
        return FitPlaneFromNPoints(samples, model_coef);
    }
    return FitPlaneFrom3points(samples, model_coef);
}

template<class PointT, int kSampleSize, class PlaneModelT>
bool
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::FitPlaneFrom3points(//3个点的平面拟合
        const std::vector<int> &samples, //samples里面存储的是点云中用于拟合平面的点的索引
        PlaneModelT &plane) {
    pcl::Array4fMap p0 = input_->points[samples[0]].getArray4fMap();//存储点的坐标
    pcl::Array4fMap p1 = input_->points[samples[1]].getArray4fMap();
    pcl::Array4fMap p2 = input_->points[samples[2]].getArray4fMap();
    
    Eigen::Array4f p1p0 = p1 - p0;
    Eigen::Array4f p2p0 = p2 - p0;
    
    // Avoid some crashes by checking for collinearity here
    // FIXME divided by zero
    Eigen::Array4f dy1dy2 = p1p0 / p2p0;
    if (dy1dy2[0] == dy1dy2[1] && dy1dy2[2] == dy1dy2[1]) { //判断是否共线
        // check for collinearity
        return false;
    }
    
    // Compute the plane coefficients from the 3 given points in a straightforward manner
    // calculate the plane normal n = (p2-p1) x (p3-p1) = cross (p2-p1, p3-p1)
    plane.coef[0] = p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1]; 
    plane.coef[1] = p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2];
    plane.coef[2] = p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0];
    plane.coef[3] = 0;
    plane.coef.normalize();//计算平面参数，并归一化
    plane.coef[3] = -1 * (plane.coef.dot(p0.matrix()));
    if (kIsWithCentroid) {
        // For efficiency, the centroid is calculated later
        // in OptimizedModel().
        //那你写在这干什么？？？？
    }
    return true;//疑惑，三点和多点为什么分开？chat说三点分开节约时间一点
}

template<class PointT, int kSampleSize, class PlaneModelT>
bool
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::FitPlaneFromNPoints( //3个点以上的
        const std::vector<int> &samples,
        PlaneModelT &plane) {
    
   if (samples.size() < 3) { //3
        return false;
    }
    else {
        std::vector<PointT> cur_points;  // 创建一个局部副本用于存储点
        for (int idx : samples) {
            cur_points.push_back(input_->points[idx]);  // 根据索引从原点云中提取点
        }
        Eigen::Matrix<float, 4, 1> pca_result;
        do{  
            Eigen::Matrix<float, 3, 1> normvec;
            Eigen::Matrix<float, Eigen::Dynamic, 3> A(cur_points.size(), 3);
            Eigen::Matrix<float, Eigen::Dynamic, 1> b(cur_points.size(), 1);

            A.setZero();
            for (size_t i = 0; i < cur_points.size(); ++i)
            { 
                A(i, 0) = static_cast<float>(cur_points[i].x);
                A(i, 1) = static_cast<float>(cur_points[i].y);
            }

            Eigen::Matrix<float, 3, 3> covarianceMatrix = (A.transpose() * A) / static_cast<float>(cur_points.size()); // 计算数据的协方差
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 3, 3>> eigenSolver(covarianceMatrix); // 进行特征值分解
            normvec = eigenSolver.eigenvectors().col(0).template cast<float>(); // 拟合平面的法向量，对应最小特征值的那个向量


            Eigen::Matrix<float, 3, 1> mean = A.colwise().mean(); // 计算所有点的均值向量

            float pa = normvec[0];
            float pb = normvec[1];
            float pc = normvec[2];
            float norm = sqrt(pa * pa + pb * pb + pc * pc);
            pa /= norm; pb /= norm; pc /= norm;
            float pd = (float) 1 / norm; //归一化

            float d = -normvec.dot(mean); // 计算参数d

            //获取平面的参数
            pca_result.template head<3>() = normvec;
            pca_result(3) = d;

            // midnorm.template head<3>() = normvec;
            // midnorm[0] = normvec[0];
            // midnorm[1] = normvec[1];
            // midnorm[2] = normvec[2];
            // midnorm[3] = d;
            float n = pca_result.norm();

            Eigen::Matrix<float, 4, 1> midnorm;
            midnorm(0) = normvec(0) / n;
            midnorm(1) = normvec(1) / n; // 归一化，方便下面进行点到平面的距离计算
            midnorm(2) = normvec(2) / n;
            midnorm(3) = d / n; // 这一段是在进行平面估计，然后计算估计出来的平面的法向量

            std::vector<float> distances(cur_points.size());
            for (size_t i = 0; i < cur_points.size(); ++i) {
                Eigen::Matrix<float, 4, 1> temp = cur_points[i].getVector4fMap().template cast<float>(); // 将坐标设成一个四维数组，并进行类型转换
                temp[3] = 1.0; // 将最后一个数设成1
                float distance = fabs(midnorm.dot(temp)); // 计算每个点到平面的距离
                distances[i] = distance; // 将距离存储到数组中
            }

            std::vector<float> absDeviations(distances.size());
            size_t size = distances.size(); // 点到平面距离的数量就是点的数量
            std::sort(distances.begin(), distances.end()); // 进行排序

            float median = 0.0;
            float mad = 0.0;
            if (size % 2 == 0) { // 用来求中位数
                median = 0.5 * (distances[size / 2] + distances[size / 2 + 1]);
            } else {
                median = distances[(size + 1) / 2];
            }

            for (size_t i = 0; i < distances.size(); ++i) {
                absDeviations[i] = std::abs(distances[i] - median);
            }
            std::sort(absDeviations.begin(), absDeviations.end());

            if (size % 2 == 0) {
                mad = 0.5 * (absDeviations[size / 2] + absDeviations[size / 2 + 1]) * 1.4826;
            } else {
                mad = absDeviations[(size + 1) / 2] * 1.4826;
            }

            //计算鲁棒值
            std::vector<float> robustZScores(distances.size());
            for (size_t i = 0; i < distances.size(); ++i) {
                robustZScores[i] = (distances[i] - median) / mad;
            }

            std::vector<bool> outliers(robustZScores.size()); // 判断离群点
            for (size_t i = 0; i < robustZScores.size(); ++i) {
                outliers[i] = std::abs(robustZScores[i]) >= 2.5; // 计算robustZScores的绝对值，再跟阈值进行比较，outliers 是 bool 类型，大于或者等于阈值就是 true
            }
 
            std::vector<PointT> inliers;
            for (size_t i = 0; i < cur_points.size(); ++i) {
                if (!outliers[i]) {
                    inliers.push_back(cur_points[i]); // 将未剔除的点加入到 inliers 中
                }
            }

            if (inliers.size() == cur_points.size()) { // 发现所有点都是局内点
                plane.coef[0] = pa;
                plane.coef[1] = pb;
                plane.coef[2] = pc;
                plane.coef[3] = pd;
                return true;
            }

            cur_points = inliers;
        } while(cur_points.size() >= 3); 
        // if (cur_points.size() < options::MIN_NUM_MATCH_POINTS){return false;}    
    }   
    return false;
}


template<class PointT, int kSampleSize, class PlaneModelT>
void
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::getDistancesToModel(
        const PlaneModelT &plane, std::vector<double> &distances) {
    distances.resize(indices_->size());//预分配距离向量的大小
    
    // Iterate through the 3d points and calculate the distances from them to the plane
    for (size_t i = 0; i < indices_->size(); ++i) {
        // Calculate the distance from the point to the plane normal as the dot product
        // D = (P-A).N/|N|
        /*distances[i] = fabs (model_coef[0] * input_->points[(*indices_)[i]].x +
                             model_coef[1] * input_->points[(*indices_)[i]].y +
                             model_coef[2] * input_->points[(*indices_)[i]].z +
                             model_coef[3]);*/
        Eigen::Vector4f pt(input_->points[(*indices_)[i]].x,
                           input_->points[(*indices_)[i]].y,
                           input_->points[(*indices_)[i]].z,
                           1);
        distances[i] = fabs(plane.coef.dot(pt));//计算点（和上面的samples不同，这里是需要计算距离的所有点）到平面的距离
    }
}


template<class PointT, int kSampleSize, class PlaneModelT>
int
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::CountWithinDistance(
        const PlaneModelT &plane, const float dist_thresh) {
    
    const int n_pts = (int) indices_->size();//计算点的个数
    int nr_p = 0;
#if defined(__GNUC__) && (__GNUC__ >= 9)
/* variables with const qualifier will not be auto pre-determined
 * as 'shared' in omp in higher gcc version */
#pragma omp parallel for num_threads(num_threads_) default(none) \
            shared(n_pts, plane, dist_thresh) reduction(+: nr_p)
#else
#pragma omp parallel for num_threads(num_threads_) default(none) shared(plane) reduction(+: nr_p)
#endif
    for (int i = 0; i < n_pts; ++i) {
        float dist = Point2PlaneDist(input_->points[(*indices_)[i]], plane.coef);
        if (dist < dist_thresh) {
            nr_p++;
        }
    }
    return nr_p; //计算满足距离阈值的点的个数
}


template<class PointT, int kSampleSize, class PlaneModelT>
void
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::SelectWithinDistance(
        const PlaneModelT &plane, const float dist_thresh,
        std::vector<int> &inliers) {
    
    const int n_pts = indices_->size();
    
    inliers.clear();
    error_sqr_dists_.clear();
    inliers.reserve(n_pts);//预分配大小
    error_sqr_dists_.reserve(n_pts);
    
    const int n_each_thread = n_pts / num_threads_ + 1; //计算每个线程需要处理的点数
    // The ReducibleVector is 5.7x faster than using omp critical clause.
    OmpReducibleVector<int> reduce_inliers(n_each_thread, &inliers);
    OmpReducibleVector<double> reduce_sqr_dists(n_each_thread, &error_sqr_dists_);
    
#if defined(__GNUC__) && (__GNUC__ >= 9)
/* variables with const qualifier will not be auto pre-determined
 * as shared in omp in higher gcc version */
#pragma omp parallel for num_threads(num_threads_) default(none) shared(n_pts, plane, dist_thresh) \
            reduction(merge_reducible_vec_i: reduce_inliers) \
            reduction(merge_reducible_vec_d: reduce_sqr_dists)
#else
#pragma omp parallel for num_threads(num_threads_) default(none) shared(plane) \
            reduction(merge_reducible_vec_i: reduce_inliers) \
            reduction(merge_reducible_vec_d: reduce_sqr_dists)
#endif
    for (int i = 0; i < n_pts; ++i) {
        float dist = Point2PlaneDist(input_->points[(*indices_)[i]], plane.coef);
        if (dist < dist_thresh) {
            reduce_inliers.FastPushBack((*indices_)[i]); //将小于距离阈值的点储存起来，并存储其距离
            reduce_sqr_dists.FastPushBack(static_cast<double>(dist));
        }
    }
}


template<class PointT, int kSampleSize, class PlaneModelT>
int
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::RemoveWithinDistance(
        const PlaneModelT &plane, const float dist_thresh,
        pcl::IndicesPtr inliers_idxs, bool is_append_all_inliers) {
    DLOG_ASSERT(indices_->size() == input_->size());
    const int n_pts = (int) indices_->size();
    if (inliers_idxs) {
        inliers_idxs->clear();//非空就清空
    } else {
        inliers_idxs = pcl::IndicesPtr(new std::vector<int>); //空创建新的
    }
    const float midlier_dist_thresh = 4 * dist_thresh;
    pcl::IndicesPtr midliers_idxs(new std::vector<int>);
    
    inliers_idxs->reserve(n_pts);//预分配内存
    midliers_idxs->reserve(n_pts);
    
    const int n_each_thread = n_pts / num_threads_ + 1; //计算每个线程要处理的点数
    // FIXME(xingyuuchen): Mixed usage of shared_ptr and raw ptr.
    OmpReducibleVector<int> reduce_inliers(n_each_thread, inliers_idxs.get());//累积满足距离条件的点的索引？？理解：就是将满足内点的索引放入这个里面
    OmpReducibleVector<int> reduce_midliers(n_each_thread, midliers_idxs.get());//用于并行累积在中间距离阈值内的点的索引??
    
#if defined(__GNUC__) && (__GNUC__ >= 9)
/* variables with const qualifier will not be auto pre-determined
 * as shared in omp in higher gcc version */
#pragma omp parallel for num_threads(num_threads_) default(none) \
        shared(n_pts, plane, dist_thresh, midlier_dist_thresh) \
        reduction(merge_reducible_vec_i: reduce_inliers, reduce_midliers)
#else
#pragma omp parallel for num_threads(num_threads_) default(none) shared(plane) \
        reduction(merge_reducible_vec_i: reduce_inliers, reduce_midliers)
#endif
    for (int i = 0; i < n_pts; ++i) {
        int idx = (*indices_)[i];
        float dist = Point2PlaneDist(input_->points[idx], plane.coef);
        if (dist < dist_thresh) {
            reduce_inliers.FastPushBack(idx); //压入符合要求的索引   搞不懂为什么这边又要判断一次内点呢？
        } else if (dist < midlier_dist_thresh) {
            reduce_midliers.FastPushBack(idx);//压入符合要求的索引（夹心层）
        }
    }
    
    int n_remove = (int) inliers_idxs->size(); //内点移除？ 设置为内点的大小 为什么是设置的内点数目的大小，去除的不应该是中间点吗?这个size有啥用
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(input_);//input_ 是一个包含点云数据的对象，可以是 pcl::PointCloud<PointT>::Ptr 类型
    extract.setIndices(inliers_idxs);//inliers_idxs 设置为需要提取或移除的点的索引列表
    if (is_append_all_inliers) {
        PointCloud inliers_cloud;
        extract.setNegative(false); //提取点：通过 setNegative(false) 配置，表示提取索引列表中指定的点
        extract.filter(inliers_cloud);//移除点：通过 setNegative(true) 配置，表示移除索引列表中指定的点
        *all_inliers_ += inliers_cloud; //将内点的个数加到全部中去
    }
    if (!midliers_idxs->empty()) {
        n_remove += midliers_idxs->size();//去除点的地方加上夹心层的点
        midliers_idxs->insert(midliers_idxs->end(),
             inliers_idxs->begin(), inliers_idxs->end());//将inliers_idxs中所有内点的索引添加到midliers_idxs的末尾
        extract.setIndices(midliers_idxs);//把提取或者删除的对象换成midliers_idxs索引
    }
    extract.setNegative(true);//删除内点和夹心层的点
    extract.filter(*input_);//将结果保存在input中  这里把内点也删除了？看看论文？回答：没有删除内点，只是单纯的删除了半内点
                            //将删除半内点之后的点云存储在input_中
    
    if (samples_radius_search_) {
        DLOG_ASSERT(samples_radius_ > 0);
        samples_radius_search_->setInputCloud(input_); //这一段不知道在干什么？？？？
    }
    
    SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::setInputCloud(input_);
    return n_remove;//返回了移除点的数量
}


template<class PointT, int kSampleSize, class PlaneModelT>
void
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::OptimizeModel(
        const std::vector<int> &inliers, PlaneModelT &optimized_plane) {
    // Need more than the minimum sample size to make a difference
    // DLOG_ASSERT(inliers.size() > kSampleSize) << " " << inliers.size();//如果条件不满足，错误信息中将包含 inliers 集合的大小
    
    PlaneModelT plane_parameters;
    
    // Use Least-Squares to fit the plane through all the given
    // sample points and find out its coefficients
    EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;//协方差矩阵
    Eigen::Vector4f xyz_centroid; //质心
    
    // TODO(xingyuuchen): OpenMp speed up
    computeMeanAndCovarianceMatrix(*input_, inliers,
                                   covariance_matrix, xyz_centroid); //计算给定点的协方差矩阵和质心
    
    // Compute the model coefficients
    EIGEN_ALIGN16 Eigen::Vector3f::Scalar eigen_value;
    EIGEN_ALIGN16 Eigen::Vector3f eigen_vector;
    pcl::eigen33(covariance_matrix, eigen_value, eigen_vector);//用协方差矩阵计算最小特征值和对应的特征向量（相当于法向量）
    
    // Hessian form (D = nc . p_plane (centroid here) + p)
    optimized_plane.coef[0] = eigen_vector[0];
    optimized_plane.coef[1] = eigen_vector[1];
    optimized_plane.coef[2] = eigen_vector[2];
    optimized_plane.coef[3] = 0;
    // noting the eigen_vector here is already normalized
    optimized_plane.coef[3] = -1 * (optimized_plane.coef.dot(xyz_centroid));
    if (kIsWithCentroid) {
        optimized_plane.centroid = xyz_centroid.template head<3>();//将计算得到的质心复制给optimized_plane.centroid
    }
}  //这里又在计算内点重新拟合的平面？？？


template<class PointT, int kSampleSize, class PlaneModelT>
void
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::projectPoints( //把点投影到平面上
        const std::vector<int> &inliers, const PlaneModelT &plane,
        PointCloud &projected_points,
        bool copy_data_fields) {
    
    projected_points.header = input_->header;
    projected_points.is_dense = input_->is_dense;
    
    Eigen::Vector4f mc(plane.coef[0], plane.coef[1], plane.coef[2], 0);
    
    // normalize the vector perpendicular to the plane...
    mc.normalize();
    // ... and store the resulting normal as a local copy of the model coefficients
    Eigen::Vector4f tmp_mc = plane.coef;
    tmp_mc[0] = mc[0];
    tmp_mc[1] = mc[1];
    tmp_mc[2] = mc[2];
    
    // Copy all the data fields from the input cloud to the projected one?
    if (copy_data_fields) {
        // Allocate enough space and copy the basics
        projected_points.points.resize(input_->points.size());
        projected_points.width = input_->width;
        projected_points.height = input_->height;
        
        typedef typename pcl::traits::fieldList<PointT>::type FieldList;
        // Iterate over each point
        for (size_t i = 0; i < input_->points.size(); ++i)
            // Iterate over each dimension
            pcl::for_each_type<FieldList>(pcl::NdConcatenateFunctor<PointT, PointT>(
                    input_->points[i], projected_points.points[i]));
        
        // Iterate through the 3d points and calculate the distances from them to the plane
        for (size_t i = 0; i < inliers.size(); ++i) {
            // Calculate the distance from the point to the plane
            Eigen::Vector4f p(input_->points[inliers[i]].x,
                              input_->points[inliers[i]].y,
                              input_->points[inliers[i]].z,
                              1);
            // use normalized coefficients to calculate the scalar projection
            float distance_to_plane = tmp_mc.dot(p);
            
            pcl::Vector4fMap pp = projected_points.points[inliers[i]].getVector4fMap();
            pp.matrix() = p - mc * distance_to_plane;        // mc[3] = 0, therefore the 3rd coordinate is safe
        }
    } else {
        // Allocate enough space and copy the basics
        projected_points.points.resize(inliers.size());
        projected_points.width = static_cast<uint32_t> (inliers.size());
        projected_points.height = 1;
        
        typedef typename pcl::traits::fieldList<PointT>::type FieldList;
        // Iterate over each point
        for (size_t i = 0; i < inliers.size(); ++i)
            // Iterate over each dimension
            pcl::for_each_type<FieldList>(pcl::NdConcatenateFunctor<PointT, PointT>(
                    input_->points[inliers[i]], projected_points.points[i]));
        
        // Iterate through the 3d points and calculate the distances from them to the plane
        for (size_t i = 0; i < inliers.size(); ++i) {
            // Calculate the distance from the point to the plane
            Eigen::Vector4f p(input_->points[inliers[i]].x,
                              input_->points[inliers[i]].y,
                              input_->points[inliers[i]].z,
                              1);
            // use normalized coefficients to calculate the scalar projection
            float distance_to_plane = tmp_mc.dot(p);
            
            pcl::Vector4fMap pp = projected_points.points[i].getVector4fMap();
            // mc[3] = 0, therefore the 3rd coordinate is safe
            pp.matrix() = p - mc * distance_to_plane;
        }
    }
}


template<class PointT, int kSampleSize, class PlaneModelT>
bool
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::doSamplesVerifyModel(
        const std::set<int> &indices, const PlaneModelT &plane,
        const double threshold) {
    
    for (const int &idx : indices) {
        Eigen::Vector4f pt(input_->points[idx].x,
                           input_->points[idx].y,
                           input_->points[idx].z,
                           1);
        if (fabs(plane.coef.dot(pt)) > threshold) {
            return false;
        } //判断模型的拟合程度
    }
    return true;
}


