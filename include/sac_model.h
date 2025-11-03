#pragma once

#include <cfloat>
#include <ctime>
#include <climits>
#include <set>
#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <pcl/console/print.h>
#include <pcl/point_cloud.h>
#include <pcl/search/search.h>
#include <pcl/filters/extract_indices.h>


template<typename PointT, int kSampleSize, typename ModelT>
class SampleConsensusModel {
  public:
    using ModelType = ModelT;
    using PointCloud = typename pcl::PointCloud<PointT>;
    using PointCloudConstPtr = typename pcl::PointCloud<PointT>::ConstPtr;
    using PointCloudPtr = typename pcl::PointCloud<PointT>::Ptr;
    using SearchPtr = typename pcl::search::Search<PointT>::Ptr;
    
    using Ptr = boost::shared_ptr<SampleConsensusModel<PointT, kSampleSize, ModelT>>;
    using ConstPtr = boost::shared_ptr<const SampleConsensusModel<
            PointT, kSampleSize, ModelT>>;
    
    
  protected:
    explicit SampleConsensusModel(bool random = false)
            : input_(), indices_(), samples_radius_(0.), samples_radius_search_(),
              shuffled_indices_(), rng_alg_(), rng_dist_(new boost::uniform_int<>(0, std::numeric_limits<int>::max())),
              rng_gen_(), error_sqr_dists_() {
        rng_alg_.seed(random ? static_cast<unsigned>(std::time(nullptr)) : 12345u);//是否随机，如果为true，根据当前的时间设置随机种子，否则使用固定的种子
        //存储点云数据的对象
        all_inliers_.template reset(new PointCloud());
        //分配了一个随机数分布对象
        rng_gen_.reset(new boost::variate_generator<boost::mt19937 &, boost::uniform_int<>>(rng_alg_, *rng_dist_));
    }
  
  public:
    explicit SampleConsensusModel(const PointCloudPtr &cloud, bool random = false)
            : input_(), indices_(), samples_radius_(0.), samples_radius_search_(),
              shuffled_indices_(), rng_alg_(), rng_dist_(new boost::uniform_int<>(0, std::numeric_limits<int>::max())),
              rng_gen_(), error_sqr_dists_() {
        rng_alg_.seed(random ? static_cast<unsigned>(std::time(nullptr)) : 12345u);
        
        // Sets the input cloud and creates a vector of "fake" indices
        setInputCloud(cloud);
    
        all_inliers_.reset(new PointCloud());
    
        // Create a random number generator object
        rng_gen_.reset(new boost::variate_generator<boost::mt19937 &, boost::uniform_int<>>(rng_alg_, *rng_dist_));
    }
    
    virtual ~SampleConsensusModel() = default;
    
    virtual void GetSamples(int &iterations, std::vector<int> &samples) { //样本的获取方法
        // DLOG_ASSERT(indices_->size() >= kSampleSize);
        // get a second point which is different from the first
        samples.resize(kSampleSize);
        if (samples_radius_ < std::numeric_limits<double>::epsilon()) {
            drawIndexSample(samples);//如果 samples_radius_ 小于非常小的数值，则使用 drawIndexSample 方法，
                                     //否则使用 drawIndexSampleRadius 方法。
        } else {
            drawIndexSampleRadius(samples);
        }
    }
    
    virtual bool ComputeModelCoef(const std::vector<int> &samples,
                                  ModelT &model_coef) = 0;
    
    virtual void OptimizeModel(const std::vector<int> &inliers,
                               ModelT &optimized_coef) = 0;
    
    virtual void getDistancesToModel(const ModelT &model_coef,
                                     std::vector<double> &distances) = 0;
    
    virtual int CountWithinDistance(const ModelT &model_coef,
                                    float dist_thresh) = 0;
    
    virtual void SelectWithinDistance(const ModelT &model_coef,
                                      float dist_thresh, std::vector<int> &inliers) = 0;
    
    virtual int RemoveWithinDistance(const ModelT &model_coef,
                                     float dist_thresh, pcl::IndicesPtr inliers_idx,
                                     bool is_append_all_inliers) = 0;
    
    virtual void projectPoints(const std::vector<int> &inliers,
                               const ModelT &model_coef,
                               PointCloud &projected_points,
                               bool copy_data_fields) = 0;
    
    /** \brief Verify whether a subset of indices verifies a given set of
      * model coefficients. Pure virtual.
      *
      * \param[in] indices the data indices that need to be tested against the model
      * \param[in] model_coef the set of model coefficients
      * \param[in] threshold a maximum admissible distance threshold for
      * determining the inliers from the outliers
      */
    virtual bool doSamplesVerifyModel(const std::set<int> &indices,
                                      const ModelT &model_coef,
                                      double threshold) = 0;
    
    inline void setInputCloud(const PointCloudPtr &cloud) { //输入云设置方法？？？输入的点云数据？
        input_ = cloud;
//        if (!indices_)
        indices_.reset(new std::vector<int>());//重置 indices_，确保 indices_ 被初始化为一个空的整数向量
        if (indices_->empty()) {//如果 indices_ 为空，则进行初始化
            // Prepare a set of indices to be used (entire cloud)
            indices_->resize(cloud->points.size());//调整 indices_ 的大小，使其与点云数据中的点数相同
            for (size_t i = 0; i < cloud->points.size(); ++i)
                (*indices_)[i] = static_cast<int> (i);//indices_ 的每个元素都被设置为从 0 到 cloud->points.size() - 1 的整数
        }
        shuffled_indices_ = *indices_;
    }
    
    inline PointCloudPtr GetInputCloud() const { return input_; } //获取输入点云
    
    inline PointCloudPtr GetInliersCloud() const { return all_inliers_; }//获取所有内点云
    
    inline size_t NumPoints() const { return input_->size(); }//获取点云的点数
    
    inline void setIndices(const boost::shared_ptr<std::vector<int>> &indices) {
        indices_ = indices;
        shuffled_indices_ = *indices_;
    }
    
    inline void setIndices(const std::vector<int> &indices) {
        indices_.reset(new std::vector<int>(indices));
        shuffled_indices_ = indices;
    }
    
    inline boost::shared_ptr<std::vector<int>> getIndices() const { return indices_; }//获取索引
    
    inline const std::string &getClassName() const { return model_name_; }
    
    inline void SetSamplesMaxDist(const double &radius, SearchPtr search) { //设置最大样本的最大距离
        samples_radius_ = radius;
        samples_radius_search_ = search;
    }
    
    inline void GetSamplesMaxDist(double &radius) { radius = samples_radius_; } //获取样本的最大距离
    
//    friend class ProgressiveSampleConsensus<PointT>;
    
    /** \brief Compute the variance of the errors to the model.
      * \param[in] error_sqr_dists a vector holding the distances
      */
    inline double ComputeVariance(const std::vector<double> &error_sqr_dists) {
        std::vector<double> dists(error_sqr_dists);
        const size_t medIdx = dists.size() >> 1;//计算中位数索引 medIdx，使用右移操作符 >> 1 等效于除以2，
                                                //这是因为中位数的位置在排序后的向量的中间位置
        std::nth_element(dists.begin(), dists.begin() + medIdx, dists.end());//此算法只部分排序，将小于中位数的元素移到它的左边，
                                                                             //大于中位数的元素移到右边，但不会保证完全排序,这样比完全排序更高效
        double median_error_sqr = dists[medIdx];
        return 2.1981 * median_error_sqr; //用于将中位数误差平方值转换为方差估计值  这个又是用来干什么的呢？
    }
    
    /** \brief Compute the variance of the errors to the model from the internally
      * estimated vector of distances. The model must be computed first (or at least
      * SelectWithinDistance must be called).
      */
    inline double ComputeVariance() {
        LOG_ASSERT(!error_sqr_dists_.empty()) << "The variance of the Sample Consensus model distances cannot be estimated, as the model has not been computed yet. Please compute the model first or at least run SelectWithinDistance before continuing.";
        return ComputeVariance(error_sqr_dists_);
    }
  
  protected:
    
    inline void drawIndexSample(std::vector<int> &sample) {
        size_t index_size = shuffled_indices_.size();
        for (unsigned int i = 0; i < kSampleSize; ++i) {
            // The 1/(RAND_MAX+1.0) trick is when the random numbers are not uniformly distributed and for small modulo
            // elements, that does not matter (and nowadays, random number generators are good)
            //std::swap (shuffled_indices_[i], shuffled_indices_[i + (rand () % (index_size - i))]);
            std::swap(shuffled_indices_[i], shuffled_indices_[i + (rnd() % (index_size - i))]);// Fisher-Yates 洗牌算法
                                                                                               //rnd() 是一个返回随机数的函数，通过取模操作 % 来
                                                                                               //确保随机数在 0 到 index_size - i - 1 范围内。
        }
        std::copy(shuffled_indices_.begin(), shuffled_indices_.begin() + kSampleSize, sample.begin());
    }//将 shuffled_indices_ 的前 kSampleSize 个元素复制到 sample 向量中
    
    inline void drawIndexSampleRadiusWtf(std::vector<int> &sample) {
        size_t index_size = shuffled_indices_.size();
        
        std::swap(shuffled_indices_[0], shuffled_indices_[rnd() % index_size]);//为什么这里的随即搜索和前面的不一样呢？？？
        
        std::vector<int> indices;
        std::vector<float> sqr_dists;
        
        // If indices have been set when the search object was constructed,
        // radiusSearch() expects an index into the indices vector as its
        // first parameter. This can't be determined efficiently, so we use
        // the point instead of the index.
        // Returned indices are converted automatically.
        //使用 samples_radius_search_ 对象在给定的半径 samples_radius_ 内进行搜索，
        //中心点是 shuffled_indices_ 的第一个元素。搜索结果存储在 indices 和 sqr_dists 中。
        samples_radius_search_->radiusSearch(input_->at(shuffled_indices_[0]),
                                             samples_radius_, indices, sqr_dists);
    
        if (indices.size() < kSampleSize - 1) {
            for (unsigned int i = 1; i < kSampleSize; ++i)
                shuffled_indices_[i] = shuffled_indices_[0];//如果半径内的点不够kSampleSize - 1个，
                //就把shuffled_indices_前kSampleSize 个元素设置为相同的点，上面的中心点
        } else {//否则，从 indices 中随机抽取 kSampleSize - 1 个点，并将它们存储在 shuffled_indices_ 的前 kSampleSize 个元素中
            for (unsigned int i = 0; i < kSampleSize - 1; ++i)
                std::swap(indices[i], indices[i + (rnd() % (indices.size() - i))]);
            for (unsigned int i = 1; i < kSampleSize; ++i)
                shuffled_indices_[i] = indices[i - 1];
        }
        
        std::copy(shuffled_indices_.begin(), shuffled_indices_.begin() + kSampleSize, sample.begin());
    }
    
    inline void drawIndexSampleRadius(std::vector<int> &sample) { 
        size_t index_size = shuffled_indices_.size();
        
        std::swap(shuffled_indices_[0], shuffled_indices_[rnd() % index_size]);
        
        std::vector<int> indices;
        std::vector<float> sqr_dists;
        
        samples_radius_search_->radiusSearch(input_->at(shuffled_indices_[0]),
                                             samples_radius_, indices, sqr_dists);
        /* Attention: indices[0] == shuffled_indices_[0] */
        DLOG_ASSERT(indices[0] == shuffled_indices_[0]);

        if (indices.size() < kSampleSize) {
            sample.clear();
        } else {
            for (unsigned int i = 1; i < kSampleSize; ++i)
                std::swap(indices[i], indices[i + (rnd() % (indices.size() - i))]);
            for (unsigned int i = 0; i < kSampleSize; ++i) {
                sample[i] = indices[i];//为什么这里又是可以直接相等了
            }
        }
    }
    
    std::string model_name_;
    
    PointCloudPtr input_;
    
    PointCloudPtr all_inliers_;
    
    boost::shared_ptr<std::vector<int>> indices_;
    
    double samples_radius_;
    
    SearchPtr samples_radius_search_;
    
    std::vector<int> shuffled_indices_;
    
    boost::mt19937 rng_alg_;
    
    boost::shared_ptr<boost::uniform_int<>> rng_dist_;
    
    /** \brief Boost-based random number generator. */
    boost::shared_ptr<boost::variate_generator<boost::mt19937 &, boost::uniform_int<>>> rng_gen_;
    
    /** \brief A vector holding the distances to the computed model. Used internally. */
    std::vector<double> error_sqr_dists_;
    
    /** \brief Boost-based random number generator. */
    inline int rnd() { return rng_gen_->operator()(); }
  
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
