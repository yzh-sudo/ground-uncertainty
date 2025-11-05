# Ground-uncertainty
This paper presents an efficient LiDAR-intertial odometry with reduced Z-axis drift, incorporating uncertainty in ground points, which effectively alleviates drift in the Z-axis direction during mapping. A related odometry is also developed based on the UA-LIO manuscript’s introduction. The contributions of this work are as follows.
-  We propose a method to reduce Z-axis drift by leveraging the uncertainty of ground points to enhance the accuracy of map construction.
-  We propose a method for merging adjacent planes based on plane similarity. Neighboring planes with similar geometric characteristics are merged, and the uncertainty coefficients of the neighboring planes are used to update the parameters of the current plane. During residual computation, the plane information stored in the root node is directly utilized for point-to-plane matching, thereby improving both computational efficiency and overall mapping accuracy.
-  We incorporate the proposed method into Faster-LIO and validated its effectiveness across public datasets. Additionally, we test the robustness of our method on our campus datasets, demonstrating that it can effectively reduce both computational time and Z-axis drift in LIO.
Thanks to the work of UA-LIO, here are the relevant citation links for UA-LIO and Faster-LIO.
```bibtex
@article{wu2025ua,
  title={UA-LIO: An Uncertainty-Aware LiDAR-Inertial Odometry for Autonomous Driving in Urban Environments},20
  author={Wu, Qi and Chen, Xieyuanli and Xu, Xiangyu and Zhong, Xinliang and Qu, Xingwei and Xia, Songpengcheng and Liu, Guoqing and Liu, Liu and Yu, Wenxian and Pei, Ling},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={74},
  pages={1--12},
  year={2025},
  publisher={IEEE}
}
```
```bibtex
@article{bai2022faster,
  title={Faster-LIO: Lightweight tightly coupled LiDAR-inertial odometry using parallel sparse incremental voxels},
  author={Bai, Chunge and Xiao, Tao and Chen, Yajie and Wang, Haoqian and Zhang, Fang and Gao, Xiang},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={2},
  pages={4861--4868},
  year={2022},
  publisher={IEEE}
}
```
The experiment video can be found on [YouTube](https://youtu.be/jDTVptWhxXg), [Bilibili](https://www.bilibili.com/video/BV1mp1xBaEsp/?vd_source=75952e5ef3a34b2c0f143e1a6ee26441).

# Dependency
## 1. **Ubuntu and ROS**
   - Ubuntu 18.04 or Ubuntu 20.04 is recommended.

     
## 2. **GCC & G++ (only for Ubuntu 18.04)**
   - `gcc & g++ >= 9`

     
## 3. **ROS (melodic or noetic)**

   
## 4. **GLOG**
   ```bash
    sudo apt-get install -y libgoogle-glog-dev
   ```


## 5. **eigen**
   ```bash
    sudo apt-get install libeigen3-dev
   ```


## 6. **pcl**
   ```bash
    sudo apt-get install libpcl-dev
   ```


## 7. **livox_ros_driver**
   ```bash
    git clone https://github.com/Livox-SDK/Livox-SDK
    cd Livox-SDK
    mkdir build && cd build
    cmake ..
    make -j4
    sudo make install
   ```

# Dataset
## 1. UrbanNav Dataset
Download UrbanNav from https://github.com/weisongwen/UrbanNavDataset

## 2. NCLT Dataset
Download NCLT from http://robots.engin.umich.edu/nclt/

## 3. ULHK Dataset
Download ULHK from https://github.com/weisongwen/UrbanLoco

## 4. UTBM Dataset
Download ULHK from https://epan-utbm.github.io/utbm_robocar_dataset/

## 5. M2DGR Dataset
Download M2DGR from https://github.com/SJTU-ViSYS/M2DGR#dataset-sequences
