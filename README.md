# ground-uncertainty
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
The experiment video can be found on [Bilibili](https://www.bilibili.com/video/BV1mp1xBaEsp/?vd_source=75952e5ef3a34b2c0f143e1a6ee26441).

