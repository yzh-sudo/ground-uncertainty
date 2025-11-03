//
// Created by xiang on 2021/10/8.
//

#ifndef FAST_LIO_OPTIONS_H
#define FAST_LIO_OPTIONS_H

namespace faster_lio::options {

/// fixed params
constexpr double INIT_TIME = 0.1;
constexpr double LASER_POINT_COV = 0.001;
constexpr int PUBFRAME_PERIOD = 20;
constexpr int NUM_MATCH_POINTS = 5;      // required matched points in current
constexpr int MIN_NUM_MATCH_POINTS = 3;  // minimum matched points in current

/// configurable params
extern int NUM_MAX_ITERATIONS;      // max iterations of ekf
extern float ESTI_PLANE_THRESHOLD;  // plane threshold
extern bool FLAG_EXIT;              // flag for exitting


///voxelmap++ related
// constexpr double G_m_s2 = 9.81;
// constexpr double PI_M = 3.14159265358; 
// constexpr int DIM_STATE = 18;  // Dimension of states (Let Dim(SO(3)) = 3)
// constexpr double INIT_COV = 1e-4;
// constexpr int NUM_MATCH_POINTS = 5;

// #define PI_M (3.14159265358)
// #define G_m_s2 (9.81)   // Gravaty const in GuangDong/China
// #define DIM_STATE (18)  // Dimension of states (Let Dim(SO(3)) = 3)
// old init
// #define INIT_COV (1e-4)
// #define NUM_MATCH_POINTS (5)

// #define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
// #define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]

}  // namespace faster_lio::options

#endif  // FAST_LIO_OPTIONS_H
