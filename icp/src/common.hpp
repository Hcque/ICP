#pragma once

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <pcl/common/common.h>


using Vector3f = Eigen::Vector3f;
using Vector3d = Eigen::Vector3d;
using Matrix3f = Eigen::Matrix3f;
using Matrix4f = Eigen::Matrix4f;
using Matrix4d = Eigen::Matrix4d;
using Matrix3d = Eigen::Matrix3d;

using Point3f = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<Point3f>;
using PointCloudTPtr = pcl::PointCloud<Point3f>::Ptr;

inline Vector3f ToVector3f(const Point3f& p) {
    return { p.x, p.y, p.z };
}

inline Vector3f ToPoint3f(const Vector3f& v) {
    return { v.x(), v.y(), v.z() };
}

inline Vector3f operator - (const Point3f& p1, const Point3f& p2) {
    return { p1.x - p2.x, p1.y - p2.y, p1.z - p2.z };
}

inline Point3f operator + (const Point3f& p, const Vector3f& v) {
    return { p.x + v.x(), p.y + v.y(), p.z + v.z() };
}

inline Point3f operator - (const Point3f& p, const Vector3f& v) {
    return { p.x - v.x(), p.y - v.y(), p.z - v.z() };
}

inline float DistSq(const Point3f& p1, const Point3f& p2) {
    return (p1 - p2).squaredNorm();
}

inline float Dist(const Point3f& p1, const Point3f& p2) {
    return (p1 - p2).norm();
}
inline float Dist(const Vector3d& p1, const Vector3d& p2) {
    // return (ToPoint3f(p1) - ToPoint3f(p2)).norm();
    using namespace std;
    return pow(p1[0] - p2[0],2) + pow(p1[1] - p2[1],2) + pow(p1[2]-p2[2], 2);
}

void cal_mean(Eigen::MatrixXd& src, Vector3d& mean_src)
{
    mean_src.setZero();
    for (int i = 0 ; i < src.rows(); i ++ ) 
       mean_src += src.block<1,3>(i,0).transpose();
    mean_src /= src.rows();
}

/* mat (Nd * 3) ->  tmp (Nd * 4) */
inline void translate(const Eigen::MatrixXd& mat, Eigen::MatrixXd& tmp)
{
    tmp.setOnes();
    int Nd = mat.rows();
    Eigen::MatrixXd new_mat = mat.transpose();
    tmp.block(0,0,3,Nd) = new_mat.block(0,0,3,Nd);

}

