#pragma once

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <pcl/common/common.h>

#include <omp.h>

namespace icp
{

    int NUM_TH = 1;
#define SQ(x) ((x)*(x))

using IndexT = uint32_t;
#define INVALID_INDEX IndexT(-1)

using Vector3f = Eigen::Vector3f;
using Vector3d = Eigen::Vector3d;
using Vector3i = Eigen::Vector3i;

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
    // std::cerr << tmp.block(0,4,4,1) << "\n";
    int Nd = mat.rows();
    Eigen::MatrixXd new_mat = mat.transpose();
    // std::cerr << new_mat.rows() << "| in translate|" << mat.cols() << "\n";
    tmp.block(0,0,3,Nd) = new_mat.block(0,0,3,Nd);
    // for (int i = 0 ; i < Nd; i ++ )
        // tmp.block(0,i, 3,1) = mat.block(i,0,1,3).transpose();
    // std::cerr << tmp.rows() << "| in translate|" << tmp.cols() << "\n";
    // std::cerr << tmp.block(0,4,4,1) << "\n";

}


inline int MaxDim(const Vector3f& v) {
    auto dim = 0u;
    auto val = -INFINITY;
    for (auto i = 0u; i < 3; i++)
        if (v[i] > val) {
            dim = i;
            val = v[i];
        }
    return dim;
}

inline float MaxComp(const Vector3f& v) {
    return std::max(std::max(v.x(), v.y()), v.z());
}

Point3f operator+(const Point3f& a, const Point3f& b)
{
    Point3f ans;
    ans.x = a.x + b.x ;
    ans.y = a.y + b.y ;
    ans.z = a.z + b.z ;
    return ans;
}

Point3f operator/(const Point3f& a, int b)
{
    Point3f ans;
    ans.x = a.x / b;
    ans.y = a.y / b;
    ans.z = a.z / b;
    return ans;
}

struct Bound3f {
    Point3f min, max;

    Bound3f(const Point3f& min, const Point3f& max) : min(min), max(max) {
        for (auto i = 0u; i < 3; i++)
            if (this->min.data[i] > this->max.data[i])
                std::swap(this->min.data[i], this->max.data[i]);
    }

    Bound3f(const PointCloudT& cloud) {
        pcl::getMinMax3D(cloud, min, max);
    }

    Vector3f Diagonal() const {
        return max - min;
    }
};

inline std::ostream& operator << (std::ostream& os, const Bound3f& bnd) {
    os << "min: " << bnd.min << " max: " << bnd.max << '\n';
    return os;
}


}