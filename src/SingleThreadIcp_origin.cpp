#pragma once

#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/common/transforms.h>

#include <vtkRenderWindow.h>
#include <vtkRendererCollection.h>
#include <vtkCamera.h>

#include "common.hpp"
#include "Registration.hpp"
#include "kdtree.cpp"

#include <unordered_map>
#include <vector>
#include <fstream>

namespace icp
{

struct Correspondance
{
    std::unordered_map<int, int> points;
    Matrix3f covariance;
};

KDTree* kdt;

class SingleThreadIcp: public Registration
{
public:
    SingleThreadIcp(PointCloudTPtr _s, PointCloudTPtr _t, int _iter = 1): 
        source(_s), target(_t), max_iter(_iter){
        src = source->getMatrixXfMap(3,4,0).transpose();
        tar = target->getMatrixXfMap(3,4,0).transpose();

        cal_mean(tar,mean_tar);
    }
    ~SingleThreadIcp() {}

    void setTarget(PointCloudTPtr _target) { target = _target; }


    virtual ICP_res registration(Matrix4f& );
    ICP_res registration() ;

    void cal_covarance();
    void naive_search();
    void kdtree_search();
    Matrix4f& best_fit_transform();
    void test_kdtree(int );

    PointCloudTPtr source, target;
    Eigen::MatrixXf src, tar;
    int max_iter;
    Correspondance Neighbour;

    Matrix4f T;
    Vector3f mean_src, mean_tar;
};

void SingleThreadIcp:: test_kdtree(int II){
    std::cerr << "test kd tree correctness \n ";
    int r = src.rows();
    std::cerr << src.rows() << "|rows|" << tar.rows() << "\n";

    std::ofstream out("./points.txt");
    std::ofstream diff("./diff.txt");
    kdt = new KDTree(target, 3);
    Neighbour.points.clear();
    //   omp_set_num_threads(4);
    // #pragma omp parallel
    // {
    // #pragma omp for
   for (int i = 0 ; i < src.rows(); i ++ ) 
   {
       if (i != II) continue;
        if (i % 100 == 0) 
            std::cerr << "i" << i << "\n";
        // Vector3f v1;
		// for(int d=0; d<3; d++)
		// 	v1[d] = src(i,d);
        auto v1 = src.block<1,3>(i,0).transpose();
        int j_kdtree = kdt->get_closest(v1);
        // Neighbour.points[i] = j;
        // std::cerr << i << "|pair|" << j << std::endl;
          double min_dist = 1e9;
       for (int k = 0; k < tar.rows(); k ++ )
       {
            auto v2 = tar.block<1,3>(k,0).transpose();
            double get_dist = Dist(v1,v2);
            // std::cerr << "get_dist" << get_dist << "\n";
            if (get_dist < min_dist) 
            {
                min_dist = get_dist; 
                Neighbour.points[i] = k;
                // std::cerr << i << "|pair|" << k << " min_dist:" <<  min_dist << std::endl;
            }
       }

        // out << i << " " << Neighbour.points[i] << "|mindist:" << min_dist << "\n";
        std::cerr << i << "truth:" << Neighbour.points[i] << "|mindist:" << min_dist << "\n";
    //    std::cerr << Neighbour.points[i] << " " <<  j_kdtree << "\n";
       if (Neighbour.points[i] !=  j_kdtree){
            diff << "DIFF!:" << i << " " <<  Neighbour.points[i] << " " <<  j_kdtree << "\n";
            std::cerr << "DIFF!:" << i <<  " kdi:| " <<  j_kdtree << "kd_Dist:" << kdt->Q.top().first << "\n";
       }

   }
   delete kdt;
    }

void SingleThreadIcp::naive_search()
{
    int r = src.rows();
    std::cerr << src.rows() << "|rows|" << tar.rows() << "\n";
 
    Neighbour.points.clear();

   for (int i = 0 ; i < src.rows(); i ++ ) 
   {
       if (i == 100) {
            std::cerr << "i" << i << "\n";
            break;
       }
        auto v1 = src.block<1,3>(i,0).transpose();
        double min_dist = 1e9;
       for (int j = 0; j < tar.rows(); j ++ )
       {
            auto v2 = tar.block<1,3>(j,0).transpose();
            double get_dist = Dist(v1,v2);
            // std::cerr << "get_dist" << get_dist << "\n";
            if (get_dist < min_dist) 
            {
                min_dist = get_dist; 
                Neighbour.points[i] = j;
                // out << i << " " << j << "\n";
                std::cerr << i << "|pair|" << j << " min_dist:" <<  min_dist << std::endl;
            }
       }
   }
}


void SingleThreadIcp::kdtree_search()
{
    int r = src.rows();
    std::cerr << src.rows() << "|rows|" << tar.rows() << "\n";

    kdt = new KDTree(target, 3);
    Neighbour.points.clear();
   for (int i = 0 ; i < src.rows(); i ++ ) 
   {
        if (i % 10000 == 0) 
            std::cerr << "i" << i << "\n";
       
        auto v1 = src.block<1,3>(i,0).transpose();
        int j = kdt->get_closest(v1);
        Neighbour.points[i] = j;
        // std::cerr << i << "|pair|" << j << std::endl;
   }
   delete kdt;
}


void SingleThreadIcp::cal_covarance()
{
   // cal covarance
   cal_mean(src,mean_src);

    Neighbour.covariance.setZero();
    for (auto it = Neighbour.points.begin(); it != Neighbour.points.end(); ++it)
    {
        // std::cerr << it->first << " ";
        auto a = src.block<1,3>(it->first,0) - mean_src.transpose();
        auto b = tar.block<1,3>(it->second,0) - mean_tar.transpose();
//    std::cerr << "a" << a << "\n";
//    std::cerr << "b" << b << "\n";
//    std::cerr << (a.transpose() * b) << "\n";
        Neighbour.covariance +=  a.transpose() * b;

    }
    Neighbour.covariance /= Neighbour.points.size();
    std::cerr << "corvance matrix: \n";
    std::cerr << Neighbour.covariance << "\n";
}

Matrix4f& SingleThreadIcp::best_fit_transform()
{
    using namespace Eigen;
    JacobiSVD<MatrixXf>svd(Neighbour.covariance, ComputeFullU|ComputeFullV);
    MatrixXf U,V, Vt;
    VectorXf S;
    Matrix3f R; Vector3f t;
    U = svd.matrixU();
    S = svd.singularValues();
	V = svd.matrixV();
	Vt = V.transpose();
    //    std::cerr << "Vt" << Vt << "\n";

    // R = U * Vt;
    //    std::cerr << "R" << R << "\n";
    	R = (U*Vt).transpose(); //#TODO

	if (R.determinant()<0)
	{
		Vt.block<1,3>(2,0) *= -1;
		R = (U*Vt).transpose();
	}

    t = mean_tar - R * mean_src;

	T = MatrixXf::Identity(4,4);
    T.block<3,3>(0,0) = R;

	T.block<3,1>(0,3) = t;
    //    std::cerr << "T" << T << "\n";

    double mse = 0.0;
    for (int i = 0 ; i < src.rows(); i ++ ) 
    {
        auto a = src.block<1,3>(i,0).transpose();
        Vector4f tmp(a[0], a[1], a[2], 1.0);
        auto res = T * tmp;
        // Vector3f tmp_res(res[0],res[1], res[2]);
        src(i,0) = res[0];
        src(i,1) = res[1];
        src(i,2) = res[2];
        auto b = src.block<1,3>(i,0).transpose();
        // std::cerr << "BB:" << b << "\n";
        mse += Dist(b,tar.block<1,3>(Neighbour.points[i],0).transpose());
    } 
    mse /= src.rows();
    std::cerr << "mse: "  << mse << "\n";
    return T;
}


ICP_res SingleThreadIcp::registration(Matrix4f& guess)
{
   std::cerr << "start register"  << "\n";
   Matrix4f finMat = guess;
   auto X = source->getMatrixXfMap(3,4,0).transpose();
   std::cerr << X.rows()  << "|" << X.cols() << "\n";

    for (int iter = 0; iter < max_iter; iter++ )
    {
        // naive_search(); 
        kdtree_search();
        cal_covarance();
        auto curMat = best_fit_transform();
        finMat = finMat * curMat;
    }
    ICP_res res;
    res.resN3 = src;
    res.transMat = finMat;

   std::cerr << "ok register"  << "\n";
   return res;

}


ICP_res registration( )
{
    ICP_res ans;
    return ans;

}
}
