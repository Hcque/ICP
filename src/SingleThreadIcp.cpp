// finMat * left or right 

// t is bad, after doing kdtree swap


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
#include <omp.h>

namespace icp
{


struct Correspondance
{
    std::vector<int> points;
    std::unordered_map<int,int> points2;
    Matrix3d covariance;
};

class SingleThreadIcp //: public Registration
{
public:
    SingleThreadIcp(PointCloudTPtr _s, PointCloudTPtr _t, int _iter = 1): kdt(_t, 3),
        source(_s), target(_t), max_iter(_iter){
        src = source->getMatrixXfMap(3,4,0).transpose().cast<double>();
        tar = target->getMatrixXfMap(3,4,0).transpose().cast<double>();
        cal_mean(tar,mean_tar);
        Nd = source->points.size();
        Nm = target->points.size();
        tempData = src;
        cal_mean(tempData,mean_tmpdata);
        Neighbour.points.resize(Nd);
    }
    ~SingleThreadIcp() {}

    void setTarget(PointCloudTPtr _target) { 
        target = _target; 
        cal_mean(tar,mean_tar);
    }
    void setSource(const PointCloudTPtr _source) { 
        source = _source; 
        src = source->getMatrixXfMap(3,4,0).transpose().cast<double>();
        tempData = src;
        Nd = source->points.size();
        cal_mean(tempData,mean_tmpdata);
        Neighbour.points.resize(Nd);
    }

    virtual ICP_res registration(Matrix4d&, const PointCloudTPtr&);

    void cal_covarance();
    void naive_search();
    void kdtree_search();
    Matrix4d& best_fit_transform();
    void test_kdtree(int );
    void test_kdtree_para(int );

    PointCloudTPtr source, target;
    Eigen::MatrixXd src, tar, tempData;
    int max_iter;
    Correspondance Neighbour;
    int Nd, Nm;

    Matrix4d T;
    Vector3d mean_src, mean_tar, mean_tmpdata;
    KDTree kdt;
};

void SingleThreadIcp:: test_kdtree(int II){
    std::cerr << "test kd tree correctness \n ";
    int r = src.rows();
    std::cerr << src.rows() << "|rows|" << tar.rows() << "\n";

    // std::ofstream out("./points.txt");
    // std::ofstream diff("./diff.txt");

    // omp_set_num_threads(2);

//    #pragma omp parallel for schedule (static)
   for (int i = 0 ; i < src.rows(); i ++ ) 
   {
    //    if (i != II) continue;
        if (i % 10000 == 0) 
            std::cerr << "i" << i << "\n";
        // Vector3f v1;
		// for(int d=0; d<3; d++)
		// 	v1[d] = src(i,d);
        auto v1 = src.block<1,3>(i,0).transpose();
        int j_kdtree = kdt.get_closest(v1);
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
            std::cerr << "DIFF!:" << i <<  " kdi:| " <<  j_kdtree << "kd_Dist:" << kdt.Q.top().first << "\n";
       }

   }

    } // omp

    }



void SingleThreadIcp:: test_kdtree_para(int II){
    std::cerr << "test kd tree correctness \n ";
    int r = src.rows();
    std::cerr << src.rows() << "|rows|" << tar.rows() << "\n";

    // std::ofstream out("./points.txt");
    // std::ofstream diff("./diff.txt");

    // omp_set_num_threads(2);

//    #pragma omp parallel for schedule (static)
   for (int i = 0 ; i < src.rows(); i ++ ) 
   {
    //    if (i != II) continue;
        if (i % 10000 == 0) 
            std::cerr << "i" << i << "\n";
        // Vector3f v1;
		// for(int d=0; d<3; d++)
		// 	v1[d] = src(i,d);
        auto v1 = src.block<1,3>(i,0).transpose();
        int j_kdtree = kdt.get_closest(v1);
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
            std::cerr << "DIFF!:" << i <<  " kdi:| " <<  j_kdtree << "kd_Dist:" << kdt.Q.top().first << "\n";
       }

   }

    } // omp

    }


void SingleThreadIcp::naive_search()
{
    Neighbour.points.clear();
   for (int i = 0 ; i < Nd; i ++ ) 
   {
       if (i == 100) {
            std::cerr << "i" << i << "\n";
            break;
       }
        auto v1 = src.block<1,3>(i,0).transpose();
        double min_dist = 1e9;
       for (int j = 0; j < Nm; j ++ )
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
    // Neighbour.points.clear();
    
    omp_set_num_threads(NUM_TH);
    std::cerr <<  "omp # of procs: " << omp_get_num_procs() << "\n";
    std::cerr <<  "omp # of threads: " << omp_get_num_threads() << "\n";

   #pragma omp parallel for
   for (int i = 0 ; i < Nd; i ++ ) 
   {
        // if (i % 10000 == 0) 
        // {
        //     std::cerr << "i" << i << "\n";
        //     std::cerr <<  "THEreads: " << omp_get_thread_num() << "\n";
        //     std::cerr <<  "omp # of threads: " << omp_get_num_threads() << "\n";
        // }
       
        auto v1 = tempData.block<1,3>(i,0).transpose();
        // std::cerr << "i" << tempData.block<1,3>(i,0) << "\n";
        int j = kdt.get_closest(v1);
        Neighbour.points[i] = j;
        // std::cerr << i << "|pair|" << j << std::endl;
   }

    // } // omp
    assert(Neighbour.points.size() == Nd);
//    delete kdt;
}


void SingleThreadIcp::kdtree_search()
{
    // Neighbour.points.clear();
    
    omp_set_num_threads(NUM_TH);
    std::cerr <<  "omp # of procs: " << omp_get_num_procs() << "\n";
    std::cerr <<  "omp # of threads: " << omp_get_num_threads() << "\n";

   #pragma omp parallel for
   for (int i = 0 ; i < Nd; i ++ ) 
   {
        // if (i % 10000 == 0) 
        // {
        //     std::cerr << "i" << i << "\n";
        //     std::cerr <<  "THEreads: " << omp_get_thread_num() << "\n";
        //     std::cerr <<  "omp # of threads: " << omp_get_num_threads() << "\n";
        // }
       
        auto v1 = tempData.block<1,3>(i,0).transpose();
        // std::cerr << "i" << tempData.block<1,3>(i,0) << "\n";
        int j = kdt.get_closest(v1);
        Neighbour.points[i] = j;
        // std::cerr << i << "|pair|" << j << std::endl;
   }

    // } // omp
    assert(Neighbour.points.size() == Nd);
//    delete kdt;
}


void SingleThreadIcp::kdtree_singleTH_search()
{
    Neighbour.points2.clear();
    
   for (int i = 0 ; i < Nd; i ++ ) 
   {
        // if (i % 10000 == 0) 
        // {
        //     std::cerr << "i" << i << "\n";
        //     std::cerr <<  "THEreads: " << omp_get_thread_num() << "\n";
        //     std::cerr <<  "omp # of threads: " << omp_get_num_threads() << "\n";
        // }
       
        auto v1 = tempData.block<1,3>(i,0).transpose();
        // std::cerr << "i" << tempData.block<1,3>(i,0) << "\n";
        int j = kdt.get_closest(v1);
        Neighbour.points2[i] = j;
        // std::cerr << i << "|pair|" << j << std::endl;
   }

    // } // omp
    assert(Neighbour.points.size() == Nd);
//    delete kdt;
}




void SingleThreadIcp::kdtree_search()
{
    // Neighbour.points.clear();
    
    omp_set_num_threads(NUM_TH);
    std::cerr <<  "omp # of procs: " << omp_get_num_procs() << "\n";
    std::cerr <<  "omp # of threads: " << omp_get_num_threads() << "\n";

   #pragma omp parallel for
   for (int i = 0 ; i < Nd; i ++ ) 
   {
        // if (i % 10000 == 0) 
        // {
        //     std::cerr << "i" << i << "\n";
        //     std::cerr <<  "THEreads: " << omp_get_thread_num() << "\n";
        //     std::cerr <<  "omp # of threads: " << omp_get_num_threads() << "\n";
        // }
       
        auto v1 = tempData.block<1,3>(i,0).transpose();
        // std::cerr << "i" << tempData.block<1,3>(i,0) << "\n";
        int j = kdt.get_closest(v1);
        Neighbour.points[i] = j;
        // std::cerr << i << "|pair|" << j << std::endl;
   }

    // } // omp
    assert(Neighbour.points.size() == Nd);
//    delete kdt;
}


void SingleThreadIcp::cal_covarance()
{
   // cal covarance
   std::cerr << " start cal cov\n";
   cal_mean(tempData,mean_tmpdata);

    Neighbour.covariance.setZero();
    Eigen::MatrixXd tempTar(Neighbour.points.size(), 3);
    int i = 0;
    for (auto it = Neighbour.points.begin(); it != Neighbour.points.end(); ++it)
    {
        tempTar.block(i,0,1,3) = tar.block<1,3>(it->second,0);
        i++;
        mean_tar += tar.block<1,3>(it->second,0).transpose();
    }
    mean_tar /= tempTar.rows();

    i = 0;
    for (auto it = Neighbour.points.begin(); it != Neighbour.points.end(); ++it)
    {
        // std::cerr << it->first << " ";
        auto a = tempData.block<1,3>(it->first,0).transpose() - mean_tmpdata;

        auto b = tempTar.block(i,0,1,3).transpose() - mean_tar;
        // tempTar.block(i,0,1,3) = tar.block<1,3>(it->second,0);
//    std::cerr << "a" << a << "\n";
//    std::cerr << "b" << b << "\n";
//    std::cerr << (a.transpose() * b) << "\n";
        Neighbour.covariance +=  a * b.transpose();
        i++;
    }

    Neighbour.covariance /= Neighbour.points.size();
    // std::cerr << "corvance matrix: \n";
    // std::cerr << Neighbour.covariance << "\n";
}

Matrix4d& SingleThreadIcp::best_fit_transform()
{
    using namespace Eigen;
    JacobiSVD<MatrixXd>svd(Neighbour.covariance, ComputeFullU|ComputeFullV);
    MatrixXd U,V, Vt;
    VectorXd S;
    Matrix3d R; Vector3d t;
    U = svd.matrixU();
    S = svd.singularValues();
    V = svd.matrixV();
	Vt = V.transpose();

    R = (U*Vt).transpose(); 
	if (R.determinant()<0)
	{
		Vt.block<1,3>(2,0) *= -1;
		R = (U*Vt).transpose();
	}
//     std::cerr << "cen_A:" << mean_tmpdata << "\n";
// std::cerr << "cen_B:" << mean_tar << "\n";

    t = mean_tar - R * mean_tmpdata;

	T = MatrixXd::Identity(4,4);
    T.block<3,3>(0,0) = R;
	T.block<3,1>(0,3) = t;

    // update tempData
    Eigen::MatrixXd tmp(4,Nd);
    translate(tempData, tmp);
    tempData = (T * tmp).block(0,0,3,Nd).transpose(); //src as initial state
    assert(tempData.rows() == Nd); 
    assert(tempData.cols() == 3); 

    double mse = 0.0;
    for (int i = 0 ; i < src.rows(); i ++ ) 
    {
        auto b = tempData.block<1,3>(i,0).transpose();
        mse += Dist(b,tar.block<1,3>(Neighbour.points[i],0).transpose());
    } 
    mse /= Nd;
    std::cerr << "each transform mse: "  << mse << "\n";
    return T;
}


ICP_res SingleThreadIcp::registration(Matrix4d& guess, const PointCloudTPtr& pData)
{
   std::cerr << "start register"  << "\n";
   setSource(pData);

   Matrix4d finMat = guess;
   auto X = source->getMatrixXfMap(3,4,0).transpose().cast<double>();
   Eigen::MatrixXd tmp(4,Nd);
   translate(X, tmp);
   tempData = (guess * tmp).block(0,0,3,Nd).transpose(); //src as initial state
   assert(tempData.rows() == Nd); 
   assert(tempData.cols() == 3); 

    std::cerr << "Mat: " << finMat << "\n";
    for (int iter = 0; iter < max_iter; iter++ )
    {
        // naive_search(); 
        std::cerr << "iter: " << iter << "\n";
        kdtree_search();
        cal_covarance();
        auto curMat = best_fit_transform(); // here tempData changes accordingly
        finMat =  curMat * finMat;
        std::cerr << "Mat: \n" << finMat << "\n";
    }
    translate(src, tmp);

    ICP_res ans;
    ans.resN3 = (finMat*tmp).block(0,0,3,Nd).transpose();
    ans.transMat = finMat;

   std::cerr << "ok register: " << ans.resN3.rows() << "||" << ans.resN3.cols()   << "\n";
   std::cerr << ans.transMat << "\n";
   return ans;

}


}
