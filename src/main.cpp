
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

#include "SingleThreadIcp.cpp"
#include "common.hpp"
#include "GoIcp.cpp"
#include "testLDT.cpp"

#include <time.h>
#include <cmath>
#include <fstream>


#include <omp.h>

using namespace icp;
using namespace std;

PointCloudTPtr cloud_source ,cloud_target;
Eigen::MatrixXd source_matrix, target_matrix;

clock_t clockBegin, clockEnd;


void loadFile(const char* file_name, PointCloudT &cloud)
{
	pcl::PolygonMesh mesh;

	if(pcl::io::loadPolygonFile(file_name, mesh)==-1)
	{
		PCL_ERROR("File loading faild.");
		return;
	}
	else
	{
		pcl::fromPCLPointCloud2<Point3f>(mesh.cloud, cloud);
	}

	std::vector<int> index;
	pcl::removeNaNFromPointCloud(cloud, cloud, index);
}


void test_kdtree()
{
	auto singleICP = new SingleThreadIcp(cloud_source, cloud_target,10);
	Matrix4d tmpMat = Eigen::Matrix4d::Identity();
    singleICP->test_kdtree_SINGLE(0);
    singleICP->test_kdtree_para(0);

}


void test_icp(int ite){
	auto singleICP = new SingleThreadIcp(cloud_source, cloud_target,ite);
	Matrix4d tmpMat = Eigen::Matrix4d::Identity();

	double start_time = omp_get_wtime();
    ICP_res icpres = singleICP->registration(tmpMat, cloud_source);
	auto source_trans_matrix = icpres.resN3;

	double end_time = omp_get_wtime();
	std::cerr << end_time - start_time << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans (new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ> temp_cloud;
		int row = source_trans_matrix.rows();
		temp_cloud.width = row;
		temp_cloud.height = 1;
		temp_cloud.points.resize(row);
		for (size_t n=0; n<row; n++) 
		{
			temp_cloud[n].x = source_trans_matrix(n,0);
			temp_cloud[n].y = source_trans_matrix(n,1);
			temp_cloud[n].z = source_trans_matrix(n,2);	
  		}
  		cloud_source_trans = temp_cloud.makeShared();

	{ // visualization
		boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(255,255,255);

		// black
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(cloud_source,0,0,0);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_source,source_color,"source");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"source");

		// blue
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(cloud_target,0,0,255);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_target,target_color,"target");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"target");

		//red
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_trans_color(cloud_source_trans,255,0,0);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_source_trans,source_trans_color,"source trans");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"source trans");

		viewer->getRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera()->SetParallelProjection(1);
		viewer->resetCamera();
		viewer->spin();
	}

}


double mse;
void test_goicp(int ite){
	auto goicp = new GoIcp(cloud_target, cloud_source, ite, mse);

     auto icpres = goicp->registration();
	 auto source_trans_matrix = icpres.resN3;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans (new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ> temp_cloud;
		int row = source_trans_matrix.rows();
		temp_cloud.width = row;
		temp_cloud.height = 1;
		temp_cloud.points.resize(row);
		for (size_t n=0; n<row; n++) 
		{
			// std::cerr << "n:" << n << "\n";
			temp_cloud[n].x = source_trans_matrix(n,0);
			temp_cloud[n].y = source_trans_matrix(n,1);
			temp_cloud[n].z = source_trans_matrix(n,2);	
  		}
  		cloud_source_trans = temp_cloud.makeShared();

	{ // visualization
		boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(255,255,255);

		// black
		// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(cloud_source,0,255,0);
		// viewer->addPointCloud<pcl::PointXYZ>(cloud_source,source_color,"source");
		// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"source");

		// blue
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(cloud_target,0,0,255);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_target,target_color,"target");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"target");

		//red
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_trans_color(cloud_source_trans,255,0,0);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_source_trans,source_trans_color,"source trans");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"source trans");

		viewer->getRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera()->SetParallelProjection(1);
		viewer->resetCamera();
		viewer->spin();
	}

}

void _normal(PointCloudTPtr &cloud_target, int file = 0)
{
	fstream _file;
	_file.open("model.txt", ios::out);
	double _min_x = 1e9, _max_x = -1e9;
	double _min_y = 1e9, _max_y = -1e9;
	double _min_z = 1e9, _max_z = -1e9;

	if (file) _file << cloud_target->points.size() << "\n";
	for (int i = 0 ; i < cloud_target->points.size(); i ++ )
	{
		double x = cloud_target->points[i].x;
		if (x < _min_x) _min_x = x;
		if (x > _max_x) _max_x = x;
		double y = cloud_target->points[i].y;
		if (y < _min_y) _min_y = y;
		if (y > _max_y) _max_y = y;
		double z = cloud_target->points[i].z;
		if (z < _min_z) _min_z = z;
		if (z > _max_z) _max_z = z;
	}


	for (int i = 0 ; i < cloud_target->points.size(); i ++ )
	{
		double x = cloud_target->points[i].x;
		x = -0.5 + (x - _min_x)*1.0/ (_max_x-_min_x);
		double y = cloud_target->points[i].y;
		y = -0.5 + (y - _min_y)*1.0/ (_max_y-_min_y);
		double z = cloud_target->points[i].z;
		z = -0.5 + (z - _min_z)*1.0/ (_max_z-_min_z);
		cloud_target->points[i] = {x,y,z};
		if (file){
		_file << 
		cloud_target->points[i].x << " " << 
		cloud_target->points[i].y << " " << 
		cloud_target->points[i].z << "\n";
		}
	}
	_file.close();

}


LDT *ldt;
KDTree *kdt;

float _X, _Y, _Z;
int CORES_ ;
void test_LinearDT(const PointCloudTPtr& pModel)
{
    // ldt = new LDT(pModel, 100, CORES_); 
    // ldt = new LDT(pModel, 200, CORES_); 
    ldt = new LDT(pModel, 300, CORES_ ); 
    kdt = new KDTree(pModel); 
	// lldt = new LinearDT(pModel);

	for (auto& pp: cloud_source->points){
		_X = pp.x;
		_Y = pp.y;
		_Z = pp.z;

		float res1, res2, res3, res4;

		res1 = ldt->Distance(_X, _Y, _Z) ;
		// res2 = lldt->Evaluate(Point3f(_X, _Y, _Z)) ;
		res3 = kdt->Distance(_X, _Y, _Z) ;
		// res4 = LDTprev->Distance(_X, _Y, _Z);
		// if (std::fabs(res1 - res2)  > 1.732 * 2 *  ldt->cellLen ) cc ++;
		// if (std::fabs(res1 - res2)  > 1.732 * 2 *  ldt->cellLen ) cc ++;
		// if (std::fabs(res1 - res2)  > 1.732 * 3 *  ldt->cellLen ) cc3 ++;
		// if (std::fabs(res1 - res2)  > 1.732 * 5 *  ldt->cellLen ) cc5 ++;

		if (std::fabs(res1 -res3) > 0.1)
			std::cerr << res1 << "||" << res2   << "||" << res3 << "||\n" ; // << res4 <<  "\n";
		
	}
}



int main(int argc, char**argv)
{
    // if (argc != 4) 
    // {
    //     std::cerr << "./a input1 input2 iter" << "\n";
    //     return -1;
    // }
	std::cerr << "load..." << "\n";
    cloud_source  = PointCloudTPtr(new PointCloudT());
    cloud_target  = PointCloudTPtr(new PointCloudT());
    loadFile(argv[1], *cloud_source);
    loadFile(argv[2], *cloud_target);
	CORES_ = atoi(argv[3]);
	// _Y = atof(argv[4]);
	// _Z = atof(argv[5]);

	// std::cerr << _X << " ";
	// std::cerr << _Y << " ";
	// std::cerr << _Z << " ";
	// std::cerr << "\n";

	// int n_sample = atoi(argv[3]);
	mse = atof(argv[4]);
	// NUM_TH = atoi(argv[6]);

	// _normal(cloud_source);
	// _normal(cloud_target, 0);

	// sampling
	// std::vector<Point3f> ss, tt;
	// for (int i = 0 ; i < n_sample; i ++){
	// 	ss.push_back( cloud_source->points[rand() % 40000]);
	// 	tt.push_back( cloud_target->points[rand() % 40000]);
	// }
	// cloud_source->points.clear();
	// cloud_target->points.clear();
	// assert (cloud_source->points.size() == 0);
	// for (int i = 0 ; i < n_sample; i ++){
	// 	cloud_source->points.push_back( ss[i]);
	// 	cloud_target->points.push_back( tt[i]);
	// }

	// source_matrix = cloud_source->getMatrixXfMap(3,4,0).transpose().cast<double>();
	// target_matrix = cloud_target->getMatrixXfMap(3,4,0).transpose().cast<double>();
	// std :: cerr << source_matrix.rows() << "\n";
	// std :: cerr << cloud_target->points.size() << "\n";
		
	// test_kdtree();
	test_icp(atoi(argv[3]));

	// test_goicp(atoi(argv[3]));
	// test_LinearDT(cloud_target);
    
    return 0;
}
