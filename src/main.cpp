
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
#include "jly_3ddt.h"

#include "jly_icp3d.cpp"
#include "common.hpp"
#include "GoIcp.cpp"
#include "LinearDT.hpp"

#include <time.h>
#include <fstream>

using namespace icp;

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

// void testkdtree(int II)
// {
	
// 	auto singleICP = new SingleThreadIcp(cloud_source, cloud_target, 1);

//      singleICP->test_kdtree(II);

// }

void test_icp(int ite){
clockBegin = clock();
	auto singleICP = new SingleThreadIcp(cloud_source, cloud_target,ite);
	Matrix4d tmpMat = Eigen::Matrix4d::Identity();
    ICP_res icpres = singleICP->registration(tmpMat, cloud_source);
	auto source_trans_matrix = icpres.resN3;


clockEnd = clock();
    cout << (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC << "\n" << endl;


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
	clockBegin = clock();
	auto goicp = new GoIcp(cloud_target, cloud_source, ite);
	goicp->sseThresh = mse;

	std::cout << "Building Distance Transform...\n";
	goicp->BuildDT();
	// cout << (double)(clockEnd - clockBegin)/CLOCKS_PER_SEC << "s (CPU)" << endl;

     auto icpres = goicp->registration();
	 clockEnd = clock();
    cout << (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC << "\n" << endl;


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
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(cloud_source,100,100,100);
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


void testDT(const PointCloudTPtr& pModel)
{
	    clock_t  clockBegin, clockEnd;
	// dt.SIZE = 300;

	 cout << "building dt(s):";
    clockBegin = clock();
    LinearDT dt(pModel); 
    // auto x = std::make_unique<double[]>(2);
    // auto y = std::make_unique<double[]>(2);
    // auto z = std::make_unique<double[]>(2);
	// int i = 0;
    //     x[i] = -0.5; y[i] = -0.5; z[i] = -0.5;
    //     i = 1; x[i] = 0.5; y[i] = 0.5; z[i] = 0.5;
    // dt.Build(x.get(), y.get(), z.get(), 2);
    clockEnd = clock();
    cout << (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC << "\n" << endl;
	std:: cerr << dt.Distance(0.1,0.1,0.1) << "\n";


	//  cout << "building dt(s):";
    // clockBegin = clock();
    // // auto x = std::make_unique<double[]>(pModel->size());
    // // auto y = std::make_unique<double[]>(pModel->size());
    // // auto z = std::make_unique<double[]>(pModel->size());
    // for (int i = 0; i < pModel->points.size(); i++) {
    //     x[i] = pModel->points[i].x;
    //     y[i] = pModel->points[i].y;
    //     z[i] = pModel->points[i].z;
    // }
    // dt.Build(x.get(), y.get(), z.get(), pModel->points.size());
    // clockEnd = clock();
    // cout << (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC << "\n" << endl;
	// std:: cerr << dt.Distance(0.1,0.1,0.1);

    // cout << "building dt(by wzh)(s):";
    // clockBegin = clock();
    // NaiveDT df(pModel);
    // clockEnd = clock();
    // cout << (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC << "\n" << endl;

    // cout << "building dt(by lq)(s):";
    // clockBegin = clock();
    // LinearDT ldt(pModel);
    // clockEnd = clock();
    // cout << (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC << "\n" << endl;

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
	int n_sample = atoi(argv[4]);
	mse = atof(argv[5]);
	NUM_TH = atoi(argv[6]);

	// _normal(cloud_source);
	// _normal(cloud_target, 0);

	// sampling
	std::vector<Point3f> ss, tt;
	for (int i = 0 ; i < n_sample; i ++){
		ss.push_back( cloud_source->points[rand() % 40000]);
		tt.push_back( cloud_target->points[rand() % 40000]);
	}
	cloud_source->points.clear();
	cloud_target->points.clear();
	assert (cloud_source->points.size() == 0);
	for (int i = 0 ; i < n_sample; i ++){
		cloud_source->points.push_back( ss[i]);
		cloud_target->points.push_back( tt[i]);
	}

	source_matrix = cloud_source->getMatrixXfMap(3,4,0).transpose().cast<double>();
	target_matrix = cloud_target->getMatrixXfMap(3,4,0).transpose().cast<double>();
	std :: cerr << source_matrix.rows() << "\n";
	std :: cerr << cloud_target->points.size() << "\n";
		
	// testkdtree(atoi(argv[3]));
	test_icp(atoi(argv[3]));

	// test_goicp(atoi(argv[3]));
	// testDT(cloud_source);
    
    return 0;
}
