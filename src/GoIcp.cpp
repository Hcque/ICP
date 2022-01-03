// not use pDataTemp
// precision


#pragma once

#include <vector>
#include <queue>
#include "common.hpp"
#include "jly_3ddt.h"

#include "SingleThreadIcp.cpp"
// #include "DT.cpp"
#include "testLDT.cpp"
#include "kdtree.cpp"

#define PI 3.1415926536
#define SQRT3 1.732050808

#define MAXROTLEVEL 20


namespace icp{

struct RotNode{
    float a,b,c,w;
    double lb,ub;
    int l;
    friend bool operator<(const RotNode& n1, const RotNode& n2)
    {
        return n1.lb > n2.lb;
    }
};

struct TransNode{
    float x,y,z,w;
    double lb,ub;
    friend bool operator<(const TransNode& n1, const TransNode& n2)
    {
        return n1.lb > n2.lb;
    }
};

class GoIcp
{
public:
    GoIcp(PointCloudTPtr pModel, PointCloudTPtr pData, int iter = 1e4, double _mse = 0.001);
    ~GoIcp();

    void BuildDT();
    void init();
    void Init();
    void clear();
     void setTarget(PointCloudTPtr _target){ pModel = _target; }
     ICP_res registration();
    float runICP(Matrix4d& );
    float innerBnB(float *maxDistL, TransNode* transOut);
    float OuterBnB();

    PointCloudTPtr pModel, pData;
    int Nd, Nm;

    float optError;
    Matrix4d optMat;
    RotNode optNodeRot; TransNode optNodeTrans;

    float mseThresh, sseThresh;
    RotNode InitRot; TransNode InitTrans;

    float **maxRotDist;
	float * maxRotDis;
    std::vector<float> minDis;
    std::vector<Point3f> pDataTemp;
    SingleThreadIcp icp;
    
    // LDT dt;
    KDTree dt;
    int ITER;
};


GoIcp::GoIcp(PointCloudTPtr pModel, PointCloudTPtr pData, int _iter, double _mse) :icp(pData, pModel),
 pModel(pModel), pData(pData), ITER(_iter), dt(pModel), mseThresh(_mse)
{

std::cerr << "===init mse:" << mseThresh << "\n";
   InitRot.a = InitRot.b = InitRot.c = -PI; 
   InitRot.w = 2*PI;
   InitRot.l = 0;
   InitRot.lb = 0;

   InitTrans.lb = 0;
   InitTrans.x = -0.5;
   InitTrans.y = -0.5;
   InitTrans.z = -0.5;
   InitTrans.w = 1.0;

   Nd = pData->points.size();
   Nm = pModel->points.size();
   pDataTemp.resize(Nd);
   minDis.resize(Nd);

}

// Run ICP and calculate sum squared L2 error
float GoIcp::runICP(Matrix4d& tmpMat) {
    std::cerr << "start run ICP\n";

    ICP_res icp_result = icp.registration(tmpMat, pData);
    tmpMat = icp_result.transMat;

    // Transform point cloud and use DT to determine the L2 error
    std::vector<float> error(Nd);
    
    // auto kdt = new KDTree(pModel);

    omp_set_num_threads(2);
    #pragma omp parallel
    {
    #pragma omp for
    for(int i = 0; i < Nd; i ++) {
        // pDataTemp[i]
        auto ptemp = icp_result.resN3.block<1,3>(i,0);
        // float dis2 = dt.Distance(ptemp[0], ptemp[1], ptemp[2]);
        float dis = dt.Distance(ptemp[0], ptemp[1], ptemp[2]);
        // std::cerr << dis << " " << dis2 << "\n";
        error[i] = dis * dis;
        // if (i % 10000 == 0) std::cerr << i << "|ERROR|" << error[i] << "\n";
    }
    } // omp
    // delete kdt;

    float SSE = 0.0f;
    for (int i = 0; i < Nd; i++)
        SSE += error[i];
    std::cerr << "ICP error SSE: " << SSE << "\n";
    std::cerr << "ICP error MSE: " << SSE / (float)Nd << "\n";
    return SSE;
}


float GoIcp::OuterBnB()
{
    std::cerr << "start Outer Bnb\n";

    int i,j;
    TransNode nodeTrans; 
    RotNode nodeRot, nodeRotParent;
    clock_t clockBeginICP;
    std::priority_queue<RotNode> que;

    using namespace std; int count = 0;

	float v1, v2, v3, t, ct, ct2,st, st2;
	float tmp121, tmp122, tmp131, tmp132, tmp231, tmp232;
	float R11, R12, R13, R21, R22, R23, R31, R32, R33;
	float lb, ub, error, dis;

    optError = 0;
    // cal min dist & optError
    omp_set_num_threads(2);
    #pragma omp parallel for
    for (int i = 0; i < Nd; i ++){
        minDis[i] = dt.Distance(pData->points[i].x, pData->points[i].y, pData->points[i].z);
        // if (i % 10000 == 0)
        //     std::cerr << "mindist outer: " << minDis[i] << " ";
    }

    for(i = 0; i < Nd; i++)
	{
		optError += minDis[i]*minDis[i];
	}
	cout << "Error*: " << optError << " (Init)" << endl;

    auto tmpMat = optMat;
    // run ICP
    error = runICP(tmpMat);
    
    if(error < optError)
	{
		optError = error;
        optMat = tmpMat;
		// optR = R_icp;
		// optT = t_icp;
		cout << "Error*: " << error << " (ICP " << (double)(clock()-clockBeginICP)/CLOCKS_PER_SEC << "s)" << endl;
		cout << "ICP-ONLY Rotation Matrix + Translation Vector:" << endl;
		cout << optMat << endl;
	}
    return 0.0f;

    que.push(InitRot);

    while (1)
    {

        if(que.size() == 0 )
		{
		  cout << "Rotation Queue Empty" << endl;
		  cout << "Error*: " << optError << ", LB: " << lb << endl;
		  break;
		}

        nodeRotParent = que.top(); que.pop();

        if (optError - nodeRotParent.lb < sseThresh)
        {
            cout << "Error*: " << optError << ", LB: " << nodeRotParent.lb << ", epsilon: " << sseThresh << endl;
			break;
        }

        // if (count == ITER) break;
        if(count>0 && count%30 == 0)
        {
			printf("LB=%f  Level=%d ========\n",nodeRotParent.lb,nodeRotParent.l);

            std::cerr << "outer Bnb que size: " << que.size() << "\n";
        }
		count ++;

        // 8 cubes
        nodeRot.l =nodeRotParent.l+1;
        nodeRot.w = nodeRotParent.w / 2;
		for(j = 0; j < 8; j++)
		{
            // std::cerr << j << "out each cube\n";
		  // Calculate the smallest rotation across each dimension
			nodeRot.a = nodeRotParent.a + (j&1)*nodeRot.w ;
			nodeRot.b = nodeRotParent.b + (j>>1&1)*nodeRot.w ;
			nodeRot.c = nodeRotParent.c + (j>>2&1)*nodeRot.w ;

			// Find the subcube centre
			v1 = nodeRot.a + nodeRot.w/2;
			v2 = nodeRot.b + nodeRot.w/2;
			v3 = nodeRot.c + nodeRot.w/2;

			// Skip subcube if it is completely outside the rotation PI-ball
			if(sqrt(v1*v1+v2*v2+v3*v3)-SQRT3*nodeRot.w/2 > PI)
			{
				continue;
			}

			// Convert angle-axis rotation into a rotation matrix
			t = sqrt(v1*v1 + v2*v2 + v3*v3);
			if(t > 0)
			{
				v1 /= t;
				v2 /= t;
				v3 /= t;

				ct = cos(t);
				ct2 = 1 - ct;
				st = sin(t);
				st2 = 1 - st;

				tmp121 = v1*v2*ct2; tmp122 = v3*st;
				tmp131 = v1*v3*ct2; tmp132 = v2*st;
				tmp231 = v2*v3*ct2; tmp232 = v1*st;

				R11 = ct + v1*v1*ct2;		R12 = tmp121 - tmp122;		R13 = tmp131 + tmp132;
				R21 = tmp121 + tmp122;		R22 = ct + v2*v2*ct2;		R23 = tmp231 - tmp232;
				R31 = tmp131 - tmp132;		R32 = tmp231 + tmp232;		R33 = ct + v3*v3*ct2;

				// Rotate data points by subcube rotation matrix
                omp_set_num_threads(2);
                #pragma omp parallel for
				for(i = 0; i < Nd; i++)
				{
					Point3f& p = pData->points[i];
					pDataTemp[i].x = R11*p.x + R12*p.y + R13*p.z;
					pDataTemp[i].y = R21*p.x + R22*p.y + R23*p.z;
					pDataTemp[i].z = R31*p.x + R32*p.y + R33*p.z;
				}
			}
            else
            {
                omp_set_num_threads(2);
                #pragma omp parallel for
                for (int i = 0; i < pDataTemp.size(); i ++ ) pData->points[i] = pDataTemp[i];
            }

            // inner Bnb for ub
            ub = innerBnB(NULL, &nodeTrans); // using dataTemp
            if (ub < optError)
            {
                // Update optimal error and rotation/translation nodes
				optError = ub;
				optNodeRot = nodeRot;
				optNodeTrans = nodeTrans;

				optMat(0,0) = R11; optMat(0,1) = R12; optMat(0,2) = R13;
				optMat(1,0) = R21; optMat(1,1) = R22; optMat(1,2) = R23;
				optMat(2,0) = R31; optMat(2,1) = R32; optMat(2,2) = R33;
				optMat(0,3) = optNodeTrans.x+optNodeTrans.w/2;
				optMat(1,3) = optNodeTrans.y+optNodeTrans.w/2;
				optMat(2,3) = optNodeTrans.z+optNodeTrans.w/2;

				// cout << "Error*: " << optError << endl;
                // cout << "optMat after 1st innerBnb: " << "\n" << optMat << endl;

				// Run ICP
				clockBeginICP = clock();
				// R_icp = optR;
				// t_icp = optT;
                tmpMat = optMat;
                // std::cerr << "run inner ICP \n" ;
				error = runICP(tmpMat);
				//Our ICP implementation uses kdtree for closest distance computation which is slightly different from DT approximation, 
				//thus it's possible that ICP failed to decrease the DT error. This is no big deal as the difference should be very small.
				if(error < optError)
				{
					optError = error;
					optMat = tmpMat;
					
					// cout << "update Error*: " << error << "(ICP " << (double)(clock() - clockBeginICP)/CLOCKS_PER_SEC << "s)" << endl;
				}

				// Discard all rotation nodes with high lower bounds in the queue
				priority_queue<RotNode> queueRotNew;
				while(!que.empty())
				{
					auto node = que.top();
					que.pop();
					if(node.lb < optError)
						queueRotNew.push(node);
					else
						break;
				}
				que = queueRotNew;
            }

            lb = innerBnB(maxRotDist[nodeRot.l], NULL);
            if (lb >= optError) continue;

            nodeRot.lb = lb;
            nodeRot.ub = ub;
            // std::cerr << nodeRot.lb << "|lb OUTER BOUND ub|" << nodeRot.ub << " ===== === \n";

        if (optError - nodeRotParent.lb < sseThresh)
        {
            cout << "Error*: " << optError << ", LB: " << nodeRotParent.lb << ", epsilon: " << sseThresh << endl;
			// break;
            return optError;
        }
            que.push(nodeRot);

        }
    }

    return optError;
}


float GoIcp::innerBnB(float *maxRotDistL, TransNode* transOut)
{
    std::cerr << "start inner Bnb\n";
    int i,j;
    float transX, transY, transZ;
    float optErrorT;
    float lb, ub;
    float maxTransDis, dis;
    std::priority_queue<TransNode> que; que.push(InitTrans);
    TransNode nodeTransParent, nodeTrans;

    optErrorT = optError;

    while (que.size())
    {
        // std::cerr << "inner que size: " << que.size() << "\n";
        // std::cerr << "opt error: " << optErrorT << " ";
        // std::cerr << "nodeTransParent.lb" << nodeTransParent.lb << "\n";
        nodeTransParent = que.top(); que.pop();

        // if (optErrorT - nodeTransParent.lb < sseThresh)  {
        //     std::cerr << "inner good enough" << optErrorT << "|error lb|" << nodeTransParent.lb << "with eps:" << sseThresh << "\n";
        //     break;
        // }

        nodeTrans.w = nodeTransParent.w / 2.0;
        maxTransDis = SQRT3 / 2.0 * nodeTrans.w;

        for (j = 0; j < 8 ;j ++ )
        {
            // std::cerr << "inner trans cube: " << j << "\n";
            nodeTrans.x = nodeTransParent.x + (j & 1) * nodeTrans.w;
            nodeTrans.y = nodeTransParent.y + (j >> 1 & 1)* nodeTrans.w;
            nodeTrans.z = nodeTransParent.z + (j >> 2 & 1)* nodeTrans.w;

            transX = nodeTrans.x + nodeTrans.w / 2;
            transY = nodeTrans.y + nodeTrans.w / 2;
            transZ = nodeTrans.z + nodeTrans.w / 2;

            omp_set_num_threads(2);
            // std:: cerr << omp_get_thread_num() << "num threads\n ";
            #pragma omp parallel
            {
            #pragma omp for
            for(i = 0; i < Nd; i++)
			{
				// Find distance between transformed point and closest point in model set ||R_r0 * x + t0 - y||
				// pDataTemp is the data points rotated by R0
				minDis[i] = dt.Distance(pDataTemp[i].x + transX, pDataTemp[i].y + transY, pDataTemp[i].z + transZ);

				// Subtract the rotation uncertainty radius if calculating the rotation lower bound
				// maxRotDisL == NULL when calculating the rotation upper bound
				if(maxRotDistL)
					minDis[i] -= maxRotDistL[i];

				if(minDis[i] < 0)
				{
					minDis[i] = 0;
				}
			}
            } // omp

            // For each data point, find the incremental upper and lower bounds
			ub = 0;
			for(i = 0; i < Nd; i++)
			{
				ub += minDis[i]*minDis[i];
			}
            // std::cerr << "ub each innerBnb: " << ub << " \n";

			lb = 0;
			for(i = 0; i < Nd; i++)
			{
				// Subtract the translation uncertainty radius
				dis = minDis[i] - maxTransDis;
                // if (i % 10000 == 0) std::cerr << minDis[i] << " ";
				if(dis > 0)
					lb += dis*dis;
			}

            if (ub < optErrorT)
            {
                optErrorT = ub;
                if (transOut)
                    *transOut = nodeTrans;
            }

            if (lb >= optErrorT) continue;

            nodeTrans.lb = lb;
            nodeTrans.ub = ub;
            // std::cerr << "ub : " << ub << " ";
            // std::cerr << "lb : " << lb << " ||" << optErrorT  << " sse:" <<  sseThresh << "\n";

            if (optErrorT - nodeTrans.lb < sseThresh)  {
                // std::cerr << "inner good enough" << optErrorT << "|error lb|" << nodeTrans.lb << "with eps:" << sseThresh << "\n";
                // break;
                return optErrorT;
            }

            que.push(nodeTrans);
        }
    }
    return optErrorT;
}

void GoIcp::Init()
{
    int i,j;
    std::vector<double> normData(Nd);
    for (int i = 0;i < Nd; i ++)
        normData[i] = std::sqrt( SQ(pData->points[i].x) + 
                                    SQ(pData->points[i].y) +
                                    SQ(pData->points[i].z));
    double sigma, tmp;
    maxRotDist = new float*[MAXROTLEVEL];
    for (int i = 0 ; i < MAXROTLEVEL; i ++ )
    {
        maxRotDist[i] = new float[Nd];
        sigma = InitRot.w / std::pow(2.0, i) / 2.0;
        tmp = sigma * SQRT3;
        if (tmp > PI) tmp = PI;
        
        for (int j = 0; j < Nd; j ++)
        {
            maxRotDist[i][j] = 2 * sin(tmp/ 2) * normData[j];
        }
    }

    // Initialise so-far-best rotation and translation nodes
    optNodeRot = InitRot;
    optNodeTrans = InitTrans;

    optMat = Eigen::Matrix4d::Identity();

    sseThresh = Nd * mseThresh;
}

// void GoIcp::init()
// {
//     int i, j;
//     float sigma, maxAngle;
//     auto normData = std::vector<float>(pData->points.size());

//     // Precompute the rotation uncertainty distance (maxRotDis) for each point in the data 
//     // and each level of rotation subcube
//     // Calculate L2 norm of each point in data cloud to origin
//     for (i = 0; i < pData->points.size(); i++) {
//         normData[i] = sqrt(SQ(pData->points[i].x) + SQ(pData->points[i].y) + SQ(pData->points[i].z));
//     }

//     // maxRotDist = new float* [MAXROTLEVEL];
//     // for (i = 0; i < MAXROTLEVEL; i++) {
//     //     maxRotDist[i] = (float*) malloc(sizeof(float*) * pData->points.size());

//     //     sigma = InitRot.w / pow(2.0, i) / 2.0; // Half-side length of each level of rotation subcube
//     //     maxAngle = SQRT3 * sigma;

//     //     if (maxAngle > PI)
//     //         maxAngle = PI;
//     //     for (j = 0; j < pData->points.size(); j++)
//     //         maxRotDist[i][j] = 2 * sin(maxAngle / 2) * normData[j];
//     // }

//     // Temporary Variable
//     // we declare it here because we don't want these space to be allocated and deallocated 
//     // again and again each time inner BNB runs.
//     minDis = std::vector<float>(pData->points.size());

//     // Initialise so-far-best rotation and translation nodes
//     optNodeRot = InitRot;
//     optNodeTrans = InitTrans;

//     optMat = Eigen::Matrix4d::Identity();
// }

void GoIcp::clear()
{
    for (int i = 0; i < MAXROTLEVEL; i ++ ) delete [] maxRotDist[i];
    delete [] maxRotDist;
    delete [] maxRotDis;
    std::cerr << "clear\n";
}

ICP_res GoIcp::registration()
{
    Init();
    OuterBnB();
    clear();
    ICP_res ans;
    Eigen::MatrixXd tmp(4,Nd);
    auto mat = pData->getMatrixXfMap(3,4,0).transpose().cast<double>(); //  Nd * 3
    translate(mat, tmp);
    ans.resN3 = (optMat * tmp).block(0,0,3,Nd).transpose();
    return ans;
}

}