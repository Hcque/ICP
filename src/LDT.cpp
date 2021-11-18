// = or ==
#pragma once

#include "common.hpp"
#include <omp.h>
// #include <tbb/tbb.h>

#include <time.h>
#include <iostream>
#include <boost/multi_array.hpp>
#include <Eigen/Core>
#include <pcl/common/common.h>

using namespace Eigen;
using namespace std;
using namespace pcl;

#define SQ(x) ((x)*(x))

int NUM_TH = 1;
using Point3f = PointXYZ;

class LDT
{
public:
    LDT(const PointCloudTPtr& _b, uint32_t _div = 300);
    float Distance(Vector3f& query);

    boost::multi_array<float, 3> b, g, dt, dt2;
private:
    Point3f _min, _max;
    float cellLen;
    uint32_t nCells, div;

    // void PP(boost::multi_array<float, 2>& dt)
    // {
    //     for (int i=0;i<div;i++){
    //         for (int j=0;j<div;j++) cout << dt[i][j] << " ";
    //         cout << "\n";
    //     }
    //     cout << "\n";
    // }

};

LDT::LDT(const PointCloudTPtr& _b, uint32_t _div): div(_div)
{
    b.resize(boost::extents[div][div][div]);
    g.resize(boost::extents[div][div][div]);
    dt.resize(boost::extents[div][div][div]);

    pcl::getMinMax3D(*_b, _min, _max);
    std::cerr << "min:" << _min << std::endl;
    std::cerr << "max:" << _max << std::endl;

    auto _fullLen = std::max( _min[2] - _max[2], std::max(_min[0] - _max[0], _mi[1], _max[1]) );
    cellLen = _fullLen / (float) div;

    auto X = source->getMatrixXfMap(3,4,0).transpose().cast<double>();
    for (int i = 0 ; i < X.cols(); i ++ )
    {
        auto _row = X.block(i,0,1,3);
        auto _x = (int) (_row[0] - _min[0] / cellLen);
        auto _y = (int) (_row[1] - _min[1] / cellLen);
        auto _z = (int) (_row[2] - _min[2] / cellLen);
        g[_x][_y][_z] += 1;
    }

    float infinity = 3 * div;

    auto f = [&](int x, int u, int y, int z)
    {
        return SQ(x-u) + SQ(g[u][y][z]);
    };
    auto Sep = [&](int i, int u, int y, int z)
    {
        return (int) ((SQ(u) - SQ(i) + SQ(g[u][y][z]) - SQ(g[i][y][z]) ) / (2*(u-i))) ;
    };

    omp_set_num_threads(NUM_TH);
    #pragma omp parallel
    {
    #pragma omp for
    for (int x = 0; x < div; x ++ ) for (int z = 0; z < div; z ++)
    {
        //  scan 1
        if (b[x][0][z]) g[x][0][z] = 0;
        else g[x][0][z] = infinity;
        
        for (int y = 1; y < div; y ++ )
        {
            if (b[x][y][z]) g[x][y][z] = 0;
            else g[x][y][z] = 1 + g[x][y-1][z];
        }

        // scan 2
        for (int y = div-2; y >= 0; y -- )
            if (g[x][y+1][z] < g[x][y][z]) g[x][y][z] = 1 + g[x][y+1][z];
    }

    } // omp


    omp_set_num_threads(NUM_TH);
    #pragma omp parallel
    {
    vector<int> s(div), t(div);
    #pragma omp for
    
    for (int z = 0; z < div; z ++) for (int y = 0; y < div; y ++ )
    {
        // vector<int> s(div), t(div);
        auto q = 0; s[0] = 0; t[0] = 0;

        // scan 3 
        for (int u = 1; u < div; u ++ )
        {
            while ( q >= 0 && f(t[q], s[q], y, z) > f(t[q], u, y, z) ) {
                q --;
            }
            if (q < 0) {
                q = 0, s[0] = u;
            }
            else
            {
                int w = 1 + Sep(s[q], u, y, z);
                if (w < div)
                {
                    q ++;
                    s[q] = u;
                    t[q] = w;
                }
            }
        }

        // scan 4 
        for (int u = div-1; u >= 0; u -- )
        {
            dt[u][y][z] = f(u,s[q],y, z);
            if (u == t[q]) q--;
        }
    }

    } // omp

    // PP(dt);

    auto f_z = [&](int z, int u, int y, int x)
    {
        return SQ(z-u) + SQ(g[x][y][u]);
    };
    auto Sep_z = [&](int i, int u, int y, int x)
    {
        return (int) ((SQ(u) - SQ(i) + SQ(g[x][y][u]) - SQ(g[x][y][i]) ) / (2*(u-i))) ;
    };

    vector<int> s(div), t(div);
    for (int x = 0; x < div; x ++ ) for (int y = 0; y < div; y ++ )
    {
        // vector<int> s(div), t(div);
        auto q = 0; s[0] = 0; t[0] = 0;

        // scan 5 
        for (int u = 1; u < div; u ++ )
        {
            while ( q >= 0 && f_z(t[q], s[q], y, x) > f_z(t[q], u, y, x) ) {
                q --;
            }
            if (q < 0) {
                q = 0, s[0] = u;
            }
            else
            {
                int w = 1 + Sep(s[q], u, y, x);
                if (w < div)
                {
                    q ++;
                    s[q] = u;
                    t[q] = w;
                }
            }
        }

        // scan 6 
        for (int u = div-1; u >= 0; u -- )
        {
            dt[x][y][u] = f(u,s[q],y, x);
            if (u == t[q]) q--;
        }
    }

}


float LDT::Distance(Vector3f& query)
{
    int x = query[0], y = query[1], z = query[2];
    return dt[x][y][z];
}

// MatrixXf testa(4,4);
// LDT *ldt;

// // vector<int> vv;
// int main(int argc, char* argv[])
// {
// int _div = atoi(argv[2]);
// // testa.resize(_div,_div);
// // for (int x =0; x < _div; x ++ ) for (int y = 0; y < _div ; y ++ )
// // {
// //     testa(x,y) =  ((rand() % 2 == 0)?1.0f:0.0f);
// // }
// // testa << 1,2,0,0,
// //             0,1,0,0,
// //             0,0,1,0,
// //             0,0,0,1;
// // cout << testa << endl;
// // cout << "\n";


// NUM_TH = atoi(argv[1]);
// clock_t clockBegin, clockEnd;

// clockBegin = clock();
// // ldt = new LDT(testa, _div);

// // NUM_TH = omp_get_max_threads();
// // tbb::parallel_for(0, _div, [&](int i) {
// //     {

// //     }

// // });
// omp_set_num_threads(NUM_TH);
// #pragma omp parallel for
// for (int i = 0; i < _div; i ++ ){

// }


// clockEnd = clock();

//     cout << (double) (clockEnd - clockBegin) / CLOCKS_PER_SEC << "CPU \n" << endl;


// }
