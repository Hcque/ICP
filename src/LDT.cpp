// = or ==

// revise of f_z
// wrong init: 0? 
// still wrong init

// clamped

// print does not help much


#pragma once

#include "common.hpp"
#include <omp.h>
// #include <tbb/tbb.h>

#include <time.h>
#include <iostream>
#include <algorithm>
#include <boost/multi_array.hpp>
#include <Eigen/Core>
#include <pcl/common/common.h>

using namespace Eigen;
using namespace std;
using namespace pcl;
using namespace icp;

#define SQ(x) ((x)*(x))

// static int NUM_TH = 1;

class LDT_prev
{
public:
    LDT_prev(const PointCloudTPtr& _b, uint32_t _div = 300);
    float Distance(float, float, float);

    boost::multi_array<float, 3> b, g, dt, dt2;
    float cellLen;
private:
    Point3f _min, _max;
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

#define GET_MAX(x,y) ((x<y)?y:x)
LDT_prev::LDT_prev(const PointCloudTPtr& _b, uint32_t _div): div(_div)
{
    float infinity = 3 * div;

    b.resize(boost::extents[div][div][div]);
    g.resize(boost::extents[div][div][div]);

    // std::fill_n(g.data(), g.num_elements(), infinity);
    // std::cout << g[0][0][0] << "\n";

    dt.resize(boost::extents[div][div][div]);
    dt2.resize(boost::extents[div][div][div]);

    pcl::getMinMax3D(*_b, _min, _max);
    std::cerr << "min:" << _min << std::endl;
    std::cerr << "max:" << _max << std::endl;

    auto _diag = _max - _min;
    auto _fullLen =  GET_MAX( GET_MAX(_diag[0], _diag[1]), _diag[2]) * 1.3;
    // auto _fullLen = 1;
    cellLen = _fullLen / (float) div;
    std::cerr << "cellLen:" << cellLen << std::endl;

    for (int i = 0 ; i < _b->points.size(); i ++ )
    {
        auto _row = _b->points[i];
        auto _x = (int) ( (_row.x - _min.x) / cellLen);
        auto _y = (int) ( (_row.y - _min.y) / cellLen);
        auto _z = (int) ( (_row.z - _min.z) / cellLen);
        assert(_x < div);
        assert(_y < div);
        assert(_z < div);
        // cout << _x << endl;
        // cout << _y << endl;
        // cout << _z << endl;
        b[_x][_y][_z] ++  ;
    }


    // keep y, z constant
    auto f = [&](int x, int u, int y, int z) 
    {
        return SQ(x-u) + SQ(g[u][y][z]);
    };
    auto Sep = [&](int i, int u, int y, int z)
    {
        return (int) ((SQ(u) - SQ(i) + SQ(g[u][y][z]) - SQ(g[i][y][z]) ) / (2*(u-i))) ;
    };

    // omp_set_num_threads(NUM_TH);
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


    // omp_set_num_threads(NUM_TH);
    #pragma omp parallel
    {
    #pragma omp for
    
    for (int z = 0; z < div; z ++) {

        vector<int> s(div), t(div);
    
    for (int y = 0; y < div; y ++ )
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
    }

    } // omp

    // PP(dt);

    auto f_z = [&](int z, int u, int y, int x)
    {
        return SQ(z-u) + SQ(dt[x][y][u]);
    };
    auto Sep_z = [&](int i, int u, int y, int x)
    {
        return (int) ((SQ(u) - SQ(i) + SQ(dt[x][y][u]) - SQ(dt[x][y][i]) ) / (2*(u-i))) ;
    };

    // vector<int> s(div), t(div);
    for (int x = 0; x < div; x ++ ) {
        

    vector<int> s(div), t(div);

        for (int y = 0; y < div; y ++ )
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
                int w = 1 + Sep_z(s[q], u, y, x);
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
            dt2[x][y][u] =  std::sqrt(f_z(u,s[q],y, x) );
            if (u == t[q]) q--;
        }
    }

    }


    // for (int i = 0; i < div; i ++ ) for (int j = 0; j < div; j ++ ) for (int k = 0; k < div; k ++ )
    // {
    //     cout << dt2[i][j][k] << " ";
    // }

    // cout << "\n";



}


float LDT_prev::Distance(float x, float y, float z)
{
    float res_x, res_y, res_z;
    res_x = res_y = res_z = 0.0f;

    // auto clamp = [&](const Point3f& _min, const Point3f& _max, float &x, float& res_x)
    // {
    //     if (x < _min.x) {res_x = _min.x - x; x = _min.x; }
    //     if (x > _max.x) {res_x = x - _max.x; x = _max.x; }
    // };

    // clamp(_min, _max, x, res_x);
    // clamp(_min, _max, y, res_y);
    // clamp(_min, _max, z, res_z);
    if (x < _min.x) {res_x = _min.x - x; x = _min.x; }
    if (x > _max.x) {res_x = x - _max.x; x = _max.x; }
    if (y < _min.y) {res_y = _min.y - y; y = _min.y; }
    if (y > _max.y) {res_y = y - _max.y; y = _max.y; }
    if (z < _min.z) {res_z = _min.z - z; z = _min.z; }
    if (z > _max.z) {res_z = z - _max.z; z = _max.z; }

    x = (int) ((x-_min.x) / cellLen ) ;
    y = (int) ((y-_min.y) / cellLen ) ;
    z = (int) ((z-_min.z) / cellLen ) ;

    assert(x >= 0 && x < div);
    assert(y >= 0 && y < div);
    assert(z >= 0 && z < div);

    float added_norm = std::sqrt(res_x*res_x + res_y*res_y + res_z*res_z);
    // std::cout << "added_norm: " << added_norm << "\n";
    return  (dt2[x][y][z]) * cellLen ; // + added_norm;
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
