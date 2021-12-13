// .. / ,, ()
// s[q]  s[0]

// while if!!

// no SQ with the thread round but diff 2!!

#pragma once

#include <iostream>
#include <Eigen/Core>
#include <boost/multi_array.hpp>

#include "common.hpp"

#include <omp.h>

using namespace std;
using namespace Eigen;
using namespace boost;
using namespace icp;

#define GET_MAX(x,y) ((x<y)?y:x)
#define SQ(x) ((x)*(x))

class LDT{
public:
    LDT(const PointCloudTPtr& , uint32_t _div = 300, int _core = 2);
    float Distance(float, float, float);

    multi_array<float,3> g, dt;
    Point3f _min, _max;
    
    uint32_t div;
    float cellLen;
    int CORES;
};


LDT::LDT(const PointCloudTPtr& _b, uint32_t _div, int _core): div(_div), CORES(_core)
{
    std::cerr << "build LDT" << std::endl;
    double start_time = omp_get_wtime();

    g.resize(boost::extents[div][div][div]);
    dt.resize(boost::extents[div][div][div]);

    int infinity = 3 * div;

    pcl::getMinMax3D(*_b, _min, _max);
    std::cerr << "min:" << _min << std::endl;
    std::cerr << "max:" << _max << std::endl;

    auto _diag = _max - _min;
    auto _fullLen =  GET_MAX( GET_MAX(_diag[0], _diag[1]), _diag[2]) * 1.1;
    
    Point3f mid = (_min + _max ) / 2;
    auto _halfLen = _fullLen/2;
    auto expansion_factor = 2.0f;

    _min.x = mid.x - _halfLen * expansion_factor;
    _min.y = mid.y - _halfLen * expansion_factor;
    _min.z = mid.z - _halfLen * expansion_factor;
    _max.x = mid.x + _halfLen * expansion_factor;
    _max.y = mid.y + _halfLen * expansion_factor;
    _max.z = mid.z + _halfLen * expansion_factor;
    _fullLen = 2 * _halfLen * expansion_factor;
    cellLen = _fullLen / (float) div;
    std::cerr << "cellLen:" << cellLen << std::endl;

    omp_set_num_threads(CORES);
    std::cerr <<  "THREAD NUM: " << omp_get_num_threads() << std::endl;

    #pragma omp parallel for  
    for (int i = 0 ; i < _b->points.size(); i ++ )
    {
        auto _row = _b->points[i];
        auto _x = (int) ( (_row.x - _min.x) / cellLen);
        auto _y = (int) ( (_row.y - _min.y) / cellLen);
        auto _z = (int) ( (_row.z - _min.z) / cellLen);
        // assert(_x < div);
        // assert(_y < div);
        // assert(_z < div);
        // cout << _x << endl;
        // cout << _y << endl;
        // cout << _z << endl;
        dt[_x][_y][_z] ++  ;
    }

    // #pragma omp parallel for schedule (static)
    #pragma omp parallel for
    for (int x = 0; x < div;  x++ )  {
        for (int y = 0; y < div; y ++ )
    {
        if (dt[x][y][0]) g[x][y][0] = 0; 
        else g[x][y][0] = infinity;

        for (int z = 1; z < div; z ++)
        {
            if (dt[x][y][z] ) g[x][y][z] = 0; 
            else g[x][y][z] = g[x][y][z-1] + 1;
        }
        for (int z = div-2; z >= 0; z --)
        {
            if (g[x][y][z+1] < g[x][y][z] ) g[x][y][z] = 1 + g[x][y][z+1];
        }
    }

    }

   auto f_y  = [&](int y, int u, int x, int z)
    {
        return SQ(y-u) + SQ(g[x][u][z]);
    };

    auto Sep_y = [&](int i, int u, int x, int z)
    {
        return (int) ( (SQ(u) - SQ(i) + SQ(g[x][u][z]) - SQ(g[x][i][z])) / (2 * (u - i) ) );

    };

    // #pragma omp parallel for schedule (static)
    #pragma omp parallel for 
    for (int x = 0; x < div; x ++ ) {

    for (int z = 0; z < div; z ++ )
    {
        vector<int> s(div), t(div);
        auto q = 0;
        s[0] = 0, t[0] = 0;

        for (int u = 1; u < div; u ++ ) // loop y
        {
            while (q >= 0 && f_y(t[q],s[q],x,z) > f_y(t[q], u, x,z)) q--;

            if (q < 0){ q = 0, s[0] = u; } //!
            else {
                auto w = 1 + Sep_y(s[q], u, x, z);
                if (w < div){
                    q ++ ;
                    t[q] = w;
                    s[q] = u;
                }
            }

        }

        for (int y = div - 1; y >= 0; y -- )
        {
            auto ans = f_y(y, s[q] , x, z);
            dt[x][y][z] = ans;
            if (x == t[q]) q -- ;
        }

    }

    }


    auto f_x  = [&](int x, int u, int y, int z)
    {
        return SQ(x-u) + (dt[u][y][z]);
    };

    auto Sep_x = [&](int i, int u, int y, int z)
    {
        return (int) ( (SQ(u) - SQ(i) + (dt[u][y][z]) - (dt[i][y][z])) / (2 * (u - i) ) );

    };

    // #pragma omp parallel for schedule (static)
    #pragma omp parallel for 
    for (int y = 0; y < div; y ++ ) 
    {
        for (int z = 0; z < div; z ++ )
    {
        vector<int> s(div), t(div);
        auto q = 0;
        s[0] = 0, t[0] = 0;

        for (int u = 1; u < div; u ++ ) // loop y
        {
            while (q >= 0 && f_x(t[q],s[q],y,z) > f_x(t[q], u, y,z)) q--;

            if (q < 0){ q = 0, s[0] = u; }
            else {
                int w = 1 + Sep_x(s[q], u, y, z);
                if (w < div){
                    q ++ ;
                    t[q] = w;
                    s[q] = u;
                }
            }

        }

        for (int x = div - 1; x >= 0; x -- )
        {
            auto ans = f_x(x, s[q] , y, z);
            g[x][y][z] = std::sqrt(ans);
            if (x == t[q]) q -- ;
        }

    }

    }


    double end_time = omp_get_wtime();
    std::cerr << "build LDT DONE " << end_time - start_time << std::endl;

}

// LDT * ldt;
// int main()
// {
//     MatrixXf M(4,4,4);
//     M << 1, 0, 0, 0,
//                 0, 1, 0, 0,
//                 0,0,1,0,
//                 0,0,0,1;

//     ldt = new LDT(M);

//     for (int i = 0; i < 4; i ++ ) {
//         for (int j = 0; j < 4; j ++ )
//     {
//         cout << ldt->g[i][j] << " ";
//     }
//     cout << "\n";
//     }

// }

    float LDT::Distance(float x, float y, float z)
    {
        double dx, dy, dz;
        dx = dy = dz = 0.0;

        float max_x = _min.x + cellLen * div;
        float max_y = _min.y + cellLen * div;
        float max_z = _min.z + cellLen * div;
        if (x < _min.x) 
        {
            dx = _min.x - x;
            x = _min.x;
        }
        if (x > max_x)
        {
            dx = x - max_x;
            x = max_x;
        }

        if (y < _min.y) 
        {
            dy = _min.y - y;
            y = _min.y;
        }
        if (y > max_y)
        {
            dy = y - max_y;
            y = max_y;
        }

        if (z < _min.z) 
        {
            dz = _min.z - z;
            z = _min.z;
        }
        if (z > max_z)
        {
            dz = z - max_z;
            z = max_z;
        }

        x = (int) ((x - _min.x) / cellLen);
        y = (int) ((y - _min.y) / cellLen);
        z = (int) ((z - _min.z) / cellLen);


        // if (x < 0) x = 0;
        // if (x > div-1) x = div - 1;
        // if (y < 0) y = 0;
        // if (y > div-1) y = div - 1;
        // if (z < 0) z = 0;
        // if (z > div-1) z = div - 1;

        return g[x][y][z] * cellLen + std::sqrt(dx*dx + dy*dy + dz*dz);

    }