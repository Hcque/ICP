// .. / ,, ()
// s[q]  s[0]

// while if!!

// no SQ

#pragma once

#include <iostream>
#include <Eigen/Core>
#include <boost/multi_array.hpp>

#include "common.hpp"

using namespace std;
using namespace Eigen;
using namespace boost;
using namespace icp;

#define GET_MAX(x,y) ((x<y)?y:x)
#define SQ(x) ((x)*(x))

class LDT{
public:
    LDT(const PointCloudTPtr& , uint32_t _div = 100);
    float Distance(float, float, float);

    multi_array<float,3> g, dt;
    Point3f _min, _max;
    
    uint32_t div;
    float cellLen;
};


LDT::LDT(const PointCloudTPtr& _b, uint32_t _div): div(_div)
{
    g.resize(boost::extents[div][div][div]);
    dt.resize(boost::extents[div][div][div]);

    int infinity = 3 * div;

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
        dt[_x][_y][_z] ++  ;
    }

    for (int x = 0; x < div;  x++ )  for (int y = 0; y < div; y ++ )
    {
        if (dt[x][y][0]) g[x][y][0] = 0; //TODO
        else g[x][y][0] = infinity;

        for (int z = 1; z < div; z ++)
        {
            if (dt[x][y][z] ) g[x][y][z] = 0; //TODO
            else g[x][y][z] = g[x][y][z-1] + 1;
        }
        for (int z = div-2; z >= 0; z --)
        {
            if (g[x][y][z+1] < g[x][y][z] ) g[x][y][z] = 1 + g[x][y][z+1];
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

    for (int x = 0; x < div; x ++ ) for (int z = 0; z < div; z ++ )
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


    auto f_x  = [&](int x, int u, int y, int z)
    {
        return SQ(x-u) + (dt[u][y][z]);
    };

    auto Sep_x = [&](int i, int u, int y, int z)
    {
        return (int) ( (SQ(u) - SQ(i) + (dt[u][y][z]) - (dt[i][y][z])) / (2 * (u - i) ) );

    };

    for (int y = 0; y < div; y ++ ) for (int z = 0; z < div; z ++ )
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

    //     // Third scan
    // // This scan is almost identical to scan2.
    // // Takes g_scan as input, and grid as the output
    // auto f_x = [&](int yCoord, int zCoord, int xCoord, int xpos) {
    //     return SQ(xCoord - xpos) + dt[xpos][yCoord][zCoord];
    // };

    // auto Sep_x = [&](int yCoord, int zCoord, int i, int u) {
    //     assert(u > i);
    //     return int((SQ(u) - SQ(i) + dt[u][yCoord][zCoord] - dt[i][yCoord][zCoord]) / (2 * (u - i)));
    // };

    // // tbb::parallel_for(0u, div, [&](int y) {
    // for (int y = 0u; y < div; y ++ ){
    //     for (auto z = 0; z < div; z++)
    //     {
    //         std::vector<int> s(div), t(div);
    //         auto q = 0;
    //         s[0] = t[0] = 0;

    //         for (auto u = 1; u < div; u++)
    //         {
    //             while ((q >= 0) && (f_x(y, z, t[q], s[q]) > f_x(y, z, t[q], u)))
    //                 q--;

    //             if (q < 0)
    //             {
    //                 q = 0;
    //                 s[0] = u;
    //             }
    //             else
    //             {
    //                 auto w = 1 + Sep_x(y, z, s[q], u);
    //                 if (w < div)
    //                 {
    //                     q++;
    //                     s[q] = u;
    //                     t[q] = w;
    //                 }
    //             }
    //         }

    //         for (int u = div - 1; u >= 0; u--)
    //         {
    //             auto point = sqrt(f_x(y, z, u, s[q]));
    //             g[u][y][z] = point;
    //             if (u == t[q])
    //                 q--;
    //         }
    //     }
    // }



//    // Second scan
//     // In this scan, grid is the input, and g_scan is the output.
//     auto f_y = [&](int xCoord, int zCoord, int yCoord, int ypos) {
//         return SQ(yCoord - ypos) + SQ(g[xCoord][ypos][zCoord]);
//     };

//     auto Sep_y = [&](int xCoord, int zCoord, int i, int u) {
//         assert(u > i);
//         return int((SQ(u) - SQ(i) + SQ(g[xCoord][u][zCoord]) - SQ(g[xCoord][i][zCoord])) / (2 * (u - i)));
//     };

//     for (int x = 0; x < div; x ++ ) {

//         for (auto z = 0; z < div; z++)
//         {
//             std::vector<int> s(div), t(div);
//             auto q = 0;
//             s[0] = t[0] = 0;

//             for (auto u = 1; u < div; u++)
//             {
            
//                 while ((q >= 0) && (f_y(x, z, t[q], s[q]) > f_y(x, z, t[q], u)))
//                     q--;

//                 if (q < 0)
//                 {
//                     q = 0;
//                     s[0] = u;
//                 }
//                 else
//                 {
//                     auto w = 1 + Sep_y(x, z, s[q], u);
//                     if (w < div)
//                     {
//                         q++;
//                         s[q] = u;
//                         t[q] = w;
//                     }
//                 }
//             }

//             for (int u = div - 1; u >= 0; u--)
//             {
//                 auto point = f_y(x, z, u, s[q]);
//                 dt[x][u][z] = point;
//                 if (u == t[q])
//                     q--;
//             }
//         }
//     }
 

    // // Third scan
    // // This scan is almost identical to scan2.
    // // Takes g_scan as input, and grid as the output
    // auto f_x = [&](int yCoord, int zCoord, int xCoord, int xpos) {
    //     return SQ(xCoord - xpos) + dt[xpos][yCoord][zCoord];
    // };

    // auto Sep_x = [&](int yCoord, int zCoord, int i, int u) {
    //     assert(u > i);
    //     return int((SQ(u) - SQ(i) + dt[u][yCoord][zCoord] - dt[i][yCoord][zCoord]) / (2 * (u - i)));
    // };

    // // tbb::parallel_for(0u, div, [&](int y) {
    // for (int y = 0u; y < div; y ++ ){
    //     for (auto z = 0; z < div; z++)
    //     {
    //         std::vector<int> s(div), t(div);
    //         auto q = 0;
    //         s[0] = t[0] = 0;

    //         for (auto u = 1; u < div; u++)
    //         {
    //             while ((q >= 0) && (f_x(y, z, t[q], s[q]) > f_x(y, z, t[q], u)))
    //                 q--;

    //             if (q < 0)
    //             {
    //                 q = 0;
    //                 s[0] = u;
    //             }
    //             else
    //             {
    //                 auto w = 1 + Sep_x(y, z, s[q], u);
    //                 if (w < div)
    //                 {
    //                     q++;
    //                     s[q] = u;
    //                     t[q] = w;
    //                 }
    //             }
    //         }

    //         for (int u = div - 1; u >= 0; u--)
    //         {
    //             auto point = sqrt(f_x(y, z, u, s[q]));
    //             g[u][y][z] = point;
    //             if (u == t[q])
    //                 q--;
    //         }
    //     }
    // }


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
        x = (int) ((x - _min.x) / cellLen);
        y = (int) ((y - _min.y) / cellLen);
        z = (int) ((z - _min.z) / cellLen);

        if (x < 0) x = 0;
        if (x > div-1) x = div - 1;
        if (y < 0) y = 0;
        if (y > div-1) y = div - 1;
        if (z < 0) z = 0;
        if (z > div-1) z = div - 1;

        return g[x][y][z] * cellLen;

    }