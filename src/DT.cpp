#pragma once

#include <omp.h>

#include "common.hpp"
#include <vector>
#include "kdtree.cpp"

namespace icp{

const int SZ = 100;

class DT
{
public:

    DT(PointCloudTPtr& cloudptr, int _dims = 3 ): kdt(cloudptr, _dims), cloudptr(cloudptr), _dims(_dims)
    {
        // kdt = new KDTree(cloudptr, _dim);
        int Nd = cloudptr->points.size();

        for(int i = 0; i < Nd; i ++) {
            xmin = std::min(xmin, (double)cloudptr->points[i].x);
            xmax = std::max(xmax, (double)cloudptr->points[i].x);
            ymin = std::min(ymin, (double)cloudptr->points[i].y);
            ymax = std::max(ymax, (double)cloudptr->points[i].y);
            zmin = std::min(zmin, (double)cloudptr->points[i].z);
            zmax = std::max(zmax, (double)cloudptr->points[i].z);
        }
        float max_range = std::max(xmax-xmin, std::max(zmax-zmin, ymax-ymin));
        scale = max_range / SZ;
        std::cerr << "scale:" << scale << "\n";
        
        // #pragma omp parallel
        // {
        // #pragma omp for
        for (int i = 0; i < SZ; i ++ )
        {
            if (i % 10 == 0) std::cerr << i << "\n";
            clock_t c_start = clock();
            for (int j = 0; j < SZ; j ++ )
            {
                // if (j % 100 == 0) std::cerr << j << " ";
                for (int k = 0; k < SZ; k ++ )
                {
                    // if (k % 10 == 0) std::cerr << k << " ";
                    grid[i][j][k] = kdt.Distance(i*scale,j*scale,k*scale);
                }
            }
            std::cerr << "each CPU time: " << (double)(clock()-c_start)/CLOCKS_PER_SEC  << "\n";
        }

        // } // omp

    }

    double Distance(double _x, double _y, double _z)
    {
        double res = 0.0;
        if (_x < xmin) 
        {
            res += xmin - _x;
            _x = xmin;
        } 
        else if (_x > xmax)
        {
            res += _x - xmax;
            _x = xmax;
        }

        if (_y < ymin) 
        {
            res += ymin - _y;
            _y = ymin;
        } 
        else if (_y > ymax)
        {
            res += _y - ymax;
            _y = ymax;
        }

        if (_z < zmin) 
        {
            res += zmin - _z;
            _z = zmin;
        } 
        else if (_z > zmax)
        {
            res += _z - zmax;
            _z = zmax;
        }

        int x = (_x - xmin) / scale;
        int y = (_y - ymin) / scale;
        int z = (_z - zmin) / scale;
        return grid[x][y][z];


    }

    // std::vector<double> X, Y, Z;
    // int nCell;
    // double cell_len;
    double scale;
    double xmin = 1e9, xmax = 0;
    double ymin = 1e9, ymax = 0;
    double zmin = 1e9, zmax = 0;

    // int SZ;
    PointCloudTPtr& cloudptr; int _dims;
    KDTree kdt;
    double grid[SZ][SZ][SZ];

};


}