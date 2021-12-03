#pragma once

#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <queue>
#include <vector>
#include "common.hpp"

namespace icp{

const int N = 1e6+10;
int _idx;

struct KDNode
{
    const static int max_dims = 5;
    double featrue[max_dims];
    int index;
    int size;
    // int region[max_dims][2];
    int dim;
    bool operator<(const KDNode& other) const {
        return featrue[_idx] < other.featrue[_idx];
    }
};
typedef std::pair<double, KDNode> D_N;

struct KDTree
{
    int dims;
    KDNode Node[N];
    KDNode data[N<<2];
    int flag[N<<2];
    PointCloudTPtr target;
    Eigen::MatrixXf tar;
    // std::priority_queue<D_N> Q;
    KDTree() {}

    KDTree(Eigen::MatrixXf _tar, int _dims = 3): tar(_tar), dims(_dims) {
        // tar = target->getMatrixXfMap(3,4,0).transpose();
        int n = tar.rows();
        assert( n < N );

        std::cerr << "start build kdtree\n";
       for (int i = 0; i < n; i ++ )
       {
            auto v1 = tar.block<1,3>(i,0).transpose();
            for (int j = 0; j < dims; j ++ )
                Node[i].featrue[j] = v1[j];
            Node[i].index = i;
       }
       build(1,0,n-1,0);
    }


    KDTree(const PointCloudTPtr& cloudptr, int _dims = 3) : dims(_dims),target(cloudptr)
    {
        tar = target->getMatrixXfMap(3,4,0).transpose();
        int n = tar.rows();
        assert( n < N );

        std::cerr << "start build kdtree\n";
       for (int i = 0; i < n; i ++ )
       {
            auto v1 = tar.block<1,3>(i,0).transpose();
            for (int j = 0; j < dims; j ++ )
                Node[i].featrue[j] = v1[j];
            Node[i].index = i;
       }
       build(1,0,n-1,0);
    }

    void build(int o,int l, int r , int dep)
    {
        if (l > r) return;
        _idx = dep%dims;
        int lc = o<<1, rc = o<<1|1;
        flag[o] = 1; flag[lc] = flag[rc] = 0;
        int mid = l + r >> 1;

        std::nth_element(Node+l,Node+mid,Node+r+1);
        data[o] = Node[mid]; data[o].dim = _idx;
        data[o].size = r-l+1;

        build(lc,l,mid-1, dep+1);
        build(rc,mid+1,r, dep+1);
    }

    float Distance(float x0, float x1, float x2)
    {
        // std::cerr << "distance\n";
        std::priority_queue<D_N> Q;
        assert(Q.size() == 0);
        // while (Q.size()) Q.pop();
        KDNode p; 
        p.featrue[0] = x0;
        p.featrue[1] = x1;
        p.featrue[2] = x2;
        k_close(p,1,1, Q);
        assert(Q.size() == 1);
        return std::sqrt(Q.top().first) ;

    }

    int get_closest(const Vector3d& x)
    {

        std::priority_queue<D_N> Q;
        assert(Q.size() == 0);
        // while (Q.size()) Q.pop();
        KDNode p; 
        // std::cerr << x << " ";
        p.featrue[0] = x[0];
        p.featrue[1] = x[1];
        p.featrue[2] = x[2];
        k_close(p,1,1, Q);
        assert(Q.size() == 1);
        return Q.top().second.index;

    }
    void k_close(const KDNode& p, int k, int o, std::priority_queue<D_N>& Q)
    {
        if (!flag[o]) return;
        int dim = data[o].dim;
        int lc = o<<1, rc = o<<1|1;

        if(p.featrue[dim] > data[o].featrue[dim])std::swap(lc,rc);

        std::pair<double,KDNode> cur(.0,data[o]);
        for (int i = 0; i < dims; i ++ ) 
            cur.first += pow(p.featrue[i]-data[o].featrue[i],2);
        // std::cerr << "cur:" << cur.first << "\n";
        if (flag[lc]) k_close(p,k,lc, Q);
        int fg = 0;
        if (Q.size() < k){
            // std::cerr << Q.size() << "||" << k << "\n";
            Q.push(cur); fg = 1;
        }
        else{
            if (cur.first < Q.top().first)
            {
            // std::cerr << "cur2:" << cur.first << "\n";
                Q.pop(); Q.push(cur);
            }
            fg = pow(p.featrue[dim] - data[o].featrue[dim],2) < Q.top().first;
        }
        if (flag[rc] && fg) k_close(p,k,rc, Q);
    }

};

}
