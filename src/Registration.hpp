


#include "common.hpp"


namespace icp
{

struct ICP_res
{
    Eigen::MatrixXd resN3;
    Matrix4d transMat;
};


class Registration
{
    public:

    virtual void setTarget(PointCloudTPtr _target) = 0;
    virtual ICP_res registration(Matrix4d&) = 0;

};


}