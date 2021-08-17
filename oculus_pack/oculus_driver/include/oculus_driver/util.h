#ifndef __OCULUS_DRIVER_UTIL__
#define __OCULUS_DRIVER_UTIL__

#include <OVR.h>
#include <oculus_msgs/HMDInfo.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Transform.h>

namespace oculus_driver
{
    void convertHMDInfoToMsg(const ovrHmdDesc& info, oculus_msgs::HMDInfo& msg);
    void convertQuaternionToMsg(const ovrQuatf& quat, geometry_msgs::Quaternion& msg);
    void convertVectorToMsg(const ovrVector3f& vec, geometry_msgs::Vector3& msg);
}  // namespace oculus_driver

#endif  // __OCULUS_DRIVER_UTIL__
