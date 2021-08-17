#include <oculus_driver/util.h>
#include "std_msgs/String.h"

namespace oculus_driver
{
    void convertHMDInfoToMsg(const ovrHmdDesc& info, oculus_msgs::HMDInfo& msg)
    {
        msg.display_device_name = info.DisplayDeviceName;
        msg.product_name = info.ProductName;
        msg.manufacturer = info.Manufacturer;
        msg.horizontal_resolution = info.Resolution.w;
        msg.vertical_resolution = info.Resolution.h;
        msg.displayId = info.DisplayId;
	msg.version = ovr_GetVersionString();
    }

    void convertQuaternionToMsg(const ovrQuatf& quat,geometry_msgs::Quaternion& msg)
    {
        msg.x = quat.x;
        msg.y = quat.y;
        msg.z = quat.z;
        msg.w = quat.w;
    }

    void convertVectorToMsg(const ovrVector3f& vec,geometry_msgs::Vector3& msg)
    {
        msg.x = vec.x;
        msg.y = vec.y;
        msg.z = vec.z;
    }


}  // namespace oculus_ros
