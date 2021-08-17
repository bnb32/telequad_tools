#ifndef __OCULUS_DRIVER_OCULUS_ROS__
#define __OCULUS_DRIVER_OCULUS_ROS__

#include <OVR.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

namespace oculus_driver
{

    class OculusRos
    {
        public:
            explicit OculusRos(ros::NodeHandle& node);
            virtual bool init();
            virtual void publish();
            virtual ~OculusRos();
        private:
            bool info_loaded;
	    bool track_on;
	    float eyeHeight{OVR_DEFAULT_EYE_HEIGHT};
	    float ipd{OVR_DEFAULT_IPD};
            std::string parent_frame;
            std::string oculus_frame;
            ros::NodeHandle node;
            ovrHmd hmd;
            ovrTrackingState state;
	    ovrPosef pose;
            ovrHmdDesc hmd_info;
	    ovrEyeRenderDesc eye_info[2];
            ovrVector3f eyeOffsets[2];
	    ovrMatrix4f eyeProj[2];
            ros::Publisher quat_pub;
            ros::Publisher hmd_pub;
            ros::Publisher pos_pub;
            ros::Publisher ang_pub;
            tf::TransformBroadcaster br;
    };

}  // namespace oculus_driver

#endif  // __OCULUS_DRIVER_OCULUS_ROS__
