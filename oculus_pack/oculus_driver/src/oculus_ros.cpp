#include <oculus_driver/oculus_ros.h>
#include <oculus_driver/util.h>
#include <oculus_msgs/HMDInfo.h>
#include <iostream>

namespace oculus_driver
{

    OculusRos::OculusRos(ros::NodeHandle& node)
    : info_loaded(false)
    , track_on(false)
    , parent_frame("parent")
    , oculus_frame("oculus")
    , node(node)
    {
        ROS_INFO("Oculus Rift Object Created");
    }

    bool OculusRos::init()
    {
        ovr_Initialize(0);

        ROS_INFO("Oculus Rift System Starting");

        ros::NodeHandle private_node("~");
        private_node.getParam("parent_frame", parent_frame);
        private_node.getParam("oculus_frame", oculus_frame);

        hmd = ovrHmd_Create(0);

        if (hmd)
        {
	    
            ROS_INFO("Oculus Rift Device Manager Running");
	    
	    hmd_info = *hmd;
	    info_loaded = 1;

	    hmd_pub = node.advertise<oculus_msgs::HMDInfo>("/oculus/hmd_info", 10);
            quat_pub = node.advertise<geometry_msgs::Quaternion>("/oculus/quaternion", 10);
            ang_pub = node.advertise<geometry_msgs::Vector3>("/oculus/euler_angles", 10);
            pos_pub = node.advertise<geometry_msgs::Vector3>("/oculus/position", 10);

	    
	    if(!ovrHmd_ConfigureTracking(hmd, ovrTrackingCap_Orientation |
	    				  ovrTrackingCap_MagYawCorrection |
					  ovrTrackingCap_Position, 0))
	    {
		track_on = 0;
		ROS_ERROR("Unable to Detect Rift Head Tracker");
 	    }
	    else 
	    {
		track_on = 1;
		ROS_INFO("Rift Head Tracker Detected");
	    }
	  
	}
        else
        {
            info_loaded = 0;
	    ROS_ERROR("Unable to Start Rift Manager");
	}

	return info_loaded & track_on;
    }

    OculusRos::~OculusRos()
    {
	ROS_INFO("Shutting Down Oculus");
	ovrHmd_Destroy(hmd);
	ovr_Shutdown();
    }

    void OculusRos::publish()
    {
        ros::Time now = ros::Time::now();
        if (info_loaded & track_on)
        {
            oculus_msgs::HMDInfo hmd_msg;
            convertHMDInfoToMsg(hmd_info, hmd_msg);
            hmd_msg.header.stamp = now;
            hmd_pub.publish(hmd_msg);
            
	    // topics
            state = ovrHmd_GetTrackingState(hmd, ovr_GetTimeInSeconds());
	    pose = state.HeadPose.ThePose;
	    
	    // quaternion
            geometry_msgs::Quaternion q_msg;
	    convertQuaternionToMsg(pose.Orientation, q_msg);
            quat_pub.publish(q_msg);

	    // head position
	    geometry_msgs::Vector3 p_msg;
	    convertVectorToMsg(pose.Position, p_msg);
            pos_pub.publish(p_msg);

	    // euler angles
	    ovrVector3f angles;
	    geometry_msgs::Vector3 a_msg;
	    OVR::Quatf quat = (OVR::Quatf)pose.Orientation;
	    quat.GetEulerAngles<
	        OVR::Axis_Y,OVR::Axis_X,OVR::Axis_Z>(
		&angles.y,&angles.x,&angles.z);
	    convertVectorToMsg(angles,a_msg);
            ang_pub.publish(a_msg);
            
	    // tf
            tf::Transform transform;
            transform.setRotation(tf::Quaternion(q_msg.x,
                         q_msg.y,
                         q_msg.z,
                         q_msg.w));
            br.sendTransform(tf::StampedTransform(transform,
                           now,
                           parent_frame,
                           oculus_frame));
        }
    }

} 	// namespace oculus_driver
