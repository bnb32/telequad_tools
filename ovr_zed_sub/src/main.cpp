#include "App.h"

int main(int argc, char** argv) {
    
    ros::init(argc, argv, "ovr_zed_sub");
   
    ROS_INFO_STREAM("------------------------------------------");
    ROS_INFO_STREAM("    TELEQUAD TOOLS: OVR ZED SUBSCRIBER    ");
    ROS_INFO_STREAM("------------------------------------------");

    App instance;

    return instance.execute(argc, argv);
}

