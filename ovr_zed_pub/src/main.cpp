#include "ros/ros.h"
#include <stdlib.h>
#include "std_msgs/String.h"
#include "App.h"
#include <sstream>

int main(int argc, char** argv) {
    
    ros::init(argc, argv, "ovr_zed_pub");

    ROS_INFO_STREAM("------------------------------------");
    ROS_INFO_STREAM("   TELEQUAD TOOLS: ZED PUBLISHER    ");
    ROS_INFO_STREAM("------------------------------------");
   
    App instance;

    bool ready = instance.init();

    if(ready)
        instance.start();
    else
	return(EXIT_FAILURE);

    while (ros::ok())
    {
        ros::Duration(1).sleep();
	ros::spinOnce();
    }	

    ROS_INFO_STREAM("... shutting down complete");
    
    return(EXIT_SUCCESS);

}
