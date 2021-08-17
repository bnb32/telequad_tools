#include <stdlib.h>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "zed_driver.h"
#include <sstream>

int main(int argc, char **argv)
{
 
  ros::init(argc, argv, "talker");

  ROS_INFO_STREAM("---------------------------------------");
  ROS_INFO_STREAM("   TELEQUAD TOOLS: ZED DRIVER NODE     ");
  ROS_INFO_STREAM("---------------------------------------");

  ZedDriver driver;

  bool ready = driver.init();

  if(ready)
      driver.start();
  else
      return(EXIT_FAILURE);
      
  while (ros::ok())
  {
      ros::Duration(1).sleep();
      ros::spinOnce();
  }

  ROS_INFO_STREAM("... shutting down complete.");

  return(EXIT_SUCCESS);
}
// %EndTag(FULLTEXT)%
