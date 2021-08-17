cmake_minimum_required(VERSION 2.8.3)
project(oculus_driver)

find_package(catkin REQUIRED COMPONENTS roscpp geometry_msgs message_generation tf oculus_msgs)

#######################################
## Declare ROS messages and services ##
#######################################

#set(ros_ovr_sdk_SOURCE_DIR "/home/bnb/catkin_ws/src/ros_ovr_sdk")

set(OVR_DIR ${ros_ovr_sdk_SOURCE_DIR}/sdk/ovr_sdk_linux_0.5.0.1)
set(OVR_INC_PATH ${OVR_DIR}/LibOVR/Include)
set(OVR_SRC_PATH ${OVR_DIR}/LibOVR/Src)
set(OVR_KSRC_PATH ${OVR_DIR}/LibOVRKernel/Src)

###################################
## catkin specific configuration ##
###################################
catkin_package(
   INCLUDE_DIRS include ${OVR_INC_PATH} ${OVR_KSRC_PATH}
   LIBRARIES oculus_driver
   CATKIN_DEPENDS roscpp geometry_msgs tf oculus_msgs
#  DEPENDS system_lib
)

include_directories(include
  ${OVR_INC_PATH}
  ${OVR_KSRC_PATH}
  ${OVR_SRC_PATH}
  ${catkin_INCLUDE_DIRS}
  )


link_directories(${OVR_DIR}/LibOVR/Lib/Linux/x86_64/Debug)
link_directories(${OVR_DIR}/LibOVRKernel/Lib/Linux/x86_64/Debug)

add_library(oculus_ros
  src/oculus_ros.cpp
  src/util.cpp
  )

add_definitions(-std=c++11)  

add_executable(oculus_node src/oculus_node.cpp)
add_dependencies(oculus_ros oculus_ros_gencpp oculus_msgs_gencpp)
add_dependencies(oculus_node oculus_ros)

target_link_libraries(oculus_ros udev Xinerama)
target_link_libraries(oculus_ros OVR GL GLEW ${catkin_LIBRARIES})
target_link_libraries(oculus_node oculus_ros)


#############
## Install ##
#############

install(TARGETS oculus_ros oculus_node
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
catkin_add_gtest(util-test test/util-test.cpp)
if(TARGET util-test)
  target_link_libraries(util-test oculus_ros)
endif()
