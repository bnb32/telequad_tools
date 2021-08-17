# %Tag(FULLTEXT)%
cmake_minimum_required(VERSION 2.8.3)
project(zed_driver)

# INCLUDES

set(zed_INC_PATH ${zed_driver_SOURCE_DIR}/include)
set(zed_INC_DIRS ${zed_INC_PATH})

include_directories(${zed_INC_DIRS})

# ZED SDK

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(LINUX_ONLY_LIBRARIES "")

SET(ZED_DIR "/usr/local/zed/")
find_package(ZED 0.9.2 REQUIRED)
#find_package(GLUT)

SET(LINUX_ONLY_LIBRARIES pthread npps nppi)

find_package(CUDA 6.5 EXACT REQUIRED)
#find_package(OpenGL)

include_directories(${ZED_INCLUDE_DIRS})
#include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
#link_directories(${GLUT_LIBRARY_DIRS})
#link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
#link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)


# CATKIN

find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  std_msgs 
  genmsg
  image_transport
  tf
)

## Declare ROS messages and services
#add_message_files(FILES Num.msg)
#add_service_files(FILES AddTwoInts.srv)

## Generate added messages and services
#generate_messages(DEPENDENCIES std_msgs)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS
    std_msgs
    sensor_msgs
    geometry_msgs
    image_transport
)

include_directories(include ${catkin_INCLUDE_DIRS})

# SOURCES

set(zed_SRC
    ${zed_driver_SOURCE_DIR}/src/zed_driver.cpp
)

set(zed_HDR
    ${zed_INC_PATH}/zed_driver.h
)    

add_executable(zed_driver_node src/zed_driver_node.cpp ${zed_SRC} ${zed_HDR})

add_definitions(-std=c++0x)#  -m64)

target_link_libraries(zed_driver_node 
    ${ZED_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${LINUX_ONLY_LIBRARIES}
    #${OPENGL_LIBRARIES}
    #${GLUT_LIBRARY}
    ${catkin_LIBRARIES}       
)

#add_dependencies(talker zed_driver_generate_messages_cpp)

add_executable(listener src/listener.cpp)
target_link_libraries(listener ${catkin_LIBRARIES})
add_dependencies(listener zed_driver_generate_messages_cpp)

# %EndTag(FULLTEXT)%
