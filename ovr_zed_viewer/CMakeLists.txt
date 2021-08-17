cmake_minimum_required(VERSION 2.4)
project(ovr_zed_viewer)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(OVR_DIR "${ros_ovr_sdk_SOURCE_DIR}/sdk/ovr_sdk_linux_0.5.0.1")
SET(OVR_KSRC_DIR "${OVR_DIR}/LibOVRKernel/Src")
SET(OVR_COMMON_DIR "${OVR_DIR}/Samples/CommonSrc")
SET(OVR_INC_DIR "${OVR_DIR}/LibOVR/Include")
SET(OVR_SRC_DIR "${OVR_DIR}/LibOVR/Src")

include_directories(${OVR_COMMON_DIR}
		    ${OVR_KSRC_DIR} 
		    ${OVR_INC_DIR}
		    ${OVR_SRC_DIR})

find_package(ZED REQUIRED)
find_package(GLUT)
find_package(GLEW)
find_package(CUDA 6.5 REQUIRED)
find_package(SDL2 REQUIRED)

# Mandatory
find_package(OpenGL)
find_package(OpenCV 2.4 REQUIRED COMPONENTS core highgui imgproc)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/shaders)
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${SDL2_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${SDL2_LIBRARY_DIRS})

find_package(catkin REQUIRED COMPONENTS roscpp geometry_msgs message_generation tf)

catkin_package(
    CATKIN_DEPENDS roscpp geometry_msgs tf
)

#add_library(zed_viewer_lib src/zed_viewer_lib.cpp)

add_executable(ovr_zed_viewer src/main.cpp src/App.cpp src/util.cpp)

add_definitions(-std=c++11)  

target_link_libraries(ovr_zed_viewer
                        ${ZED_LIBRARIES}
			${GLEW_LIBRARIES}                       
                        ${OPENGL_LIBRARIES} 
		        ${OpenCV_LIBRARIES}
                        ${GLUT_LIBRARY} 			
			${CUDA_LIBRARIES} 
			${CUDA_nppi_LIBRARY} 
			${CUDA_npps_LIBRARY}
			OVR udev Xinerama
			dl m pthread
			${SDL2_LIBRARY}
			${catkin_LIBRARIES}
		    )

#############
## Install ##
#############

install(TARGETS ovr_zed_viewer 
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
