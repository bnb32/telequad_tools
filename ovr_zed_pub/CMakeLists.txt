cmake_minimum_required(VERSION 2.4)
project(ovr_zed_pub)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

find_package(ZED REQUIRED)
find_package(CUDA 6.5 REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${catkin_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

find_package(catkin REQUIRED COMPONENTS roscpp geometry_msgs message_generation image_transport)

catkin_package(
    CATKIN_DEPENDS roscpp geometry_msgs image_transport
)

add_executable(ovr_zed_pub src/main.cpp src/App.cpp)

add_definitions(-std=c++11)  

target_link_libraries(ovr_zed_pub
                        ${ZED_LIBRARIES}
			${CUDA_LIBRARIES} 
			${CUDA_nppi_LIBRARY} 
			${CUDA_npps_LIBRARY}
			${catkin_LIBRARIES}
		    )

#############
## Install ##
#############

install(TARGETS ovr_zed_pub 
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
