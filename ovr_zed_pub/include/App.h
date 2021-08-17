#ifndef _APP_H_
#define _APP_H_

#include <zed/Camera.hpp>
#include <pthread.h>
#include <list>
#include <csignal>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

class App {
	private:
		static bool stopping;

                sl::zed::Camera* zed;

		sl::zed::Mat gpuImgLeft;
		sl::zed::Mat gpuImgRight;

		// ros image subscribe
		ros::NodeHandle handle;
		image_transport::ImageTransport rightImgTr;
		image_transport::ImageTransport leftImgTr;
		image_transport::CameraPublisher rightImgPub;
		image_transport::CameraPublisher leftImgPub;
		sensor_msgs::Image rightImgMsg;
		sensor_msgs::Image leftImgMsg;
		sensor_msgs::CameraInfo rightImgInfo;
		sensor_msgs::CameraInfo leftImgInfo;
		sensor_msgs::RegionOfInterest rightROI;
		sensor_msgs::RegionOfInterest leftROI;
		std_msgs::Header leftImgHeader;
		std_msgs::Header rightImgHeader;

		pthread_t rosThread;	
	
	private:
		void* run();
		void initInfo(sl::zed::StereoParameters* params);
		void updateView();
		static void sighandler(int signo)
		{
		    App::stopping = (signo == SIGINT);
		}

	public:
		App();
		~App();
		void releaseCamera();
		bool init();
		void start();
		static void* callRunROS(void *arg) {return ((App*)arg)->run(); }

};

#endif
