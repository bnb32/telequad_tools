//==============================================================================
#include "App.h"
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/distortion_models.h>

bool App::stopping = false;

App::App() 
    : leftImgTr(handle)
    , rightImgTr(handle)
    , zed(NULL)
{
    struct sigaction sigAct;
    memset( &sigAct, 0, sizeof(sigAct) );
    sigAct.sa_handler = App::sighandler;
    sigaction(SIGINT, &sigAct, 0);
    
    rightImgPub = rightImgTr.advertiseCamera("right/image", 1, false);
    leftImgPub = leftImgTr.advertiseCamera("left/image", 1, false);

}

App::~App()
{
    releaseCamera();
}

bool App::init() 
{
    releaseCamera();

   //Enable zed cam 
    zed = new sl::zed::Camera(sl::zed::VGA, 60);
    sl::zed::ERRCODE err = zed->init(sl::zed::MODE::PERFORMANCE, -1, true);
    
    if(err != sl::zed::SUCCESS) {
        
	releaseCamera();
	
	ROS_ERROR_STREAM("ZED Init Err: " << sl::zed::errcode2str(err));
	return false;
    }
 
    return true;
}

void App::releaseCamera()
{
    if( !stopping )
        stopping = true;

    if( zed)
        delete zed;
    zed = NULL;	
}

void App::initInfo( sl::zed::StereoParameters* params)
{
    rightImgInfo.distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;
    leftImgInfo.distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;
    
    rightImgInfo.D.resize(5);
    rightImgInfo.D[0] = 0.0;
    rightImgInfo.D[1] = 0.0;
    rightImgInfo.D[2] = 0.0;
    rightImgInfo.D[3] = 0.0;
    rightImgInfo.D[4] = 0.0;
    rightImgInfo.K.fill(0.0);
    rightImgInfo.K[0] = params->RightCam.fx;
    rightImgInfo.K[2] = params->RightCam.cx;
    rightImgInfo.K[4] = params->RightCam.fy;
    rightImgInfo.K[5] = params->RightCam.cy;
    rightImgInfo.K[8] = 1.0;
    rightImgInfo.R.fill( 0.0);
    rightImgInfo.P.fill( 0.0);
    rightImgInfo.P[0] = params->RightCam.fx;
    rightImgInfo.P[2] = params->RightCam.cx;
    rightImgInfo.P[3] = -params->RightCam.fx * params->baseline;
    rightImgInfo.P[5] = params->RightCam.fy;
    rightImgInfo.P[6] = params->RightCam.cy;
    rightImgInfo.P[10] = 1.0;

    rightImgInfo.width = zed->getImageSize().width;
    rightImgInfo.height = zed->getImageSize().height;
   
    leftImgInfo.D.resize(5);
    leftImgInfo.D[0] = 0.0;
    leftImgInfo.D[1] = 0.0;
    leftImgInfo.D[2] = 0.0;
    leftImgInfo.D[3] = 0.0;
    leftImgInfo.D[4] = 0.0;
    leftImgInfo.K.fill(0.0);
    leftImgInfo.K[0] = params->LeftCam.fx;
    leftImgInfo.K[2] = params->LeftCam.cx;
    leftImgInfo.K[4] = params->LeftCam.fy;
    leftImgInfo.K[5] = params->LeftCam.cy;
    leftImgInfo.K[8] = 1.0;
    leftImgInfo.R.fill( 0.0);
    leftImgInfo.P.fill( 0.0);
    leftImgInfo.P[0] = params->LeftCam.fx;
    leftImgInfo.P[2] = params->LeftCam.cx;
    leftImgInfo.P[5] = params->LeftCam.fy;
    leftImgInfo.P[6] = params->LeftCam.cy;
    leftImgInfo.P[10] = 1.0;
    
    leftImgInfo.width = zed->getImageSize().width;
    leftImgInfo.height = zed->getImageSize().height;


}

void App::updateView()
{
    gpuImgLeft = zed->getView_gpu(sl::zed::VIEW_MODE::STEREO_LEFT);

    //leftImgHeader.stamp = ros::Time::now();

    leftImgMsg.step = gpuImgLeft.step;
    leftImgMsg.height = gpuImgLeft.height;
    leftImgMsg.width = gpuImgLeft.width;
    
    leftImgMsg.data.resize( gpuImgLeft.height * gpuImgLeft.step );

    cudaMemcpy( (uint8_t*)(&leftImgMsg.data[0]), (uint8_t*)(&gpuImgLeft.data[0]), gpuImgLeft.height * gpuImgLeft.step, cudaMemcpyDeviceToHost);
    
    gpuImgRight = zed->getView_gpu(sl::zed::VIEW_MODE::STEREO_RIGHT);
    
    //rightImgHeader.stamp = ros::Time::now();
    
    rightImgMsg.step = gpuImgRight.step;
    rightImgMsg.height = gpuImgRight.height;
    rightImgMsg.width = gpuImgRight.width;
    
    rightImgMsg.data.resize( gpuImgRight.height * gpuImgRight.step );
    
    cudaMemcpy( (uint8_t*)(&rightImgMsg.data[0]), (uint8_t*)(&gpuImgRight.data[0]), gpuImgRight.height * gpuImgRight.step, cudaMemcpyDeviceToHost);

}

void* App::run()
{
    if( !zed)
    {
        ROS_ERROR_STREAM("Camera not initialized");
	return 0;
    }

    static int frameCnt = 0;
    stopping = false;

    sl::zed::StereoParameters* params = zed->getParameters();
    initInfo(params);

    while(1) { //ros::ok()) {

	if(stopping) break;

	if(!zed) 
	{
	    ROS_ERROR_STREAM("Camera not initialized or stopped. Closing");
	    break;
	}

	ros::Time acqStart = ros::Time::now();
	frameCnt++;

	rightImgHeader.frame_id = "right";
	leftImgHeader.frame_id = "left";
	rightImgHeader.seq = leftImgHeader.seq = frameCnt;
	rightImgHeader.stamp = leftImgHeader.stamp = ros::Time::now();

	rightROI.width = rightImgInfo.width;
	leftROI.width = leftImgInfo.width;
	rightROI.height = rightImgInfo.height;
	leftROI.height = leftImgInfo.height;

	if(!zed->grab(sl::zed::SENSING_MODE::RAW, false, false))
	    continue;
     
        //ROS_INFO_STREAM( "Grabbing: " << (ros::Time::now()-acqStart)*1000.0 << " msec");
        //ROS_INFO_STREAM( "FPS: " << zed->getCurrentFPS());

	updateView();

	rightImgInfo.header = rightImgHeader;
	leftImgInfo.header = leftImgHeader;
	rightImgInfo.roi = rightROI;
	leftImgInfo.roi = leftROI;
 
	if(leftImgPub.getNumSubscribers()>0) {
	    leftImgPub.publish( leftImgMsg, leftImgInfo);
	}    
	
	if(rightImgPub.getNumSubscribers()>0) {
	    rightImgPub.publish( rightImgMsg, rightImgInfo);
	}    
	
	ros::spinOnce();
    }

    ROS_INFO_STREAM("Camera stopped...");

    releaseCamera();

    ROS_INFO_STREAM( "Stopping node...");

    ros::shutdown();

    ROS_INFO_STREAM("...done");

    return 0;
}

void App::start(void) {
	
    int result = pthread_create(&rosThread, 0, App::callRunROS, this);
    if(result == 0) pthread_detach(rosThread);

}
