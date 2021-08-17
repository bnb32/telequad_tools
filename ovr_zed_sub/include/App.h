#ifndef _APP_H_
#define _APP_H_

#include <GL/glew.h>
#include "Log.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_mixer.h>
#include "util.h"
#include <OVR.h>
#include <OVR_CAPI.h>
#include <OVR_CAPI_GL.h>

#include <pthread.h>
#include <csignal>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <boost/bind.hpp>

//shader
static const std::string strFragShader = ("uniform sampler2D texImage;\n"
		" void main() {\n"
		" vec4 color = texture2D(texImage, gl_TexCoord[0].st);\n"
		" gl_FragColor = vec4(color.b, color.g, color.r, color.a);\n}");

class App {
	private:
		bool isRunning = true;
		static bool stopping;

		SDL_Window* window = NULL;
		SDL_Renderer* renderer = NULL;
                SDL_GLContext glContext = NULL;

		int windowWidth = 640;
		int windowHeight = 480;

                int zedWidth = 640;
                int zedHeight = 480;

                ovrHmd hmd;
                ovrSizei eyeres[2];
                ovrEyeRenderDesc eye_rdesc[2];
                ovrGLTexture fbOVRglTexure[2];
                union ovrGLConfig glcfg;
                unsigned int flags, displayCapabilities;

	        // shaders

	        GLuint program, shaderF;

                //render target for oculus
                GLuint fbo;
                GLuint fbTexture, fbDepth;
                int fbWidth, fbHeight, fbTextureWidth, fbTextureHeight;

                GLuint rawLeftTexture = 0;
                GLuint rawRightTexture = 0;
                
		float ipd{ OVR_DEFAULT_IPD };
		float eyeHeight{ OVR_DEFAULT_EYE_HEIGHT };

		cudaGraphicsResource* pcuImgResLeft;
		cudaGraphicsResource* pcuImgResRight;
		cudaArray_t ArrImgLeft;
		cudaArray_t ArrImgRight;

		// ros image subscribe
		ros::NodeHandle handle;
		image_transport::ImageTransport rightImgTr;
		image_transport::ImageTransport leftImgTr;
		image_transport::Subscriber rightImgSub;
		image_transport::Subscriber leftImgSub;
		image_transport::CameraPublisher rightImgPub;
		image_transport::CameraPublisher leftImgPub;
		sensor_msgs::Image rightImgMsg;
		sensor_msgs::Image leftImgMsg;
		sensor_msgs::CameraInfo rightImgInfo;
		sensor_msgs::CameraInfo leftImgInfo;
		pthread_t rosThread;	
	
	private:
		void onEvent(SDL_Event* event);
		bool initShader();
		bool initApp();
		bool initTex();
		bool initOVR();
		bool initSDL();
		bool initGL();
		void update();
		void render();
		void cleanup();
		void texQuads();
		void* runROS();
		void texSphere(float xcoord, int divLong, int divLat, 
			       float fovLong, float fovLat, float radius);
		static void sighandler(int signo)
		{
		    App::stopping = (signo == SIGINT);
		}   

	public:
		App();
		int execute(int argc, char* argv[]);
		static void* callRunROS(void *arg) {return ((App*)arg)->runROS(); }
		void zedImgCallback(const sensor_msgs::ImageConstPtr& imgMsg, const std::string& topic);

};

#endif
