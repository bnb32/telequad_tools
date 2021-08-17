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

#include <zed/Camera.hpp>
#include <thread>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

//shader
static const std::string strFragShader = ("uniform sampler2D texImage;\n"
		" void main() {\n"
		" vec4 color = texture2D(texImage, gl_TexCoord[0].st);\n"
		" gl_FragColor = vec4(color.b, color.g, color.r, color.a);\n}");

class App {
	private:
		static App instance;

		bool isRunning = true;

		SDL_Window* window = NULL;
		SDL_Renderer* renderer = NULL;
                SDL_GLContext glContext = NULL;

		int windowWidth = 640;
		int windowHeight = 480;

                sl::zed::Camera* zed;
                int zedWidth = 0;
                int zedHeight = 0;

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
		int zedRes;

                GLuint rawLeftTexture = 0;
                GLuint rawRightTexture = 0;
                
		float ipd{ OVR_DEFAULT_IPD };
		float eyeHeight{ OVR_DEFAULT_EYE_HEIGHT };

		cudaGraphicsResource* pcuImgResLeft;
		cudaGraphicsResource* pcuImgResRight;
		cudaArray_t ArrImgLeft;
		cudaArray_t ArrImgRight;
		sl::zed::Mat gpuImgLeft;
		sl::zed::Mat gpuImgRight;



	private:
		App();
		void onEvent(SDL_Event* event);
		int shader();
		bool init();
		void update();
		void render();
		void cleanup();
                void working_render();
		void zedInfo();
		void texQuads();
		void texSphere(int divLong, int divLat, float fovLong, float fovLat, float radius);

	public:
		int execute(int argc, char* argv[]);

	public:
		static App* getInstance();

};

#endif
