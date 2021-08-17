//==============================================================================
#include "App.h"
#include <GL/glu.h>
#include <string>

bool App::stopping = false;

App::App() 
    : leftImgTr(handle)
    , rightImgTr(handle)
{
    struct sigaction sigAct;
    memset( &sigAct, 0, sizeof(sigAct) );
    sigAct.sa_handler = App::sighandler;
    sigaction(SIGINT, &sigAct, 0);
}

void* App::runROS()
{
    leftImgSub = leftImgTr.subscribe("left/image", 1, boost::bind(&App::zedImgCallback, this, _1, "left"));
    rightImgSub = rightImgTr.subscribe("right/image", 1, boost::bind(&App::zedImgCallback, this, _1, "right"));

    stopping = false;

    while(ros::ok()) {

	if(stopping) break;
	// copy img msgs into local zed mats
	
	ros::spinOnce();
    }
}


int App::execute(int argc, char** argv) {
	if(!initApp()) return 0;
	SDL_Event event;

	int result;
	result = pthread_create(&rosThread, 0, App::callRunROS, this);
	if(result == 0) pthread_detach(rosThread);

	while(isRunning) {
	    if(stopping) break;

	    while(SDL_PollEvent(&event) != 0) { onEvent(&event); }
	    update();
	    render();
	    SDL_Delay(1);
	}
	cleanup();
	return 0;
}

void App::update() {

	//ROS_INFO_STREAM("Updating gpu data");

	cudaGraphicsMapResources(1, &pcuImgResLeft, 0);
        cudaGraphicsSubResourceGetMappedArray(&ArrImgLeft, pcuImgResLeft, 0, 0);
	cudaMemcpy2DToArray(ArrImgLeft, 0, 0, (const void*)&leftImgMsg.data[0], leftImgMsg.step, 
		            leftImgMsg.width * 4, leftImgMsg.height, cudaMemcpyHostToDevice);
	cudaGraphicsUnmapResources(1, &pcuImgResLeft, 0);

	cudaGraphicsMapResources(1, &pcuImgResRight, 0);
        cudaGraphicsSubResourceGetMappedArray(&ArrImgRight, pcuImgResRight, 0, 0);
	cudaMemcpy2DToArray(ArrImgRight, 0, 0, (const void*)&rightImgMsg.data[0], rightImgMsg.step, 
			    rightImgMsg.width * 4, rightImgMsg.height, cudaMemcpyHostToDevice);
	cudaGraphicsUnmapResources(1, &pcuImgResRight, 0); 

}

void App::zedImgCallback(const sensor_msgs::ImageConstPtr& imgMsg, const std::string& topic)
{
    // copy ros msgs to zed arrays
    if( topic == "left" ) {
        //ROS_INFO_STREAM("updating left texture data");
	leftImgMsg = *imgMsg;
    }	
    else if( topic == "right" ) {
        //ROS_INFO_STREAM("updating right texture data");
	rightImgMsg = *imgMsg;
    }
}

void App::render() {
    
    static int frameIndex = 0;
    ++frameIndex;

    ovrMatrix4f rot_mat;

    ovrMatrix4f proj;
    ovrPosef eyeRenderPoses[2];
    ovrTrackingState hmdState;
    ovrVector3f hmdToEyeViewOffset[2] = { eye_rdesc[0].HmdToEyeViewOffset, eye_rdesc[1].HmdToEyeViewOffset };
    ovrHmd_GetEyePoses(hmd, frameIndex, hmdToEyeViewOffset, eyeRenderPoses, &hmdState);

    ovrHmd_BeginFrame(hmd, frameIndex);
    glEnable(GL_DEPTH_TEST);
 
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0,0,0,1);
 
    for(int i=0; i<2; i++) {
 	ovrEyeType eye = hmd->EyeRenderOrder[i];
 	glViewport(eye == ovrEye_Left ? 0 : fbWidth / 2, 0, fbWidth / 2, fbHeight);
 	proj = ovrMatrix4f_Projection(hmd->DefaultEyeFov[eye], 0.5, 500.0, 1);
 	glMatrixMode(GL_PROJECTION);
 	glLoadTransposeMatrixf(proj.M[0]);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//glTranslatef(eye_rdesc[eye].HmdToEyeViewOffset.x,
	// 	     eye_rdesc[eye].HmdToEyeViewOffset.y,
	//	     eye_rdesc[eye].HmdToEyeViewOffset.z);
	rot_mat = OVR::Matrix4f(eyeRenderPoses[eye].Orientation);
	glMultMatrixf((GLfloat*)&rot_mat);
        glTranslatef(-eyeRenderPoses[eye].Position.x, 
		     -eyeRenderPoses[eye].Position.y, 
		     -eyeRenderPoses[eye].Position.z);
        //glTranslatef(0.0, -ovrHmd_GetFloat(hmd, OVR_KEY_EYE_HEIGHT, 1.65), -1.0);
        glTranslatef(0.0, 0.0, -2.0);//-OVR_DEFAULT_IPD : OVR_DEFAULT_IPD , 0.0, -2.0);
	glScalef(1.0, 1.0, 1.0);

        glUseProgram(program);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, eye == ovrEye_Left ? rawLeftTexture : rawRightTexture);

	// draw texture surface
	//texQuads();

	// need to adjust center of sphere for IPD
        texSphere(eye == ovrEye_Left ? OVR_DEFAULT_IPD/2.0 : -OVR_DEFAULT_IPD/2.0, 
		  10, 10, M_PI, M_PI*2.0/3.0, 4.0f);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0); //Done rendering to fbo
    glViewport(0, 0, windowWidth, windowHeight);
    ovrHmd_EndFrame(hmd, eyeRenderPoses, (ovrTexture*)fbOVRglTexure);

    glUseProgram(0);
}

void App::texQuads() {

    glBegin(GL_QUADS); // Draw A Quad for camera image
    glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, 1.0f, 0.0f); // Top Left
    glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, 1.0f, 0.0f); // Top Right
    glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,-1.0f, 0.0f); // Bottom Right
    glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,-1.0f, 0.0f); // Bottom Left
    glEnd(); // Done Drawing The Quad
   

}

void App::texSphere(float xcoord, int divLong, int divLat, float fovLong, float fovLat, float radius) {

    float v0, v1, u;
    int i, j;

    for (i=0; i < divLat; ++i) {
        v0 = ((float) i) / divLat;
        v1 = ((float) i+1) / divLat;

	float lat0 = fovLat * (0.5 - (float) i / divLat);
	float lat1 = fovLat * (0.5 - (float) (i+1) / divLat);

        glBegin(GL_QUAD_STRIP);
	for(j=0; j < divLong+1; ++j) {
            u = ((float) j) / divLong;

	    float lng = fovLong * (-0.5 + (float) j / divLong);

	    glTexCoord2d(u,v0);
	    glVertex3d( radius * cos(lat0) * sin(lng) + xcoord,
	    		radius * sin(lat0),
			-radius * cos(lat0) * cos(lng));

	    glTexCoord2d(u,v1);
	    glVertex3d( radius * cos(lat1) * sin(lng) + xcoord,
	    		radius * sin(lat1),
			-radius * cos(lat1) * cos(lng));
	}
	glEnd();
    }	


}

void App::onEvent(SDL_Event* event) {
    if(event->type == SDL_QUIT) isRunning = false;
    if(event->type == SDL_KEYDOWN){
        ovrHmd_DismissHSWDisplay(hmd);
    }
}

void App::cleanup() {

	if(hmd) {
 		ovrHmd_Destroy(hmd);
 	}
 	ovr_Shutdown();

	
	if(renderer) {
		SDL_DestroyRenderer(renderer);
		renderer = NULL;
	}
	

	if(window) {
		SDL_DestroyWindow(window);
		window = NULL;
	}

    SDL_Quit();
}

bool App::initApp() {

    if(!initSDL()) return false;

    if(!(glContext = SDL_GL_CreateContext(window))) {
        fprintf(stderr, "failed to create OpenGL context\n");
	return false;
    }

    if(!initOVR()) return false;
    
    SDL_SetWindowSize(window, windowWidth, windowHeight);

    glewInit();

    SDL_GL_SetSwapInterval(0);
     
    SDL_GL_MakeCurrent(window, glContext);

    if(!initTex()) return false;
 
    if(!initGL()) return false;

    if(!initShader()) return false;
    
    Log("gl_renderer: %s", glGetString(GL_RENDERER));
    Log("gl_version: %s", glGetString(GL_VERSION));
    Log("glsl_version: %s", glGetString(GL_SHADING_LANGUAGE_VERSION));

    return true;
}

bool App::initGL()
{
    //Oculus OpenGL setup
    ovrHmd_ConfigureTracking(hmd, ovrTrackingCap_Orientation | ovrTrackingCap_MagYawCorrection | ovrTrackingCap_Position, 0);
    eyeres[0] = ovrHmd_GetFovTextureSize(hmd, ovrEye_Left, hmd->DefaultEyeFov[0], 1.0);
    eyeres[1] = ovrHmd_GetFovTextureSize(hmd, ovrEye_Right, hmd->DefaultEyeFov[1], 1.0);
    
    fbWidth = eyeres[0].w + eyeres[1].w;
    fbHeight = eyeres[0].h > eyeres[1].h ? eyeres[0].h : eyeres[1].h;

    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &fbDepth);

    glGenTextures(1, &fbTexture);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    fbTextureWidth = (fbWidth);
    fbTextureHeight = (fbHeight);

    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fbTextureWidth, fbTextureHeight, 0,
            GL_BGRA, GL_UNSIGNED_BYTE, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbTexture, 0);

    /* create and attach the renderbuffer that will serve as our z-buffer */
    glBindRenderbuffer(GL_RENDERBUFFER, fbDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, fbTextureWidth, fbTextureHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, fbDepth);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "incomplete framebuffer!\n");
	return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    printf("created render target: %dx%d (texture size: %dx%d)\n", fbWidth, fbHeight, fbTextureWidth, fbTextureHeight);

 	/* fill in the ovrGLTexture structures that describe our render target texture */
    for(int i=0; i<2; i++) {
    	fbOVRglTexure[i].OGL.Header.API = ovrRenderAPI_OpenGL;
    	fbOVRglTexure[i].OGL.Header.TextureSize.w = fbTextureWidth;
    	fbOVRglTexure[i].OGL.Header.TextureSize.h = fbTextureHeight;
    	/* this next field is the only one that differs between the two eyes */
    	fbOVRglTexure[i].OGL.Header.RenderViewport.Pos.x = i == 0 ? 0 : fbWidth / 2.0;
    	fbOVRglTexure[i].OGL.Header.RenderViewport.Pos.y = fbTextureHeight - fbHeight;
    	fbOVRglTexure[i].OGL.Header.RenderViewport.Size.w = fbWidth / 2.0;
    	fbOVRglTexure[i].OGL.Header.RenderViewport.Size.h = fbHeight;
    	fbOVRglTexure[i].OGL.TexId = fbTexture;	/* both eyes will use the same texture id */
    }

    memset(&glcfg, 0, sizeof glcfg);
    glcfg.OGL.Header.API = ovrRenderAPI_OpenGL;
    glcfg.OGL.Header.BackBufferSize = hmd->Resolution;
    glcfg.OGL.Header.Multisample = 1;

    glcfg.OGL.Disp = glXGetCurrentDisplay();

    ovrHmd_SetEnabledCaps(hmd, ovrHmdCap_LowPersistence | ovrHmdCap_DynamicPrediction);
    displayCapabilities = ovrDistortionCap_TimeWarp | ovrDistortionCap_Overdrive | 
		      ovrDistortionCap_LinuxDevFullscreen;
    if(!ovrHmd_ConfigureRendering(hmd, &glcfg.Config, displayCapabilities, hmd->DefaultEyeFov, eye_rdesc)) {
 	fprintf(stderr, "failed to configure distortion renderer\n");
        return false;
    }

    return true;
}

bool App::initOVR()
{
    //Oculus init
    ovr_Initialize(0);
    
    if(!(hmd = ovrHmd_Create(0))) {
    	fprintf(stderr, "failed to open Oculus HMD, falling back to virtual debug HMD\n");
    	if(!(hmd = ovrHmd_CreateDebug(ovrHmd_DK2))) {
    	    fprintf(stderr, "failed to create virtual debug HMD\n");
    	    return false;
    	}
    }

    printf("initialized HMD: %s - %s\n", hmd->Manufacturer, hmd->ProductName);
    windowWidth = hmd->Resolution.w;
    windowHeight = hmd->Resolution.h;

    if(hmd) {
	ovrHmd_AttachToWindow(hmd, window, NULL, NULL);
    }

    return true;
}

bool App::initTex()
{
    rawLeftTexture = createTextureReference(zedWidth, zedHeight);
    cudaError_t errL = cudaGraphicsGLRegisterImage(&pcuImgResLeft, rawLeftTexture, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);

    rawRightTexture = createTextureReference(zedWidth, zedHeight);
    cudaError_t errR = cudaGraphicsGLRegisterImage(&pcuImgResRight, rawRightTexture, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);

    if(errL != 0 || errR != 0) {
	fprintf(stderr, "Texture initialization failure\n");
	return false;
    }

    return true;

}

bool App::initSDL()
{
    //SDL setup
    if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) {
    	Log("Unable to Init SDL: %s", SDL_GetError());
    	return false;
    }


    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    std::string vererr = SDL_GetError();
    if(!vererr.empty()) Log("Error setting version: %s", (char*)&vererr);

    if(!SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1")) {
        Log("Unable to Init hinting: %s", SDL_GetError());
    }

    if((window = SDL_CreateWindow(
    	"ZED Camera VR",
    	SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
    	1920, 1080, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL)
    ) == NULL) {
    	Log("Unable to create SDL Window: %s", SDL_GetError());
    	return false;
    }


    if((renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED)) == NULL) {
        Log("Unable to create renderer");
        return false;
    }

    return true;
}

bool App::initShader()
{
    //Create the shader program
    
    shaderF = glCreateShader(GL_FRAGMENT_SHADER);
    const char* pszConstString = strFragShader.c_str();
    glShaderSource(shaderF, 1, (const char**) &pszConstString, NULL);
    glCompileShader(shaderF);
    GLint compile_status = GL_FALSE;
    glGetShaderiv(shaderF, GL_COMPILE_STATUS, &compile_status);
    if(compile_status != GL_TRUE) {
        fprintf(stderr, "Error compiling shader");
	return false;
    }	

    program = glCreateProgram();
    glAttachShader(program, shaderF);

    glLinkProgram(program);
    GLint link_status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if(link_status != GL_TRUE) {
        fprintf(stderr, "Error linking shader");
        return false;
    }

    glUniform1i(glGetUniformLocation(program, "texImage"), 0);

    return true;
}
