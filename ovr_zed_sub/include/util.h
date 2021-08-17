
#ifndef _UTIL_H_
#define _UTIL_H_

#include <stdio.h>
#include <GL/glew.h>
#include <GL/glx.h>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <SDL2/SDL.h>


GLuint createProgram(std::string vertex, std::string fragment);
GLuint createTextureReference(int w, int h);
void printProgramInfoLog(GLuint obj);
void printShaderInfoLog(GLuint obj);
std::string getFileContents(std::string filename);
GLfloat* createPerspectiveMatrix(float fov, float aspect, float near, float far);
void setPerspectiveFrustrum(GLdouble fovY, GLdouble aspect, GLdouble near, GLdouble far);

#endif 
