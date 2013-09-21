// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef CUDASTRUCTS_H
#define CUDASTRUCTS_H

#include "glm/glm.hpp"
#include "cudaMat4.h"
#include <cuda_runtime.h>
#include <string>

enum GEOMTYPE{ SPHERE, CUBE, MESH };
struct ParameterSet
{
	float ks,kd,ka;
	int shadowRays, hasSubray;
};
struct ray {
	glm::vec3 origin;
	glm::vec3 direction;
};
struct parallelRay
{
	glm::vec3 origin;
	glm::vec3 direction;
	int index;
	int iters;
	float coeff;
};
struct vec6
{
	glm::vec3 point;
	glm::vec3 normal;
};
struct geom {
	enum GEOMTYPE type;
	int materialid;
	int frames;
	glm::vec3* translations;
	glm::vec3* rotations;
	glm::vec3* scales;
	cudaMat4* transforms;
	cudaMat4* inverseTransforms;
};

struct staticGeom {
	enum GEOMTYPE type;
	int materialid;
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	cudaMat4 transform;
	cudaMat4 inverseTransform;
};

struct cameraData {
	glm::vec2 resolution;
	glm::vec3 position;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec2 fov;
	float ambient;
};

struct camera {
	glm::vec2 resolution;
	glm::vec3* positions;
	glm::vec3* views;
	glm::vec3* ups;
	int frames;
	glm::vec2 fov;
	unsigned int iterations;
	glm::vec3* image;
	glm::vec3* shadowVal;
	ray* rayList;
	std::string imageName;
	float ambient;
};

struct material{
	glm::vec3 color;
	float specularExponent;
	glm::vec3 specularColor;
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float hasScatter;
	glm::vec3 absorptionCoefficient;
	float reducedScatterCoefficient;
	float emittance;
};

struct staticMaterial{
	glm::vec3 color;
	float specularExponent;
	glm::vec3 specularColor;		//secondary hit specular info
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float hasScatter;
	glm::vec3 absorptionCoefficient;
	float reducedScatterCoefficient;
	float emittance;
};

struct hitInfo
{
	bool hit;
	glm::vec3 hitPoint;
	glm::vec3 normal;
	glm::vec3 incidentDir;
	int hitID;
	int materialid;	
	int firsthitmatid;
	
};

#endif //CUDASTRUCTS_H
