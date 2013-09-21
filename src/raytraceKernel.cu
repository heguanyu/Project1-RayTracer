// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, float x, float y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
    ray r;

	glm::vec3 A=glm::cross(view,up);
	glm::vec3 B=glm::cross(A,view);
	glm::vec3 M=eye+view;
	glm::vec3 V=B*(glm::length(view)*tan(fov.y)/glm::length(B));
	glm::vec3 H=A*(glm::length(view)*tan(fov.x)/glm::length(A));

	float t1=(x/(resolution.x+0.0f))*2.0f-1.0f;
	float t2=(y/(resolution.y+0.0f))*2.0f-1.0f;
	glm::vec3 P=M-t1*H+t2*V;
	glm::vec3 R=glm::normalize(P-eye);

	r.origin = eye;
	r.direction = R;
	return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

__host__ __device__ float findIntersection(int index, ray r, staticGeom* geoms, int numberOfGeoms ,int& hitidx, glm::vec3& p, glm::vec3& n )
{
		glm::vec3 intersectPoint;
		glm::vec3 normalValue;

		glm::vec3 final_intersectPoint;
		glm::vec3 final_normal;

		float mindist=1000000000;
		glm::vec3 outColor(0,0,0);
		float tempd=0;
		bool isLight=false;

		for(int i=0;i<numberOfGeoms;i++)
		{
			if(geoms[i].type==0)
			{
				tempd=sphereIntersection(geoms[i], r, intersectPoint, normalValue,0.5);
			}
			else if (geoms[i].type==1)
			{
				tempd=boxIntersection(geoms[i], r, intersectPoint, normalValue);
			}
			if(tempd>0 && tempd<mindist)
			{
				mindist=tempd;
				final_intersectPoint=intersectPoint;
				final_normal=normalValue;
				hitidx=i;
			}
		}
		p=final_intersectPoint;
		n=final_normal;
		if(mindist<10000000) return mindist; else return -1;
}
//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
//first step: ray intersect with the scene
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, staticMaterial* materials, staticGeom* lights, int numberOfLights, hitInfo* cudahitinfo, ParameterSet ps ,int offset, float subraycoeff){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  int subray=0;
  if((x<=resolution.x && y<=resolution.y)){
	    int offsetx=offset/2;
		int offsety=offset%2;
		ray r=raycastFromCameraKernel(resolution, time,x+(2*offsetx-1)*0.25f,  y+(2*offsety-1)*0.25f, cam.position, cam.view, cam.up, cam.fov);
		glm::vec3 intersectPoint;
		glm::vec3 normalValue;

		glm::vec3 final_intersectPoint;
		glm::vec3 final_normal;

		float mindist=1000000000;
		glm::vec3 outColor(0,0,0);
		staticMaterial tempMat;
		float tempd=0;
		bool isLight=false;
		staticMaterial targetMat;

		int hitidx,matid;
		tempd=findIntersection(index,r,geoms,numberOfGeoms,hitidx,final_intersectPoint,final_normal);
		cudahitinfo[index].hit=(tempd>0);
		if(tempd<0) return;

		targetMat=materials[geoms[hitidx].materialid];
		outColor=targetMat.color*(ps.ka+targetMat.emittance);
		cudahitinfo[index].hitPoint=final_intersectPoint;
		cudahitinfo[index].normal=final_normal;
		cudahitinfo[index].hitID=hitidx;
		cudahitinfo[index].materialid=geoms[hitidx].materialid;
		cudahitinfo[index].incidentDir=r.direction;

		cudahitinfo[index].firsthitmatid=cudahitinfo[index].materialid;
		cudahitinfo[index].dof=glm::dot(final_intersectPoint-cam.position,cam.view);
		colors[index]+=outColor*subraycoeff;
  }
}


//SECOND step: addRefelctance//!!!!DISCARDED
__global__ void addRefelctance(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, staticMaterial* materials, staticGeom* lights, int numberOfLights, hitInfo* cudahitinfo, ParameterSet ps, float subraycoeff){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  ray r;
  glm::vec3 intersectPoint, normalValue;
  if((x<=resolution.x && y<=resolution.y)){
	if(!cudahitinfo[index].hit)return;

	staticMaterial hitMat=materials[cudahitinfo[index].materialid];
	if(hitMat.hasReflective<1) return;


	ray r;
	r.origin=cudahitinfo[index].hitPoint;
	normalValue=cudahitinfo[index].normal;
	glm::vec3 incidentDir=cudahitinfo[index].incidentDir;
	r.direction=glm::normalize(incidentDir-normalValue*(2*glm::dot(incidentDir,normalValue)));
	


	glm::vec3 final_intersectPoint;
	glm::vec3 final_normal;
	glm::vec3 outColor(0,0,0);
	float tempd=0;
	
	staticMaterial targetMat;
	int hitidx;

	tempd=findIntersection(index,r,geoms,numberOfGeoms,hitidx,final_intersectPoint,final_normal);
	if(tempd<0) return;

	targetMat=materials[geoms[hitidx].materialid];
	outColor=targetMat.color*(targetMat.emittance+ps.ka*0.3f);
	colors[index]+=outColor*subraycoeff;
//	if(hitMat.specularExponent<EPSILON) colors[index]+=outColor;
//	else colors[index]+=outColor*pow(abs(glm::dot(incidentDir,cudahitinfo[index].normal)),hitMat.specularExponent);
  }
}

__global__ void dofBlur(glm::vec2 resolution, float time, glm::vec3* toColors, glm::vec3* colors, hitInfo* cudahitinfo)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if((x<=resolution.x && y<=resolution.y)){
	  toColors[index]=colors[index];
	  if(!cudahitinfo[index].hit) return;
	  
	  float standardDOF=cudahitinfo[(int)(resolution.x*resolution.y/2)].dof;
	  float myDOF=cudahitinfo[index].dof;
	  if(abs(myDOF/standardDOF-1.0f)<0.1f) return;
	  int blurradius=(int)(abs(myDOF/standardDOF-1.0f)/0.1f);
	  if(blurradius>6) blurradius=6;
	  int blurnum=0;
	  toColors[index]=glm::vec3(0,0,0);
	  for(int i=-blurradius;i<=blurradius;i++) for(int j=-blurradius;j<=blurradius;j++)
	  {
		  int xx=x+i, yy=y+j;
		  if(xx<=0 || xx>resolution.x || yy<=0 || yy>resolution.y) continue;
		  int newidx=xx+yy*resolution.x;
		  if(abs(cudahitinfo[newidx].dof/standardDOF-1.0f)<0.1f) continue;
		  blurnum++;
		  toColors[index]+=colors[newidx];
	  }
	  toColors[index]*=(1.0f/(float)blurnum);
  }
}
//3RD STEP: add Shadow and specular, or else to say, trace the lights
__global__ void addShadow(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, staticMaterial* materials, staticGeom* lights, int numberOfLights, hitInfo* cudahitinfo, ParameterSet ps , glm::vec3* shadow, float subraycoeff){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  ray r;
  glm::vec3 intersectPoint, normalValue;
  if((x<=resolution.x && y<=resolution.y)){
	if(!cudahitinfo[index].hit) return;
	staticMaterial hitMat=materials[cudahitinfo[index].materialid];
	if(hitMat.emittance>0.01f) return;
	glm::vec3 final_intersectPoint=cudahitinfo[index].hitPoint;
	glm::vec3 final_normal=cudahitinfo[index].normal;
	
	float tempd;
	glm::vec3 outColor=colors[index];
	glm::vec3 accumulateInLight(0,0,0);
	glm::vec3 tempLight;
	glm::vec3 lightRaysSum(0,0,0);
	bool flag=false;

	float coeff=0;
	r.origin=final_intersectPoint+0.001f*final_normal;
	for(int i=0;i<numberOfLights;i++)
	{
		

			tempLight=getRandomPoint(lights[i],time*index);
		r.direction=glm::normalize(tempLight-r.origin);
		if(glm::dot(final_normal,r.direction)<0) continue;
		float mind=0;
		if(lights[i].type==0) mind=sphereIntersection(lights[i],r,intersectPoint, normalValue,0.5);
		else if (lights[i].type==1) mind=boxIntersection(lights[i],r,intersectPoint,normalValue);
		if(mind<0)continue;
		mind-=0.0001f;
		flag=false;
		float cloest=mind;
		bool isTransmittance=false;
		float ior;
		int theID;
		glm::vec3 theHitpoint, thenormal;
		glm::vec3 transmittancecolor(1,1,1);
		for(int k=0;k<numberOfGeoms;k++)
		{
			if(materials[geoms[k].materialid].emittance>0.01f) continue;
			//if(materials[geoms[k].materialid].hasRefractive>0.5f) continue;
			if(geoms[k].type==0) tempd=sphereIntersection(geoms[k],r,intersectPoint,normalValue,0.5);
			else if(geoms[k].type==1) tempd=boxIntersection(geoms[k],r,intersectPoint,normalValue);
			if(tempd<cloest && tempd>0){
				cloest=tempd; 
				isTransmittance=(materials[geoms[k].materialid].hasRefractive>0.5f);
				flag=!isTransmittance;
				ior=materials[geoms[k].materialid].indexOfRefraction;
				theID=k;
				theHitpoint=intersectPoint;thenormal=normalValue;
				transmittancecolor=materials[geoms[k].materialid].color;
			}
		}
		ray rr;
		if(isTransmittance)
		{

			glm::vec3 dir1=calculateTransmissionDirection(thenormal,glm::normalize(lights[i].translation-r.origin),1,ior);
			if(glm::length(dir1)<0.5f)return;
			glm::vec3 hp1=theHitpoint;
			glm::vec3 hp2,normal2;
			rr.origin=hp1+dir1*0.001f;
			rr.direction=dir1;
			if(geoms[theID].type==0) tempd=sphereIntersection(geoms[theID],rr,hp2,normal2,0.5f);
			else if (geoms[theID].type==1) tempd=boxIntersection(geoms[theID],rr,hp2,normal2);
			normal2=-normal2;
			glm::vec3 dir2=calculateTransmissionDirection(normal2, dir1,ior,1);
			rr.origin=hp2+dir2*0.01f;
			rr.direction=dir2;

			if(lights[i].type==0) mind=sphereIntersection(lights[i],rr,intersectPoint, normalValue,0.5);
			else if (lights[i].type==1) mind=boxIntersection(lights[i],rr,intersectPoint,normalValue);
			if(mind<0) flag=true;
		//	if(!flag) {colors[index]=glm::vec3(0,0,0);return;}
		}
		else
		{
			transmittancecolor=glm::vec3(1,1,1);
		}
		if(!flag)
		{
			coeff=0;
			Fresnel f=calculateFresnel(final_normal,r.direction*-1.0f,1,hitMat.indexOfRefraction,hitMat.specularExponent);
			glm::vec3  R=r.direction*(-1.0f);
			R=calculateReflectionDirection(final_normal,R);
			coeff=glm::dot(R,-cudahitinfo[index].incidentDir);
			if(coeff<0) coeff=0;
			else
				coeff=glm::pow(coeff,hitMat.specularExponent)*hitMat.hasReflective*ps.ks;	///specular light intensity
			
			coeff+=glm::dot(final_normal,r.direction)*ps.kd;//*f.reflectionCoefficient;    ///Diffuse light intensity
			accumulateInLight+=colorMultiply(materials[lights[i].materialid].color*coeff,transmittancecolor);
		}	

		accumulateInLight*=materials[lights[i].materialid].emittance;
		lightRaysSum+=accumulateInLight;
		accumulateInLight=glm::vec3(0,0,0);
	}
	
	glm::vec3 matcolor=materials[cudahitinfo[index].firsthitmatid].color;
	outColor=colorMultiply(matcolor,lightRaysSum);//.x,matcolor.y*lightRaysSum.y,matcolor.z*lightRaysSum.z);
	shadow[index]+=outColor*subraycoeff;
	colors[index]+=shadow[index]/(time+1)*subraycoeff;
	
   }
}
//3RD STEP: add Shadow and specular, or else to say, trace the lights

__global__ void refractionCorrection(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, staticMaterial* materials, staticGeom* lights, int numberOfLights, hitInfo* cudahitinfo, ParameterSet ps,float subraycoeff ){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  ray r;
  glm::vec3 intersectPoint, normalValue;

  if((x<=resolution.x && y<=resolution.y)){
	if(!cudahitinfo[index].hit) return;
	staticMaterial hitMat=materials[cudahitinfo[index].materialid];
	
	if(hitMat.emittance>0.01f) return;
	if(hitMat.hasRefractive<0.5f)return;
	glm::vec3 final_intersectPoint=cudahitinfo[index].hitPoint;
	glm::vec3 final_normal=cudahitinfo[index].normal;
	
	float tempd;
	glm::vec3 outColor(0,0,0);
	glm::vec3 dir1=calculateTransmissionDirection(final_normal,cudahitinfo[index].incidentDir,1,hitMat.indexOfRefraction);
	if(glm::length(dir1)<0.5f)
	{
		r.origin=cudahitinfo[index].hitPoint+final_normal*0.001f;
		r.direction=calculateReflectionDirection(final_normal,cudahitinfo[index].incidentDir);
	}
	else
	{
		glm::vec3 hp1=final_intersectPoint+dir1*0.001f;
		glm::vec3 hp2,normal2;
		r.origin=hp1;
		r.direction=dir1;
		if(geoms[cudahitinfo[index].hitID].type==0) tempd=sphereIntersection(geoms[cudahitinfo[index].hitID],r,hp2,normal2,0.5f);
		else if (geoms[cudahitinfo[index].hitID].type==1) tempd=boxIntersection(geoms[cudahitinfo[index].hitID],r,hp2,normal2);
		normal2=-normal2;
		glm::vec3 dir2=calculateTransmissionDirection(normal2, dir1,hitMat.indexOfRefraction,1);
		r.origin=hp2+dir2*0.01f;
		r.direction=dir2;
		r.origin-=normal2*0.001f;
	}
	staticMaterial targetMat;
	int hitidx,matid;
	tempd=findIntersection(index,r,geoms,numberOfGeoms,hitidx,final_intersectPoint,final_normal);
	cudahitinfo[index].hit=(tempd>0);
	if(tempd<0) return;
	targetMat=materials[geoms[hitidx].materialid];
	outColor=targetMat.color*(ps.ka+targetMat.emittance);
	cudahitinfo[index].hitPoint=final_intersectPoint;
	cudahitinfo[index].normal=final_normal;
	cudahitinfo[index].hitID=hitidx;
	cudahitinfo[index].materialid=geoms[hitidx].materialid;
	cudahitinfo[index].incidentDir=r.direction;

	colors[index]+=outColor*subraycoeff;
   }
}
__global__ void smoothImage(glm::vec2 resolution, glm::vec3* src,glm::vec3* targ){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  int idxtar=(x/2)+(y*resolution.x/4);
  targ[idxtar]+=src[index]*0.25f;
}

parallelRay* raypool;
glm::vec3* cudashadow;

__global__ void initializeRayPool(glm::vec2 resolution, float time, cameraData cam, parallelRay* raypool){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if((x>resolution.x || y>resolution.y)) return;
  ray r=raycastFromCameraKernel(resolution, time,x,  y, cam.position, cam.view, cam.up, cam.fov);
  parallelRay pr;
  pr.direction=r.direction;
  pr.index=index;
  pr.iters=0;
  pr.origin=r.origin;
  pr.coeff=1.0f;
  raypool[index]=pr;
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos,camera* renderCam, ParameterSet* pSet, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = pSet->shadowRays;	//don't care about this var name. it is tilesize from input file
  int numberOfLights=0;

  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
  cam.ambient=renderCam->ambient;

  if(iterations<1.5f)
  {
	  cudaMalloc((void**)&cudashadow, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	  cudaMemcpy( cudashadow, renderCam->shadowVal, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	  cudaMalloc((void**)&raypool, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(parallelRay));
	  initializeRayPool<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,(float)iterations,cam,raypool);
  }
  
  hitInfo* cudahitinfo= NULL;
  cudaMalloc((void**)&cudahitinfo,(int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(hitInfo));


  hitInfo* hitInfoNullList = new hitInfo[(int)renderCam->resolution.x*(int)renderCam->resolution.y];
  for(int i=0;i<(int)renderCam->resolution.x*(int)renderCam->resolution.y;i++)
  {
	
	  hitInfoNullList[i].hitPoint=glm::vec3(0,0,0);
	  hitInfoNullList[i].normal=glm::vec3(0,0,0);
	  hitInfoNullList[i].hit=false;
	  hitInfoNullList[i].hitID=0;
	  hitInfoNullList[i].materialid=0;
	  hitInfoNullList[i].incidentDir=glm::vec3(0,0,0);
	  hitInfoNullList[i].firsthitmatid=0;
  }
  cudaMemcpy( cudahitinfo,hitInfoNullList,(int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(hitInfo), cudaMemcpyHostToDevice);

  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
	if(materials[newStaticGeom.materialid].emittance>0.01f) numberOfLights++;
    geomList[i] = newStaticGeom;
  }
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  staticGeom* lightList = new staticGeom[numberOfLights];
  int now=0;
  for(int i=0;i<numberOfGeoms;i++)
  {
	if(materials[geomList[i].materialid].emittance<=0.01f) continue;
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    lightList[now] = newStaticGeom;
	now++;
  }

  staticGeom* cudalights=NULL;
  cudaMalloc((void**)&cudalights, numberOfLights*sizeof(staticGeom));
  cudaMemcpy( cudalights, lightList, numberOfLights*sizeof(staticGeom), cudaMemcpyHostToDevice);


  staticMaterial* matList = new staticMaterial[numberOfMaterials];
  for(int i=0; i<numberOfMaterials; i++){
    staticMaterial newStaticMat;

    newStaticMat.color = materials[i].color;
    newStaticMat.specularExponent = materials[i].specularExponent;
    newStaticMat.specularColor = materials[i].specularColor;
    newStaticMat.hasReflective = materials[i].hasReflective;
    newStaticMat.hasRefractive = materials[i].hasRefractive;
    newStaticMat.indexOfRefraction = materials[i].indexOfRefraction;
	newStaticMat.hasScatter = materials[i].hasScatter;
	newStaticMat.absorptionCoefficient = materials[i].absorptionCoefficient;
	newStaticMat.reducedScatterCoefficient = materials[i].reducedScatterCoefficient;
	newStaticMat.emittance = materials[i].emittance;
	
    matList[i] = newStaticMat;
  }

  staticMaterial* cudamats = NULL;
  cudaMalloc((void**)&cudamats, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamats, matList, numberOfMaterials*sizeof(staticMaterial), cudaMemcpyHostToDevice);

  //package camera


  ParameterSet ps;
  ps.ka=pSet->ka;
  ps.kd=pSet->kd;
  ps.ks=pSet->ks;
  ps.shadowRays=pSet->shadowRays;
  ps.hasSubray=pSet->hasSubray;


  
  //kernel launches
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage);
  int subrays=pSet->hasSubray;
  float subraycoeff=1.0f/(float)subrays;
  for(int i=0;i<subrays;i++)
  {
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms , cudamats, cudalights, numberOfLights, cudahitinfo ,ps,i,subraycoeff);
  addRefelctance<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms , cudamats, cudalights, numberOfLights, cudahitinfo,ps,subraycoeff);
  refractionCorrection<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms , cudamats, cudalights, numberOfLights, cudahitinfo,ps,subraycoeff);
  addShadow<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms , cudamats, cudalights, numberOfLights, cudahitinfo,ps, cudashadow,subraycoeff);
  }
  

//  cudaMemcpy(renderCam->shadowVal, cudashadow,(int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  glm::vec3* blurredimage = NULL;
  cudaMalloc((void**)&blurredimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  dofBlur<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,(float)iterations,blurredimage, cudaimage, cudahitinfo);
  cudaMemcpy( cudaimage, blurredimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  
  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamats );
  cudaFree( cudalights);
  cudaFree( cudahitinfo);
  cudaFree( blurredimage);
//  cudaFree( cudashadow);

  delete geomList;
  delete matList;
  delete lightList;
  delete hitInfoNullList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
