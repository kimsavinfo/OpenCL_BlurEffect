//
//  main.cpp
//  Blur
//
//  Created by Kim SAVAROCHE on 14/06/2015.
//
//

#include <iostream>
#include <stdio.h>
#include <string.h>

#include "../common/cpu_bitmap.h"
#include <OpenCL/cl.h>
#include "ocl_macros.h"

// Common defines
#define MAX_SOURCE_SIZE (0x100000)
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU
#define DIM 800 // longueur et largeur du canevas = DIM * DIM pixels
#define WORKGROUP_DIM 10 // nombre d'items dans un workgroup = NB_WORK_ITEM_LINE * NB_WORK_ITEM_LINE


/* ================== JULIA : DEBUT ================== */
typedef struct cuComplex {
    float r;
    float i;
    
} cuComplex;

cuComplex createComplex(float p_r, float p_i)
{
    cuComplex complex;
    complex.r = p_r;
    complex.i = p_i;
    return complex;
}

cuComplex multiply(cuComplex p_complex1, cuComplex p_complex2)
{
    return  createComplex(
                          p_complex1.r * p_complex2.r - p_complex1.i * p_complex2.i,
                          p_complex1.i * p_complex2.r + p_complex1.r * p_complex2.i
                          );
    
}

cuComplex add(cuComplex p_complex1, cuComplex p_complex2)
{
    return createComplex(
                         p_complex1.r + p_complex2.r,
                         p_complex1.i + p_complex2.i
                         );
    
}

float magnitude2(cuComplex z)
{
    return z.r * z.r + z.i * z.i;
}

int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);
    
    cuComplex c = createComplex(-0.8, 0.156);
    cuComplex a = createComplex(jx, jy);
    
    int i = 0;
    for (i = 0; i<200; i++) {
        a = add(multiply(a, a), c);
        if (magnitude2(a) > 1000)
        return 0;
    }
    
    return 1;
}

/* ================== JULIA : FIN ================== */

// Init bitmap en blanc
void set_julia_bitmap(unsigned char *ptr)
{
    int nbWorkGroup = (DIM * DIM) / (WORKGROUP_DIM * WORKGROUP_DIM);
    int nbWorkGroupParLigne = DIM / WORKGROUP_DIM;
    int nbItemsParWorkGroup = WORKGROUP_DIM * WORKGROUP_DIM;
    
    for (int iWorkGroup = 0; iWorkGroup < nbWorkGroup - 1; iWorkGroup++)
    {
        for (int iLocalId = 0; iLocalId < nbItemsParWorkGroup; iLocalId++)
        {
            int iLigneWG = iWorkGroup / nbWorkGroupParLigne;
            int iColonneWG = iWorkGroup - iLigneWG * nbWorkGroupParLigne;
            
            int iLigneLocale = iLocalId / WORKGROUP_DIM;
            int iColonneLocale = iLocalId - iLigneLocale * WORKGROUP_DIM;
            
            int ligne = iLigneWG * (DIM * WORKGROUP_DIM) + iLigneLocale * DIM;
            int colonne = iColonneWG * WORKGROUP_DIM + iColonneLocale;
            
            int iPtr = ligne + colonne;
            int offset = iPtr * 4;
            
            int juliaLine = iPtr / DIM;
            int juliaColumn = iPtr - juliaLine * DIM;
            ptr[offset] = julia(juliaColumn, juliaLine) ? 255 : 0;
            ptr[offset + 1] = 0;
            ptr[offset + 2] = 0;
            ptr[offset + 3] = 0;
        }
    }
}


int main(int argc, const char * argv[])
{
    
    // ====================== Init ======================
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *ptr = bitmap.get_ptr();
    set_julia_bitmap(ptr);
    
    
    // ====================== Importer le kernel ======================
    FILE *fp;
    char *source_str;
    size_t source_size;
    const char *kernel_path = "/Users/kimsavinfo/Dropbox/epsi/I4/cpp/Ex_Workgroup/Ex_Workgroup/custom_kernel.cl";
    fp = fopen(kernel_path, "r");
    if(!fp)
    {
        perror(kernel_path);
        fprintf(stderr, "Erreur import kernel");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    const char *custom_kernel = source_str;
    
    // ====================== Appeler le kernel ======================
    cl_int clStatus; //Keeps track of the error values returned.
    
    // Get platform and device information
    cl_platform_id * platforms = NULL;
    
    // Set up the Platform. Take a look at the MACROs used in this file.
    // These are defined in common/ocl_macros.h
    OCL_CREATE_PLATFORMS(platforms);
    
    // Get the devices list and choose the type of device you want to run on
    cl_device_id *device_list = NULL;
    OCL_CREATE_DEVICE(platforms[0], DEVICE_TYPE, device_list);
    
    // Create OpenCL context for devices in device_list
    cl_context context;
    
    // An OpenCL context can be associated to multiple devices, either CPU or GPU
    // based on the value of DEVICE_TYPE defined above.
    context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateContext Failed...");
    
    // Create a command queue for the first device in device_list
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateCommandQueue Failed...");
    
    // Create memory buffers on the device for each vector
    cl_mem ptr_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(unsigned char) * DIM * DIM * 4, NULL, &clStatus);
    
    // Copy the Buffer A and B to the device. We do a blocking write to the device buffer.
    clStatus = clEnqueueWriteBuffer(command_queue, ptr_clmem, CL_TRUE, 0, sizeof(unsigned char) * DIM * DIM * 4, ptr, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed...");
    
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&custom_kernel, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateProgramWithSource Failed...");
    
    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    if (clStatus != CL_SUCCESS)
    LOG_OCL_COMPILER_ERROR(program, device_list[0]);
    
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "custom_kernel", &clStatus);
    
    // Set the arguments of the kernel. Take a look at the kernel definition in sum_event
    // variable. First parameter is a constant and the other three are buffers.
    clStatus |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&ptr_clmem);
    LOG_OCL_ERROR(clStatus, "clSetKernelArg Failed...");
    
    // ******************** Execute the OpenCL kernel on the list
    // Set number of work-items in 1 work-group
    // Calcul du nombre de work group à créer : global_size / local_size
    size_t global_size = DIM * DIM;
    size_t local_size = WORKGROUP_DIM * WORKGROUP_DIM;
    cl_event saxpy_event;
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &saxpy_event);
    LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed...");
    
    clStatus = clEnqueueReadBuffer(command_queue, ptr_clmem, CL_TRUE, 0,
                                   sizeof(unsigned char) * DIM * DIM * 4, ptr, 1, &saxpy_event, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueReadBuffer Failed...");
    
    // Clean up and wait for all the comands to complete.
    clStatus = clFinish(command_queue);
    
    // Finally release all OpenCL objects and release the host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(ptr_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    
    free(platforms);
    free(device_list);
    // ====================== Afficher l'image ======================
    
    bitmap.display_and_exit();
    
    return 0;
}
