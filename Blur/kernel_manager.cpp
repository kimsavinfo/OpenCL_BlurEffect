const char* importKernel()
{
    FILE *fp;
    char *source_str;
    size_t source_size;
    const char *kernel_path = KERNEL_PATH;
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
    return source_str;
}

void launchKernel(const char *custom_kernel, unsigned char *ptr)
{
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
    cl_mem ptr_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(unsigned char) * GLOBAL_DIM * GLOBAL_DIM * 4, NULL, &clStatus);
    
    // Copy the Buffer for ptr to the device. We do a blocking write to the device buffer.
    clStatus = clEnqueueWriteBuffer(command_queue, ptr_clmem, CL_TRUE, 0, sizeof(unsigned char) * GLOBAL_DIM * GLOBAL_DIM * 4, ptr, 0, NULL, NULL);
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
    
    // Execute the OpenCL kernel on the list
    // The number of work groups will be automatically calculated = global_size / local_size
    size_t global_size = GLOBAL_DIM * GLOBAL_DIM;
    size_t local_size = WORKGROUP_DIM * WORKGROUP_DIM; // the number of work-items in 1 work-group
    cl_event saxpy_event;
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &saxpy_event);
    LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed...");
    
    clStatus = clEnqueueReadBuffer(command_queue, ptr_clmem, CL_TRUE, 0,
                                   sizeof(unsigned char) * GLOBAL_DIM * GLOBAL_DIM * 4, ptr, 1, &saxpy_event, NULL);
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
}