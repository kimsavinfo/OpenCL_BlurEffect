//
//  main.cpp
//  OpenCL : kernel and parallel programming
//  Create a Julia fractal then Blur the result
//  Created by Kim SAVAROCHE on 14/06/2015.
//
//  On Xcode :
//  New projet > Other, Empty
//  Add target > OS X console Application
//  Build Phase > Link Binary With Libraries
//                  OpenCL.framework
//
//  In prt, 1rst pixel =
//  ptr[0] : Red
//  ptr[1] : Green
//  ptr[2] : Blue
//  ptr[3] : Alpha

#include <iostream>
#include <stdio.h>
#include <string.h>

#include "../common/cpu_bitmap.h"
#include <OpenCL/cl.h>
#include "ocl_macros.h"

#include "main.h"

void initBitmap(unsigned char *ptr)
{
    for (int tid = 0; tid < GLOBAL_DIM * GLOBAL_DIM; tid++)
    {
        int iPixel = tid * 4;
        int iLine = tid / GLOBAL_DIM;
        int iColumn = tid - iLine * GLOBAL_DIM;
        int offset = iPixel;
        
        bool isJulia = julia(iColumn, iLine) ? 255 : 0;
        ptr[offset] = isJulia ? 10 : 0;
        ptr[offset + 1] = isJulia ? 10 : 0;
        ptr[offset + 2] = isJulia ? 255 : 0;
        ptr[offset + 3] = isJulia ? 10 : 0;
    }
}

int main(int argc, const char * argv[])
{
    // Create a bitmap
    CPUBitmap bitmap(GLOBAL_DIM, GLOBAL_DIM);
    unsigned char *ptr = bitmap.get_ptr();
    initBitmap(ptr);
    
    // Import and launch kernels
    const char *custom_kernel = importKernel();
    launchKernel(custom_kernel, ptr);
    
    // Show the result
    bitmap.display_and_exit();
    
    return 0;
}

