// Common defines
#define MAX_SOURCE_SIZE (0x100000)
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU

// We'll use he following square dimension :
#define GLOBAL_DIM 800 // How many pixels are there on one image's side ?
// Total pixels = GLOBAL_DIM * GLOBAL_DIM
#define WORKGROUP_DIM 10 // How many pixels are there on one workgroup's side ?
// Total items in one workgroup = NB_WORK_ITEM_LINE * NB_WORK_ITEM_LINE

#define KERNEL_PATH "/Users/kimsavinfo/Dropbox/epsi/I4/cpp/Ex_Workgroup/Ex_Workgroup/custom_kernel.cl"
#include "kernel_manager.cpp"
#include "julia_manager.cpp"