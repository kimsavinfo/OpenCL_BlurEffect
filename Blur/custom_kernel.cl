#define GLOBAL_DIM 800
#define WORKGROUP_DIM 10

__kernel
void custom_kernel(  __global unsigned char *ptr )
{
    // All items have to wait for the cache to be initialize
    __local int cache[4];
    cache[0] = 0; // Red axis
    cache[1] = 0; // Green axis
    cache[2] = 0; // Blue axis
    cache[3] = 0; // Alpha axis
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Get the essential/global information
    // We calculate them to be more agile and avoid settings too many "#define"
    int nbItemsPerWorkGroup = get_local_size(0);
    int iWorkGroup = get_group_id(0);
    int iLocalId = get_local_id(0);
    int nbWorkGroupsPerLine = GLOBAL_DIM / WORKGROUP_DIM;
    
    // Cacuate the line
    int iLineWG = iWorkGroup / nbWorkGroupsPerLine;
    int iLineLocale = iLocalId / WORKGROUP_DIM;
    int iLineGlobal = iLineWG * (GLOBAL_DIM * WORKGROUP_DIM) + iLineLocale * GLOBAL_DIM;
    
    // Calculate the column
    int iColumnWG = iWorkGroup - iLineWG * nbWorkGroupsPerLine;
    int iColumnLocale = iLocalId - iLineLocale * WORKGROUP_DIM;
    int iColumnGlobal = iColumnWG * WORKGROUP_DIM + iColumnLocale;
    
    // Calculate the pixel offset and Julia Set
    int offset = (iLineGlobal + iColumnGlobal) * 4;
    
    // We have to wait for all items to get the pixel information
    cache[0] += ptr[offset];
    cache[1] += ptr[offset + 1];
    cache[2] += ptr[offset + 2];
    cache[3] += ptr[offset + 3];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Then we can assign the axis' average for each pixel
    ptr[offset]     = cache[0] / nbItemsPerWorkGroup;
    ptr[offset + 1] = cache[1] / nbItemsPerWorkGroup;
    ptr[offset + 2] = cache[2] / nbItemsPerWorkGroup;
    ptr[offset + 3] = cache[3] / nbItemsPerWorkGroup;
}