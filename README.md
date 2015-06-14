# OpenCL_BlurEffect

## Initialize a bitmap with Julia fractal in cpp.

* julia_manager.cpp : julia's / maths library

![ScreenShot](/imgs/julia_init.PNG)


##Then with OpenCL, create a kernel and then blur the image.

* kernel_manager.cpp : how to import and launch the kernel
* custom_kernel.cl : what do we want to do in the kernel

![ScreenShot](/imgs/julia_blur.PNG)

### memento.cl
It will illustrate the following scheme from the file "pdf/intro_to_opencl.pdf"
![ScreenShot](/imgs/memento.PNG)
This pdf is not from me obviously, I recommend it.

### On Xcode :
New project > Other, Empty
Add target > OS X console Application
Build Phase > Link Binary With Libraries > OpenCL.framework