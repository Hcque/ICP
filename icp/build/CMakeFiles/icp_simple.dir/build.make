# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/haochen/Desktop/ICP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/haochen/Desktop/ICP/build

# Include any dependencies generated for this target.
include CMakeFiles/icp_simple.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/icp_simple.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/icp_simple.dir/flags.make

CMakeFiles/icp_simple.dir/src/icp_simple.cpp.o: CMakeFiles/icp_simple.dir/flags.make
CMakeFiles/icp_simple.dir/src/icp_simple.cpp.o: ../src/icp_simple.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/haochen/Desktop/ICP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/icp_simple.dir/src/icp_simple.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/icp_simple.dir/src/icp_simple.cpp.o -c /home/haochen/Desktop/ICP/src/icp_simple.cpp

CMakeFiles/icp_simple.dir/src/icp_simple.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/icp_simple.dir/src/icp_simple.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/haochen/Desktop/ICP/src/icp_simple.cpp > CMakeFiles/icp_simple.dir/src/icp_simple.cpp.i

CMakeFiles/icp_simple.dir/src/icp_simple.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/icp_simple.dir/src/icp_simple.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/haochen/Desktop/ICP/src/icp_simple.cpp -o CMakeFiles/icp_simple.dir/src/icp_simple.cpp.s

# Object files for target icp_simple
icp_simple_OBJECTS = \
"CMakeFiles/icp_simple.dir/src/icp_simple.cpp.o"

# External object files for target icp_simple
icp_simple_EXTERNAL_OBJECTS =

icp_simple: CMakeFiles/icp_simple.dir/src/icp_simple.cpp.o
icp_simple: CMakeFiles/icp_simple.dir/build.make
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_people.so
icp_simple: /usr/lib/x86_64-linux-gnu/libboost_system.so
icp_simple: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
icp_simple: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
icp_simple: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
icp_simple: /usr/lib/x86_64-linux-gnu/libboost_regex.so
icp_simple: /usr/lib/x86_64-linux-gnu/libqhull.so
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libfreetype.so
icp_simple: /usr/lib/x86_64-linux-gnu/libz.so
icp_simple: /usr/lib/x86_64-linux-gnu/libjpeg.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpng.so
icp_simple: /usr/lib/x86_64-linux-gnu/libtiff.so
icp_simple: /usr/lib/x86_64-linux-gnu/libexpat.so
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_features.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_search.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_io.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
icp_simple: /usr/lib/x86_64-linux-gnu/libpcl_common.so
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libfreetype.so
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
icp_simple: /usr/lib/x86_64-linux-gnu/libz.so
icp_simple: /usr/lib/x86_64-linux-gnu/libGLEW.so
icp_simple: /usr/lib/x86_64-linux-gnu/libSM.so
icp_simple: /usr/lib/x86_64-linux-gnu/libICE.so
icp_simple: /usr/lib/x86_64-linux-gnu/libX11.so
icp_simple: /usr/lib/x86_64-linux-gnu/libXext.so
icp_simple: /usr/lib/x86_64-linux-gnu/libXt.so
icp_simple: CMakeFiles/icp_simple.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/haochen/Desktop/ICP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable icp_simple"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/icp_simple.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/icp_simple.dir/build: icp_simple

.PHONY : CMakeFiles/icp_simple.dir/build

CMakeFiles/icp_simple.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/icp_simple.dir/cmake_clean.cmake
.PHONY : CMakeFiles/icp_simple.dir/clean

CMakeFiles/icp_simple.dir/depend:
	cd /home/haochen/Desktop/ICP/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/haochen/Desktop/ICP /home/haochen/Desktop/ICP /home/haochen/Desktop/ICP/build /home/haochen/Desktop/ICP/build /home/haochen/Desktop/ICP/build/CMakeFiles/icp_simple.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/icp_simple.dir/depend

