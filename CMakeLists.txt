cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
# set(CMAKE_VERBOSE_MAKEFILE 1)
set(CMAKE_BUILD_TYPE Debug)
project(ICP)
find_package(PCL 1.7 REQUIRED)
find_package(TBB REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenMP)
if (OPENMP_FOUND)
    message("openMP fouND!======================")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

aux_source_directory(src DIR_SRC)

add_executable(icp src/main.cpp)
target_link_libraries(icp ${PCL_LIBRARIES}  tbb)
# target_link_libraries(icp Threads::Threads)