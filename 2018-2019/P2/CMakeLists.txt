CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(practicas_fsiv)
FIND_PACKAGE(OpenCV 	REQUIRED )
include_directories ("${OpenCV_INCLUDE_DIRS}")
LINK_LIBRARIES(${OpenCV_LIBS} )
add_executable(unsharp src/unsharp3.cpp)

#add_executable(pract1 pract1.cc)
#add_executable(pract2 pract2.cc)
#...