CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(practicas_fsiv)
FIND_PACKAGE(OpenCV 	REQUIRED )
include_directories ("${OpenCV_INCLUDE_DIRS}")
LINK_LIBRARIES(${OpenCV_LIBS} )
add_executable(examen examen.cpp)

#add_executable(pract1 pract1.cc)
#add_executable(pract2 pract2.cc)
#...