######## A simple cmakelists.txt file for OpenCV() #############  
cmake_minimum_required(VERSION 3.0)                            
PROJECT(main)                                         

FIND_PACKAGE( OpenCV REQUIRED )                              
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

ADD_EXECUTABLE(main main.cpp)                         
TARGET_LINK_LIBRARIES (main ${OpenCV_LIBS})         
########### end ####################################  