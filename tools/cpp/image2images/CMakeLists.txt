project(image2images)

# Inlucde header files
include_directories(include)

# Add exectuable
add_executable(${PROJECT_NAME} src/Image.cpp src/File.cpp src/image2images.cpp)

# Add opencv
find_package( OpenCV REQUIRED )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
