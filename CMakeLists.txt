set(SOURCES
    "src/simple_visual_odometry.cpp"
    "src/interface.cpp"
)

set(HEADERS
    "include/simple_visual_odometry.h"
    "include/vo_features.h"
)

include_directories(include)

add_library(simple_visual_odometry MODULE ${SOURCES} ${HEADERS})
target_link_libraries(simple_visual_odometry PRIVATE lmscore ${CONAN_LIBS} lms_imaging)
