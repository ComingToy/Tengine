# 0. clear var
UNSET (_DEV_EIGEN3_HEADER_PATH)
UNSET (_EIGEN3_BASE_SOURCE)
UNSET (_EIGEN3_OPS_SOURCE)
UNSET (_DEV_EIGEN3_DEVICE_SOURCE)
UNSET (_DEV_EIGEN3_COMPILER_DEFINES)
UNSET (_DEV_EIGEN3_COMPILER_OPTIONS)
UNSET (_DEV_EIGEN3_LINKER_OPTIONS)
UNSET (_DEV_EIGEN3_LINK_LIBRARIES)


# 1.  set source root path
SET(_EIGEN3_ROOT ${CMAKE_SOURCE_DIR}/source/device/eigen3)


# 2.  add header file path
LIST (APPEND _DEV_EIGEN3_HEADER_PATH      ${_EIGEN3_ROOT})


# 3.  add linking lib searching path

# 4.  add source files
AUX_SOURCE_DIRECTORY("${_EIGEN3_ROOT}"    _EIGEN3_BASE_SOURCE)
AUX_SOURCE_DIRECTORY("${_EIGEN3_ROOT}/op" _EIGEN3_OPS_SOURCE)
LIST (APPEND _DEV_EIGEN3_DEVICE_SOURCE    ${_EIGEN3_BASE_SOURCE})
LIST (APPEND _DEV_EIGEN3_DEVICE_SOURCE    ${_EIGEN3_OPS_SOURCE})


# 5.  add build options for cpu device
# 5.1 is a gcc or clang like compiler
# LIST (APPEND _DEV_EIGEN3_COMPILER_OPTIONS -std=c++11)


# 5.2 is Microsoft Visual C++
IF (TENGINE_COMPILER_MSVC)
ENDIF()


# 6.  add link options


# 7.  add link libs

# 8. set all to cmake cache
SET (TENGINE_EIGEN3_HEADER_PATH       ${_DEV_EIGEN3_HEADER_PATH}        CACHE INTERNAL  "Tengine Eigen3 device header files searching path"   FORCE)
SET (TENGINE_EIGEN3_LINK_PATH         ${_DEV_EIGEN3_LINK_PATH}          CACHE INTERNAL  "Tengine Eigen3 device link libraries searching path" FORCE)
SET (TENGINE_EIGEN3_DEVICE_SOURCE     ${_DEV_EIGEN3_DEVICE_SOURCE}      CACHE INTERNAL  "Tengine Eigen3 device main source files"             FORCE)
SET (TENGINE_EIGEN3_COMPILER_DEFINES  ${_DEV_EIGEN3_COMPILER_DEFINES}   CACHE INTERNAL  "Tengine Eigen3 about compiler defines"               FORCE)
SET (TENGINE_EIGEN3_COMPILER_OPTIONS  ${_DEV_EIGEN3_COMPILER_OPTIONS}   CACHE INTERNAL  "Tengine Eigen3 about compiler options"               FORCE)
SET (TENGINE_EIGEN3_LINKER_OPTIONS    ${_DEV_EIGEN3_LINKER_OPTIONS}     CACHE INTERNAL  "Tengine Eigen3 about linker options"                 FORCE)
SET (TENGINE_EIGEN3_LINK_LIBRARIES    ${_DEV_EIGEN3_LINK_LIBRARIES}     CACHE INTERNAL  "Tengine Eigen3 about link libraries"                 FORCE)


# 9. install device option
INSTALL (FILES ${_EIGEN3_ROOT}/eigen3_define.h DESTINATION include/tengine RENAME eigen3_device.h)
