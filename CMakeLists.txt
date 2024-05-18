# Since one could use std::mdspan, we may need C++23 standard support
# so we insist on CMake version >= 3.20 as per
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
cmake_minimum_required(VERSION 3.20)

project("KTensor" VERSION 1.0.0
        DESCRIPTION "A header-only C++ library that enables Einstein summation for arbitrary array-like structures with compile-time error checking."
        LANGUAGES CXX)  # TODO: Add HOMEPAGE_URL here

include(GNUInstallDirs)
include(CMakeDependentOption)

add_library(KTensor INTERFACE)  # Interface since the library is header-only

set(KTENSOR_MDSPAN_NAMESPACE "Kokkos" CACHE STRING "Namespace in which mdspan resides")
set_property(CACHE KTENSOR_MDSPAN_NAMESPACE PROPERTY STRINGS std Kokkos)

OPTION(ENABLE_TESTS "Enable testing with CTest" OFF)
cmake_dependent_option(KTENSOR_RANDOM_INIT "Member function to initialize MDTensors to random integers" ON "NOT ENABLE_TESTS" ON)

# KTENSOR_RANDOM_INIT_FLAG becomes the value to which KTENSOR_RANDOM_INIT
# is defined in the configured header file. Using 1 and 0 as the flags lets
# those symbols work in #if directives. This way we can control behavior
# across all the header files by configuring only one file at build time and
# leaving the others untouched after installation.
if(KTENSOR_RANDOM_INIT)
    set(KTENSOR_RANDOM_INIT_FLAG "1" CACHE INTERNAL "" FORCE)
else()
    set(KTENSOR_RANDOM_INIT_FLAG "0" CACHE INTERNAL "" FORCE)
endif()

add_subdirectory(KTensor)

target_include_directories(
  KTensor
  INTERFACE $<BUILD_INTERFACE>:${CMAKE_CURRENT_SOURCE_DIR}/KTensor
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Set the CXX standard based on the mdspan namespace
if(KTENSOR_MDSPAN_NAMESPACE STREQUAL std)
    # mdspan is only in namespace std as of C++23
    set(KTENSOR_CXX_STANDARD 23 CACHE INTERNAL "" FORCE)
    set(KTENSOR_USE_KOKKOS_FLAG "0" CACHE INTERNAL "" FORCE)
else()
    # Kokkos mdspan could be used with C++20 or later
    set(KTENSOR_CXX_STANDARD 20 CACHE INTERNAL "" FORCE)
    
    set(KTENSOR_USE_KOKKOS_FLAG "1" CACHE INTERNAL "" FORCE)
    set(KTENSOR_KOKKOS_MDSPAN_DIR "/usr/local/include/mdspan" CACHE STRING "Directory containing mdspan.hpp")
endif()

set_property(TARGET KTensor PROPERTY CXX_STANDARD ${KTENSOR_CXX_STANDARD})

install(TARGETS KTensor
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

include(CMakePackageConfigHelpers)
        write_basic_package_version_file("KTensorConfigVersion.cmake"
                                         VERSION ${PROJECT_VERSION}
                                         COMPATIBILITY SameMajorVersion)
                                         
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(FILES ${KTENSOR_HEADER_FILES}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

if (ENABLE_TESTS)
    # Need Python in order to run the tests
    find_package(
        Python3
        REQUIRED
        COMPONENTS Interpreter
    )
    enable_testing()
    add_subdirectory(tests)
endif()