configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/KTensor.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/KTensor.hpp
    @ONLY)

set(KTENSOR_HEADER_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/KTensor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Auxiliary.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Checks.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Expression.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Functions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/IO.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Macros.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NestedLoops.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Operations.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SpecialTensors.hpp
    CACHE INTERNAL "" FORCE)