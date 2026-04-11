include(GenerateExportHeader)

add_library(simple_library simple_library.cpp)
add_library(mlirtoy::simple_library ALIAS simple_library)

target_link_libraries(simple_library PRIVATE mlirtoy_options mlirtoy_warnings)
target_include_directories(simple_library ${WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                                  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)
target_compile_features(simple_library PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
set_target_properties(
  simple_library
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN YES)

generate_export_header(simple_library EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/mlirtoy/simple_library_export.h)
