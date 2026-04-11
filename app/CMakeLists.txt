include(GenerateExportHeader)

add_executable(main main.cpp)
add_executable(mlirtoy::main ALIAS main)

target_link_libraries(main PRIVATE simple_library mlirtoy_options mlirtoy_warnings)
target_include_directories(main ${WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                                  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)
target_compile_features(main PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
set_target_properties(
  main
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN YES)
