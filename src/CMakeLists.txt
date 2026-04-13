add_library(ast AST.cpp)
add_library(mlirtoy::ast ALIAS ast)

mlirtoy_link_llvm_components(ast Support Core)
target_link_libraries(ast PRIVATE mlirtoy_options mlirtoy_warnings)
target_include_directories(ast 
                           ${WARNING_GUARD} 
                           PUBLIC 
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)
target_compile_features(ast PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
set_target_properties(ast
                      PROPERTIES 
                      VERSION 
                      ${PROJECT_VERSION}
                      CXX_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN YES)
