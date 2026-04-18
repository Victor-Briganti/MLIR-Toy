# ---- Parser Tests ----
add_executable(parser_tests parser_tests.cpp)
mlirtoy_link_llvm_components(parser_tests)
target_link_libraries(parser_tests
                      PRIVATE
                      mlirtoy::AST
                      mlirtoy::mlirtoy_options
                      mlirtoy::mlirtoy_warnings
                      Catch2::Catch2WithMain)

target_include_directories(parser_tests
                           PRIVATE
                           ${PROJECT_SOURCE_DIR}/include)

# Let Catch2 discover tests
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
catch_discover_tests(parser_tests)
