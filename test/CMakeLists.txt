include(FetchContent)

# Fetch Catch2 v3
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0 # or a later version
)

FetchContent_MakeAvailable(Catch2)

# ---- Lexer Tests ----
add_executable(lexer_tests lexer_tests.cpp)
mlirtoy_link_llvm_components(lexer_tests)
target_link_libraries(lexer_tests
                      PRIVATE 
                      mlirtoy::mlirtoy_options
                      mlirtoy::mlirtoy_warnings
                      Catch2::Catch2WithMain)

target_include_directories(lexer_tests 
                           PRIVATE 
                           ${PROJECT_SOURCE_DIR}/include)

# Let Catch2 discover tests
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
catch_discover_tests(lexer_tests)
