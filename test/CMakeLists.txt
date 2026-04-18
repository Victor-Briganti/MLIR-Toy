include(FetchContent)

# Disable deprecated-literal-operator warning for Catch2
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  add_compile_options(-Wno-deprecated-literal-operator)
endif()

# Fetch Catch2 v3
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0
)

FetchContent_MakeAvailable(Catch2)

# Include subdirectories for different test types
add_subdirectory(lexer)
add_subdirectory(parser)
add_subdirectory(mlir)
