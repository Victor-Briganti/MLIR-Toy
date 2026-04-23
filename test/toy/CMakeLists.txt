# Add the compiler generation code
add_test(NAME toy-compiler-tests
         COMMAND ${LIT_EXECUTABLE} -v ${CMAKE_CURRENT_SOURCE_DIR})
