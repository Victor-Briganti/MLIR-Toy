# Add the LLVM IR generation tests
add_test(NAME toy-llvm-tests
         COMMAND ${LIT_EXECUTABLE} -v ${CMAKE_CURRENT_SOURCE_DIR})
