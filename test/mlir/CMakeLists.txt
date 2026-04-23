# Add the MLIR transformation tests
add_test(NAME toy-mlir-tests
         COMMAND ${LIT_EXECUTABLE} -v ${CMAKE_CURRENT_SOURCE_DIR})
