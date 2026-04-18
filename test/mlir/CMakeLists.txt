# ---- MLIR Integration Tests (lit) ----

# Path to the lit script provided by the user
set(LIT_EXECUTABLE "lit")

# Create a custom target to run the lit tests
add_custom_target(check-toy
  COMMAND ${CMAKE_COMMAND} -E echo "Running Toy MLIR tests..."
  COMMAND ${LIT_EXECUTABLE} -v ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS main # Ensure the compiler is built before running tests
  COMMENT "Running MLIR integration tests with lit"
  VERBATIM
)

# Also add them as a ctest test
add_test(NAME toy-mlir-tests
         COMMAND ${LIT_EXECUTABLE} -v ${CMAKE_CURRENT_SOURCE_DIR})
