function(mlirtoy_set_project_warnings project_name)
  # 1. Warnings shared by BOTH Clang and GCC
  set(BASE_WARNINGS
      -Wall
      -Wextra
      -Wshadow               # warn if variable declaration shadows one from a parent context
      -Wnon-virtual-dtor     # warn if a class with virtual functions has a non-virtual destructor
      -Wold-style-cast       # warn for c-style casts
      -Wcast-align           # warn for potential performance problem casts
      -Wunused               # warn on anything being unused
      -Woverloaded-virtual   # warn if you overload (not override) a virtual function
      -Wpedantic             # warn if non-standard C++ is used
      -Wconversion           # warn on type conversions that may lose data
      -Wsign-conversion      # warn on sign conversions
      -Wnull-dereference     # warn if a null dereference is detected
      -Wdouble-promotion     # warn if float is implicit promoted to double
      -Wformat=2             # warn on security issues around formatting functions (printf)
      -Wimplicit-fallthrough # warn on statements that fallthrough without explicit annotation
  )

  # 2. Warnings exclusive to GCC
  set(GCC_SPECIFIC_WARNINGS
      -Wmisleading-indentation # warn if indentation implies blocks where they don't exist
      -Wduplicated-cond        # warn if if/else chain has duplicated conditions
      -Wduplicated-branches    # warn if if/else branches have duplicated code
      -Wlogical-op             # warn about logical operations used where bitwise were wanted
      -Wuseless-cast           # warn if you perform a cast to the same type
      -Wsuggest-override       # warn if overridden member function isn't marked 'override'/'final'
  )

  # 4. Resolve which flags to use based on the current compiler
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(PROJECT_WARNINGS ${BASE_WARNINGS})
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(PROJECT_WARNINGS ${BASE_WARNINGS} ${GCC_SPECIFIC_WARNINGS})
  else()
    message(AUTHOR_WARNING "No compiler warnings set for '${CMAKE_CXX_COMPILER_ID}'. Only GCC and Clang are supported by this macro.")
  endif()

  # 5. Apply the warnings to the target for both C and C++
  target_compile_options(
    ${project_name}
    INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:${PROJECT_WARNINGS}>
      $<$<COMPILE_LANGUAGE:C>:${PROJECT_WARNINGS}>
  )

endfunction()
