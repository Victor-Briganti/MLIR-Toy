# MLIR Toy

This repository contains an implementation of the Toy language using MLIR, developed out-of-tree from the main LLVM source tree.

The goal of this project is to serve as a learning resource for understanding the different phases involved in MLIR-based code generation, including parsing, intermediate representation construction, optimization, and lowering.

## Getting Started

### Dependencies

This project has a direct dependency on the LLVM and MLIR development libraries.
Make sure they are properly installed on your system before building the project.

Typical requirements include:

- LLVM
- CMake
- A C++ compiler with C++17 (or newer) support

### Compiling

The project is structured to be built using CMake.
To configure and build the project, run:

```bash
cmake -S . -B build
cmake --build build
```

### Testing

For testing this project uses the [Catch2](https://github.com/catchorg/Catch2) framework. The code can be tested using the following command:

```bash
ctest --test-dir build
```

## License

This project is licensed under the MIT License.
See the LICENSE file for details.

## Acknowledgments

This project is heavily inspired by the Toy language example implemented in the official LLVM source tree and documentation.
