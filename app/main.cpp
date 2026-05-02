//===--- main.cpp - The Toy Compiler --------------------------------------===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <string>

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/Lexer.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"
#include "toy/Passes.h"
#include "ErrorCode.h"

using namespace toy;
namespace cl = llvm::cl;

//===------------------------------------------------------------------------===
// CLI Options
//===------------------------------------------------------------------------===

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum class InputType { Toy, MLIR };
} // namespace
static cl::opt<InputType>
    inputType("x", cl::init(InputType::Toy),
              cl::desc("Decided the kind of output desired"),
              cl::values(clEnumValN(InputType::Toy, "toy",
                                    "load the input file as a Toy source")),
              cl::values(clEnumValN(InputType::MLIR, "mlir",
                                    "load the input file as a MLIR file")));

namespace {
enum class Action {
  None,
  DumpAST,
  DumpMLIRToy,
  DumpMLIRAffine,
  DumpMLIRLLVM,
  DumpLLVMIR
};
} // namespace
static cl::opt<Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(Action::DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(Action::DumpMLIRToy, "mlir",
                          "output the MLIR Toy dialect dump")),
    cl::values(clEnumValN(Action::DumpMLIRAffine, "mlir-affine",
                          "output the MLIR Affine dialect dump")),
    cl::values(clEnumValN(Action::DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR LLVM dialect dump")),
    cl::values(clEnumValN(Action::DumpLLVMIR, "llvm",
                          "output the LLVM IR dump"))

);

static cl::opt<bool>
    enableOpt("opt", cl::desc("Enable the optimizations in code generation"));

static cl::opt<bool> enablePrintPasses(
    "print-llvm-passes",
    cl::desc("Print all the passes being used in the LLVM IR"));

//===------------------------------------------------------------------------===
// Parsers (Toy and MLIR)
//===------------------------------------------------------------------------===

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
static std::unique_ptr<toy::ModuleAST>
parseInputFile(llvm::StringRef filename) {
  // Reads the source code to a MemoryBuffer or receive it directly from the
  // stdin when using the '-'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);

  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }

  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

/// Fill the module with the compiled code.
static int loadMLIR(mlir::MLIRContext &context,
                    mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Handle a '.toy' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST) {
      return TOY_PARSE_FAIL;
    }

    module = mlirGen(context, *moduleAST);
    return !module ? TOY_GEN_FAIL : TOY_SUCCESS;
  }

  // Otherwise, the input is '.mlir'
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return TOY_INPUT_INVALID;
  }

  // Parse the input MLIR.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return TOY_PARSE_FAIL;
  }

  return TOY_SUCCESS;
}

//===------------------------------------------------------------------------===
// Optimizer Pipeline
//===------------------------------------------------------------------------===

static int loadAndProcessMLIR(mlir::MLIRContext &context,
                              mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (int error = loadMLIR(context, module)) {
    return error;
  }

  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Could not apply the command line passes\n";
    return TOY_CLI_PASS_FAIL;
  }

  // Check to see what granularity of MLIR we are compiling to.
  bool loweringToAffine = emitAction >= Action::DumpMLIRAffine;
  bool loweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

  if (enableOpt || loweringToAffine) {
    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each
    // operation.
    mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
    optPM.addPass(mlir::toy::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (loweringToAffine) {
    // Partially lower the toy dialect
    pm.addPass(mlir::toy::createLowerToAffinePass());

    // Add a few cleanups post lowering
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    // Add optimizations if enabled.
    if (enableOpt) {
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }

  if (loweringToLLVM) {
    // Fully lower the toy dialect
    pm.addPass(mlir::toy::createLowerToLLVMPass());
    // This is necessary to have line tables emitted and basic debugger working.
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Could not run optimizers passes on the module\n";
    return TOY_PIPELINE_PASS_FAIL;
  }

  return TOY_SUCCESS;
}

//===------------------------------------------------------------------------===
// Dump
//===------------------------------------------------------------------------===

static int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
    return TOY_INPUT_INVALID;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST) {
    return TOY_PARSE_FAIL;
  }

  dump(*moduleAST);
  return TOY_SUCCESS;
}

//===------------------------------------------------------------------------===
// Compiler Entry Point
//===------------------------------------------------------------------------===

int main(int argc, char **argv) {
  // Initialize the infraestructure to give better errors when using LLVM code.
  llvm::InitLLVM start(argc, argv);
  
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  // Initialize the target as the current architecture.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  if (emitAction == Action::DumpAST) {
    dumpAST();
    return TOY_SUCCESS;
  }

  // If we aren't dumping the AST, then we are compiling with/to MLIR.
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);

  mlir::MLIRContext context(registry);
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module)) {
    return error;
  }

  // If we aren't exporting to non-mlir, then we are done.
  if (Action::DumpMLIRLLVM >= emitAction) {
    module->dump();
    return TOY_SUCCESS;
  }

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return TOY_UNDEFINED;
}
