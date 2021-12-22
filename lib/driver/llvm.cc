/* Copyright 2015-2017 Philippe Tillet
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <fstream>
#include <unistd.h>
#include <memory>
#include <regex>
#include "triton/driver/llvm.h"
#include "triton/driver/dispatch.h"
#include "triton/driver/error.h"
#include "triton/tools/sha1.hpp"
#include "triton/tools/sys/getenv.hpp"
#include "triton/tools/sys/mkdir.hpp"
#include "triton/tools/sys/exec.hpp"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Transforms/Utils/Cloning.h"

// begin AMD stuff
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
// end AMD stuff

namespace triton{
namespace driver{

void init_llvm() {
  static bool init = false;
  if(!init){
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();
    init = true;
  }
}

/* ------------------------ */
//         CUDA             //
/* ------------------------ */
static bool find_and_replace(std::string& str, const std::string& begin, const std::string& end, const std::string& target){
  size_t start_replace = str.find(begin);
  size_t end_replace = str.find(end, start_replace);
  if(start_replace == std::string::npos)
    return false;
  str.replace(start_replace, end_replace + 1 - start_replace, target);
  return true;
}

int vptx(int version){
  if(version >= 11030) return 73;
  if(version >= 11020) return 72;
  if(version >= 11010) return 71;
  if(version >= 11000) return 70;
  if(version >= 10020) return 65;
  if(version >= 10010) return 64;
  if(version >= 10000) return 63;
  throw std::runtime_error("Triton requires CUDA 10+");
}

std::string llir_to_ptx(llvm::Module* module, int cc, int version){
  // LLVM version in use may not officially support target hardware
  int max_nvvm_cc = 75;
  int max_nvvm_ptx = 64;
  // options
  auto options = llvm::cl::getRegisteredOptions();
  auto* short_ptr = static_cast<llvm::cl::opt<bool>*>(options["nvptx-short-ptr"]);
  assert(short_ptr);
  short_ptr->setValue(true);
  // compute capability
  std::string sm = "sm_" + std::to_string(cc);
  // max PTX version
  int ptx = vptx(version);
  int ptx_major = ptx / 10;
  int ptx_minor = ptx % 10;
  // create
  llvm::SmallVector<char, 0> buffer;
  std::string triple = "nvptx64-nvidia-cuda";
  std::string proc = "sm_" + std::to_string(std::min(cc, max_nvvm_cc));
  std::string layout = "";
  std::string features = "+ptx" + std::to_string(std::min(ptx, max_nvvm_ptx));
  init_llvm();
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(*module);
  // create machine
  module->setTargetTriple(triple);
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  llvm::TargetOptions opt;
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  llvm::TargetMachine *machine = target->createTargetMachine(module->getTargetTriple(), proc, features, opt,
                                                             llvm::Reloc::PIC_, llvm::None, llvm::CodeGenOpt::Aggressive);
  // set data layout
  if(layout.empty())
    module->setDataLayout(machine->createDataLayout());
  else
    module->setDataLayout(layout);
  // emit machine code
  for (llvm::Function &f : module->functions())
    f.addFnAttr(llvm::Attribute::AlwaysInline);
  llvm::legacy::PassManager pass;
  llvm::raw_svector_ostream stream(buffer);
  // emit
  machine->addPassesToEmitFile(pass, stream, nullptr, llvm::CodeGenFileType::CGFT_AssemblyFile);
  pass.run(*module);

  // post-process
  std::string result(buffer.begin(), buffer.end());
  find_and_replace(result, ".version", "\n", ".version " + std::to_string(ptx_major) + "." + std::to_string(ptx_minor) + "\n");
  find_and_replace(result, ".target", "\n", ".target " + sm + "\n");
  while(find_and_replace(result, "\t// begin inline asm", "\n", ""));
  while(find_and_replace(result, "\t// end inline asm", "\n", ""));
  return result;
}

std::string ptx_to_cubin(const std::string& ptx, int cc) {
    return "";
}

//CUmodule ptx_to_cumodule(const std::string& ptx, int cc) {
//  // JIT compile source-code
//  try{
//    // use ptxas if present in PATH. Otherwise, use JIT from the driver
//    std::string ptxas = "ptxas";
//    std::string version;
//    int use_system_ptxas = tools::exec(ptxas + " --version 2>&1", version) == 0;

//    // Use PTXAS via system call
//    if(use_system_ptxas){
//      // compile ptx with ptxas
//      char _fsrc[] = "/tmp/triton_k_XXXXXX";
//      char _flog[] = "/tmp/triton_l_XXXXXX";
//      mkstemp(_fsrc);
//      mkstemp(_flog);
//      std::string fsrc = _fsrc;
//      std::string flog = _flog;
//      std::string fbin = fsrc + ".o";
//      const char* _fbin = fbin.c_str();
//      std::ofstream ofs(fsrc);
//      ofs << ptx;
//      ofs.close();
//      std::string cmd;
//      int err;
//      cmd = ptxas + " -v --gpu-name=sm_" + std::to_string(cc) + " " + fsrc + " -o " + fsrc + ".o 2> " + flog;
//      err = system(cmd.c_str());
//      CUmodule ret;
//      std::ifstream _cubin(_fbin, std::ios::binary );
//      std::string cubin(std::istreambuf_iterator<char>(_cubin), {});
//      _cubin.close();
//      dispatch::cuModuleLoadData(&ret, cubin.c_str());
//      unlink(_fsrc);
//      unlink(_flog);
//      unlink(_fbin);
//      return ret;
//    }

//    // Use PTXAS included in driver
//    CUjit_option opt[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER,
//                          CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER,
//                          CU_JIT_LOG_VERBOSE};
//    unsigned int errbufsize = 8192;
//    unsigned int logbufsize = 8192;
//    char _err[errbufsize];
//    char _log[logbufsize];
//    void* optval[] = {(void*)(uintptr_t)errbufsize, (void*)_err, (void*)(uintptr_t)logbufsize, (void*)_log, (void*)1};
//    CUmodule ret;
//    dispatch::cuModuleLoadDataEx(&ret, ptx.data(), 5, opt, optval);
//    return ret;
//  }
//  catch(exception::cuda::invalid_ptx const &){
//    std::cout << ptx << std::endl;
//    std::cerr << "It appears that Triton produced invalid PTX code:" << std::endl;
//    throw;
//  }
//}

/* ------------------------ */
//         HIP              //
/* ------------------------ */

std::string llir_to_amdgpu(llvm::Module* module, const std::string& _proc) {
  init_llvm();

//  proc = std::get<0>(GetFeatureStrFromGCNArchName(rocminfo));
//  features = std::get<1>(GetFeatureStrFromGCNArchName(rocminfo));

  // create
  llvm::SmallVector<char, 0> buffer;
  std::string triple = "amdgcn-amd-amdhsa";
  std::string layout = "";
  std::string features;
  std::string proc = "gfx908";
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(*module);
  // create machine
  module->setTargetTriple(triple);
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  llvm::TargetOptions opt;
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  llvm::TargetMachine *machine = target->createTargetMachine(module->getTargetTriple(), proc, features, opt,
                                                             llvm::Reloc::PIC_, llvm::None,
                                                             llvm::CodeGenOpt::Aggressive);
  // set data layout
  if(layout.empty())
    module->setDataLayout(machine->createDataLayout());
  else
    module->setDataLayout(layout);
  // emit machine code
  for (llvm::Function &f : module->functions())
    f.addFnAttr(llvm::Attribute::AlwaysInline);
  llvm::legacy::PassManager pass;
  llvm::raw_svector_ostream stream(buffer);

  // create dump files
  std::string module_name = module->getModuleIdentifier();
  std::error_code ec;

  // Save GCN ISA binary.
  std::string isabin_path = std::string("/tmp/") + module_name + std::string(".o");
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(
      new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
  if (ec)
  {
    std::cout << isabin_path << " was not created. error code: " << ec << std::endl;
  }

  // emit
  machine->addPassesToEmitFile(pass, *isabin_fs, nullptr, llvm::CGFT_ObjectFile);
  pass.run(*module);
  // Save GCN ISA.
  std::string amdgcn_path = std::string("/tmp/") + module_name + std::string(".gcn");
  std::string result(buffer.begin(), buffer.end());
  std::ofstream amdgcn(amdgcn_path);
  amdgcn << result;
  amdgcn.close();

  // generate HASCO file
  std::string hsaco_path = std::string("/tmp/") + module_name + std::string(".hsaco");
  std::string error_message;
  int lld_result =
      llvm::sys::ExecuteAndWait("/opt/rocm/llvm/bin/ld.lld",
                                {"/opt/rocm/llvm/bin/ld.lld", "-flavor", "gnu", "-shared", "-o", hsaco_path, isabin_path},
                                llvm::None, {}, 0, 0, &error_message);
  if (lld_result)
  {
    std::cout << "ld.lld execute fail: " << std::endl;
    std::cout << error_message << std::endl;
    std::cout << lld_result << std::endl;
  }

  return hsaco_path;
}


hipModule_t amdgpu_to_hipmodule(const std::string& path) {
  // Read HSACO.
  std::ifstream hsaco_file(path, std::ios::binary | std::ios::ate);
  std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();

  std::vector<unsigned char> hsaco(hsaco_file_size);
  hsaco_file.seekg(0, std::ios::beg);
  hsaco_file.read(reinterpret_cast<char*>(&hsaco[0]), hsaco_file_size);
  hsaco_file.close();
  hipJitOption opt[] = {hipJitOptionErrorLogBufferSizeBytes, hipJitOptionErrorLogBuffer,
                            hipJitOptionInfoLogBufferSizeBytes, hipJitOptionInfoLogBuffer,
                            hipJitOptionLogVerbose};
  unsigned int errbufsize = 8192;
  unsigned int logbufsize = 8192;
  char _err[errbufsize];
  char _log[logbufsize];
  void* optval[] = {(void*)(uintptr_t)errbufsize,
                    (void*)_err, (void*)(uintptr_t)logbufsize,
                    (void*)_log, (void*)1};
  hipModule_t ret;
  dispatch::hipModuleLoadDataEx(&ret, hsaco.data(), 5, opt, optval);
  return ret;
}



}
}

