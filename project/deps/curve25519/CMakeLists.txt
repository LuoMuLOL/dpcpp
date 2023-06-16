# CMakeList.txt: curve25519 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#指定运行该CMakeLists.txt文件的最低版本
cmake_minimum_required (VERSION 3.8)

# 如果支持，可以为MSVC编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()


# 项目定义
set(This curve25519)
#设置这个CMake项目的名称为MyProject，This是一个变量名
project (${This} LANGUAGES CXX)
#定义了一个项目，指定项目名和使用的语言类型为C++
#用来检查名为DPCPP_CPDIR的环境变量是否定义
if (DEFINED ENV{DPCPP_CPDIR}) 
  set(_TARGET ${This})
elseif (DEFINED ENV{DPCPP_CUDA_CPDIR}) 
  set(_TARGET ${This})
endif()
#检查环境变量，为CMake项目定义不同的编译器选项，从而生成适当的项目构建规则

# 头文件&源文件定义
set(Headers
  # curve25519.h
  curve25519_donna.h
)
set(Sources
  curve25519_donna.cpp
  test.cpp
)
add_executable(${_TARGET}
  ${Headers}
  ${Sources}
)

# 子项目定义

# add_subdirectory(deps/googletest)
include_directories(../deps/curve25519)


# 编译环境设置
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
# set(CMAKE_CXX_FLAGS_DEBUG "/MTd /Zi /Ob0 /Od /RTC1 /W4")
target_compile_options(${_TARGET}
  PRIVATE "-g" "-Wall"
)
if (CMAKE_VERSION VERSION_GREATER 3.12)
   set_property(TARGET ${_TARGET} PROPERTY CXX_STANDARD 20)
endif()


# 添加Sodium库
find_library(SODIUM_LIBRARY NAMES sodium PATHS /usr/lib/x86_64-linux-gnu/)
include_directories(/usr/include/)
target_link_libraries(${_TARGET} ${SODIUM_LIBRARY})


message("---------------curve25519----------------------------------")
if (DEFINED ENV{DPCPP_CPDIR}) # DEFINED必须大写
  message("find DPCPP_CPDIR")  
  set(ENV{PATH} $ENV{DPCPP_CPDIR}/build/bin:$ENV{PATH} )
  set(ENV{LD_LIBRARY_PATH} $ENV{DPCPP_CPDIR}/build/lib:$ENV{LD_LIBRARY_PATH} )
# cuda版本
elseif (DEFINED ENV{DPCPP_CUDA_CPDIR}) 
  message("find DPCPP_CUDA_CPDIR")
  set(ENV{PATH} $ENV{DPCPP_CUDA_CPDIR}/build/bin:$ENV{PATH} )
  set(ENV{LD_LIBRARY_PATH} $ENV{DPCPP_CUDA_CPDIR}/build/lib:$ENV{LD_LIBRARY_PATH} )
  target_compile_options(${_TARGET}
    PRIVATE "-fsycl-targets=nvptx64-nvidia-cuda"
  ) 
  target_link_options(${_TARGET}
    PRIVATE "-fsycl-targets=nvptx64-nvidia-cuda"
  ) 
else()
  message("not find")
endif()

# message($ENV{PATH})
# message($ENV{LD_LIBRARY_PATH})
# message(${This})
# message("-----------------------------------------------------")

# # 生成普通版本
# set(target_base ${This})
# set(_DPCPP_CPDIR "/home/axdhy/____WorkSpace____/C_C++_WorkSpace/sycl_workspace/llvm")
# set(CMAKE_C_COMPILER ${_DPCPP_CPDIR}/build/bin/clang)
# set(CMAKE_CXX_COMPILER ${_DPCPP_CPDIR}/build/bin/clang++)
# set(ENV{PATH} ${_DPCPP_CPDIR}/build/bin:${_PATH} )
# set(ENV{LD_LIBRARY_PATH} ${_DPCPP_CPDIR}/build/lib:${_LD_LIBRARY_PATH} )
# set(CMAKE_CXX_FLAGS ${_CMAKE_CXX_FLAGS} )
# add_executable (${target_base} "curve25519.cpp" "curve25519.h")

# # 生成cuda版本
# set(target_cuda ${This}-cuda)
# set(_DPCPP_CUDA_CPDIR "/home/axdhy/____WorkSpace____/C_C++_WorkSpace/sycl_workspace/llvm-cuda")
# set(CMAKE_C_COMPILER ${_DPCPP_CUDA_CPDIR}/build/bin/clang)
# set(CMAKE_CXX_COMPILER ${_DPCPP_CUDA_CPDIR}/build/bin/clang++)
# set(ENV{PATH} ${_DPCPP_CUDA_CPDIR}/build/bin:${_PATH} )
# set(ENV{LD_LIBRARY_PATH} ${_DPCPP_CUDA_CPDIR}/build/lib:${_LD_LIBRARY_PATH} )
# set(CMAKE_CXX_FLAGS "${_CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")
# add_executable (${target_cuda} "curve25519.cpp" "curve25519.h")



# 将源代码添加到此项目的可执行文件。
# add_executable (${This} "curve25519.cpp" "curve25519.h")
# add_executable (DPCplusplusPRGPractice_cuda "curve25519.cpp" "curve25519.h")

# TODO: 如有需要，请添加测试并安装目标。
