# 第1章：CUDA C编程及GPU基本知识

## 第1节: GPU基本架构及特点

- CPU和GPU架构
  - GPU：**吞吐**导向内核——>单位时间内处理的指令条数
  - CPU：**延迟**导向内核——>指令从发出到最终返回结果中间经历的时间间隔

- CPUs：延迟导向设计
  - 内存大
    - 多级缓存结构提高访存速度
  - 控制复杂
    - 分支预测机制
    - 流水线数据前送
  - 运算单元强大
    - 整型浮点型复杂运算速度快
- GPUs：吞吐导向设计
  - 缓存小
    - 提高内存吞吐
  - 控制简单
    - 没有分支预测
    - 没有数据转发
  - 精简运算单元
    - 多长延时流水线以实现高吞吐量
    - 需要大量线程来容忍延迟
- GPU&CPU特点
  - CPUs：连续计算部分，延迟优先
    - CPU比GPU，单条复杂指令延迟快10倍以上
  - GPUs：并行计算部分，吞吐优先
    - GPU比CPU，单位时间内执行指令数量10倍以上

## 第2节: CUDA C编程基本知识

- 什么样的问题适合GPU
  - 计算密集：数值计算的比例要远大于内存操作，因此内存访问的延时可以被计算掩盖；
  - 数据并行：大任务可以拆解为执行相同指令的小任务，因此对复杂流程控制的需求较低；

### GPU编程与CUDA

- CUDA (Compute Unified Device Architecture)，由英伟达公司2007年开始推出，初衷是为GPU增加一个易用的编程接口，让开发者无需学习复杂的着色语言或者图形处理原语；
- OpenCL (Open Computing Languge)，是2008年发布的异构平台并行编程的开放标准，也是一个编程框架。OpenCL相比于CUDA，支持的平台更多，除了GPU还支持CPU、DSP、FPGA等设备。

### CUDA编程并行计算整体流程

```c
void GPUkernel (float *A, float *B, float *C, int n)
{
    // 1. Allocate device memory for A, B, and C
    // copy A and B to device memory
    
    // 2. Kernel launch code - to have the device
    // to perform the actual vector addition
    
    // 3. copy C from the device memory
    // Free device vectors
}
```

### CUDA编程术语：

- 硬件
  - Device = GPU
  - Host = CPU
  - Kernel = GPU上运行的函数
  - 内存模型
    - CUDA中的内存模型分为以下几个层次：
      - 每个线程处理器(SP)都用自己的registers（寄存器）
      - 每个SP都有自己的local memory（局部内存），register和local memory只能被线程自己访问
      - 每个多核处理器(SM)内都有自己的shared memory（共享内存），shared memory可以被线程块内所有线程访问
      - 一个GPU的所有SM共有一块global memory（全局内存），不同线程块的线程都可使用

- 软件

  - CUDA中的内存模型分为以下几个层次：

    - 线程处理器(SP)对应线程(thread)
    - 多核处理器(SM)对应线程块(thread block)
    - 设备端(device)对应线程块组合体(grid)

  - 一个kernel其实由一个grid来执行

  - 一个kernel一次只能在一个GPU上执行

  - 线程块：可扩展的集合体

    - 将线程数组分为多个块
    - 块内的线程通过共享内存、原子操作和屏障同步进行协作(shared memory, atomic operations and barrier synchronization)
    - 不同块中的线程不能协作

  - 网格(grid)：并行线程块组合

    - CUDA核函数由线程网格（数组）执行
      - 每个线程都有一个索引，用于计算内存地址和做出控制决策

  - 线程块id&线程id：定位独立线程的门牌号

    - 每个线程使用索引来决定要处理的数据

      - blockIdx：1D, 2D, or 3D
      - threadIdx：1D, 2D, or 3D

    - 线程id计算

      ```c
      dim3 dimGrid(M, N);
      dim3 dimBlock(P, Q, S);
      
      threadId.x = blockIdx.x*blockDim.x + threadIdx.x;
      threadId.y = blockIdx.y*blockDim.y + threadIdx.y;
      ```

  - 线程束(warp)

    - SM采用的SIMT (Single-Instruction, Multiple-Thread，单指令多线程)架构，warp（线程束）是最基本的执行单元，一个warp包含32个并行thread，这些thread以不同数据资源执行相同的指令。warp本质上是线程在GPU上运行的最小单元。
    - 当一个kernel被执行时，grid中的线程块被分配到SM上，一个线程块的thread只能在一个SM上调度，SM一般可以调度多个线程块，大量的thread可能被分到不同的SM上。每个thread拥有它自己的程序计数器和状态寄存器，并且用该线程自己的数据执行指令，这就是所谓的Single-Instruction Multiple-Thread (SIMT)。
    - 由于warp的大小为32，所以block所含的thread大小一般要设置为32的倍数。

## 第3节: 并行计算实例：向量相加

```c
// Compute vector sum C = A + B
void vecAdd(float *A, float *B, float *C, int n)
{
    for (i = 0, i < n, i++)
        C[i] = A[i] + B[i];
}

int main()
{
    // Memory allocation for A_h, B_h, and C_h
    	// I/O to read A_h, and B_h, N elements
    	// ...
    	vecAdd(A_h, B_h, C_h, N);
}
```

```c
void vecAdd(float *A, float *B, float *C, int n)
{
    int size = n * sizeof(float);
    float* A_d, B_d, C_d;
    // ...
    
    // 1. Allocate device memory for A, B, and C
    // copy A and B to device memory
    
    // 2. Kernel launch code - to have the device
    // to perform the actual vector addition
    
    // 3. copy C from the device memory
    // Free device vectors
}

```

- 设备端代码：

  - 读写线程寄存器
  - 读写Grid中全局内存
  - 读写block中共享内存

- 主机端代码：

  - Grid中全局内存拷贝转移

  - `cudaMalloc()`——申请显存

    - `cudaError_t cudaMalloc(void *devPtr, size_t size)`
    - 在设备全局内存中分配对象
    - 两个参数
      - 地址
      - 申请内存大小

  - `cudaFree()`——释放

    - `cudaError_t cudaFree(void *devPtr)`
    - 从设备全局内存中释放对象
    - 指向释放对象的指针

  - cudaMemcpy()

    - cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)

    - 内存数据复制传递

    - 目前支持的四种选项

      - cudaMemcpyHostToDevice
      - cudaMemcpyDeviceToHost
      - cudaMemcpyDeviceToDevice
      - cudaMemcpyDefault

    - 调用cudaMemcpy()传输内存是同步的

    - ```c
      void vecAdd(float *A, float *B, float *C, int n)
      {
          int size = n * sizeof(float);
          float* A_d, B_d, C_d;
          
          // 1. Transfer A and B to device memory
          cudaMalloc((void **) &A_d, size);
          cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
          cudaMalloc((void **) &B_d, size);
          cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
          
          // Allocate device memory for
          cudaMalloc((void **) &C_d, size);
          
          // 2. Kernel invocation code - to be shown later
          // ...
          
          // 3. Transfer C from device to host
          cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
          // Free device memory for A, B, C
          cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
      }
      
      ```

### 核函数调用

- 在GPU上执行的函数；
- 一般通过标识符`__global__`修饰；
- 调用通过`<<<参数1, 参数2>>>`，用于说明内核函数中的线程数量
