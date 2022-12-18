# CUDA_and_Deep_Learning
本仓库系个人学习[深蓝学院《CUDA入门与深度神经网络加速》](https://www.shenlanxueyuan.com/my/course/544)课程心得体会及笔记记录，在此分享供个人学习使用。试听课程可在[深蓝学院《CUDA入门与深度神经网络加速》](https://www.shenlanxueyuan.com/my/course/474)链接试听！更多视频内容参考深蓝学院官网，值得推荐选修，支持正版！

通过本课程的学习：

- 理论部分：
  - 学习认识GPU以及如何使用CUDA
  - 如何编程和维护
- 技能部分：
  - 并行计算的基本准则和样式
  - 并行处理器特征和限制
  - 使用方法

## [第1章：CUDA C编程及GPU基本知识](./doc/ch01_CUDA-C编程及GPU基本知识.md)

- 第1节: GPU基本架构及特点
- 第2节: CUDA C编程基本知识
- 第3节: 并行计算向量相加
- 第4节: 实践

## [第2章：CUDA C编程：矩阵乘法](./doc/ch02_矩阵乘法.md)

- 第1节: 为什么矩阵乘法适合GPU实现
- 第2节: 矩阵乘法的GPU基础实现
- 第3节: 矩阵乘法GPU进阶实现
- 第4节: 代码实践
- 第5节: 作业题目        

## 第3章：CUDA stream和Event

- 第1节: CUDA Stream介绍
- 第2节: CUDA Stream为什么有效
- 第3节:  CUDA Stream 默认流的表现
- 第4节: CUDA Event
- 第5节: CUDA 同步操作
- 第6节: NVVP工具演示        

## 第4章：cuDNN和cuBLAS

- 第1节: 课程回顾
- 第2节: cuBLAS
- 第3节: cuDNN
- 第4节: 实践        

## 第5章：TensorRT介绍

- 第1节:  TensorRT是什么
- 第2节: TensorRT整体工作流程与优化策略
- 第3节: TensorRT的组成与基本使用流程
- 第4节: TensorRT demo：SampleMNIST
- 第5节:  TensorRT进阶
- 第6节: Demo演示
- 第7节: 作业实践        

## 第6章：TensorRT plugin用法

- 第1节: Plugin介绍
- 第2节: Static Shape Plugin
- 第3节: Dynamic Shape Plugin
- 第4节: PluginCreator注册
- 第5节: 延伸：TensorRT如何debug 
- 第6节: 实践作业        

## 第7章：TensorRT量化加速

- 第1节: TRT FP16优化
- 第2节: TRT INT8量化算法
- 第3节: TRT大规模上线经验        
