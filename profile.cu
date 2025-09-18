#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 根据GPU架构估算理论峰值性能（GFLOPS）
double getTheoreticalGFLOPS(int major, int minor, int multiProcessorCount, int clockRate) {
    double gflops = 0.0;
    
    // CUDA核心数估算（每个SM的核心数）
    int coresPerSM = 0;
    
    if (major == 2) {
        coresPerSM = 32;  // Fermi
    } else if (major == 3) {
        coresPerSM = 192; // Kepler
    } else if (major == 5) {
        coresPerSM = 128; // Maxwell
    } else if (major == 6) {
        if (minor == 0) coresPerSM = 64;   // Pascal GP100
        else coresPerSM = 128;             // Pascal GP10x
    } else if (major == 7) {
        if (minor == 0) coresPerSM = 64;   // Volta V100
        else coresPerSM = 64;              // Turing
    } else if (major == 8) {
        if (minor == 0) coresPerSM = 64;   // Ampere A100
        else coresPerSM = 128;             // Ampere GA10x
    } else if (major == 9) {
        coresPerSM = 128; // Ada Lovelace / Hopper (估算)
    } else {
        coresPerSM = 64;  // 默认估算
    }
    
    int totalCores = coresPerSM * multiProcessorCount;
    // 理论峰值 = 核心数 × 时钟频率 × 2 (FMA指令可以同时做乘法和加法)
    gflops = (totalCores * clockRate * 2.0) / 1000000.0; // 转换为GFLOPS
    
    return gflops;
}

// 计算内存带宽 (GB/s)
double getMemoryBandwidth(int memoryClockRate, int memoryBusWidth) {
    // 内存带宽 = 内存时钟频率 × 总线宽度 × 2(DDR) / 8(位到字节)
    return (memoryClockRate * 2.0 * memoryBusWidth) / (8.0 * 1000.0);
}

// 计算算术强度的界限点
double calculateBoundaryArithmeticIntensity(double peakGFLOPS, double memoryBandwidthGB) {
    // 界限点：算术强度 = 峰值计算性能 / 内存带宽
    return peakGFLOPS / memoryBandwidthGB;
}

int main() {
    std::cout << "=== GPU Memory Bound vs Computing Bound 分析 ===\n\n";
    
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cout << "没有发现CUDA设备！\n";
        return 1;
    }
    
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        
        std::cout << "=== 设备 " << device << ": " << prop.name << " ===\n";
        
        // 基本硬件信息
        std::cout << "\n硬件规格:\n";
        std::cout << "  计算能力: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  流多处理器(SM)数量: " << prop.multiProcessorCount << "\n";
        std::cout << "  基础时钟频率: " << prop.clockRate / 1000 << " MHz\n";
        std::cout << "  内存时钟频率: " << prop.memoryClockRate / 1000 << " MHz\n";
        std::cout << "  内存总线宽度: " << prop.memoryBusWidth << " bit\n";
        std::cout << "  全局内存大小: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n";
        std::cout << "  共享内存大小: " << prop.sharedMemPerBlock / 1024 << " KB per block\n";
        
        // 计算理论峰值性能
        double theoreticalGFLOPS = getTheoreticalGFLOPS(prop.major, prop.minor, 
                                                       prop.multiProcessorCount, 
                                                       prop.clockRate);
        
        // 计算内存带宽
        double memoryBandwidth = getMemoryBandwidth(prop.memoryClockRate, prop.memoryBusWidth);
        
        // 计算界限点
        double boundaryAI = calculateBoundaryArithmeticIntensity(theoreticalGFLOPS, memoryBandwidth);
        
        std::cout << "\n性能分析:\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  理论峰值计算性能: " << theoreticalGFLOPS << " GFLOPS\n";
        std::cout << "  理论峰值内存带宽: " << memoryBandwidth << " GB/s\n";
        std::cout << "  界限算术强度: " << boundaryAI << " FLOP/Byte\n";
        
        std::cout << "\nMemory Bound vs Computing Bound 分析:\n";
        std::cout << "  当算术强度 < " << boundaryAI << " FLOP/Byte 时: Memory Bound\n";
        std::cout << "  当算术强度 > " << boundaryAI << " FLOP/Byte 时: Computing Bound\n";
        
        // 实际应用示例
        std::cout << "\n常见操作的算术强度参考:\n";
        std::cout << "  向量加法 (C = A + B): ~0.33 FLOP/Byte";
        if (0.33 < boundaryAI) std::cout << " → Memory Bound\n";
        else std::cout << " → Computing Bound\n";
        
        std::cout << "  矩阵-向量乘法: ~2 FLOP/Byte";
        if (2.0 < boundaryAI) std::cout << " → Memory Bound\n";
        else std::cout << " → Computing Bound\n";
        
        std::cout << "  小矩阵乘法 (64x64): ~8 FLOP/Byte";
        if (8.0 < boundaryAI) std::cout << " → Memory Bound\n";
        else std::cout << " → Computing Bound\n";
        
        std::cout << "  大矩阵乘法 (1024x1024+): ~40+ FLOP/Byte";
        if (40.0 < boundaryAI) std::cout << " → Memory Bound\n";
        else std::cout << " → Computing Bound\n";
        
        std::cout << "  卷积 (大kernel): ~20-100 FLOP/Byte";
        if (60.0 < boundaryAI) std::cout << " → Memory Bound\n";
        else std::cout << " → Computing Bound\n";
        
        // 优化建议
        std::cout << "\n优化建议:\n";
        std::cout << "  Memory Bound操作优化策略:\n";
        std::cout << "    - 减少内存访问次数\n";
        std::cout << "    - 使用合并内存访问\n";
        std::cout << "    - 利用共享内存和缓存\n";
        std::cout << "    - 数据重用\n";
        
        std::cout << "  Computing Bound操作优化策略:\n";
        std::cout << "    - 增加并行度\n";
        std::cout << "    - 使用Tensor Cores (如果支持)\n";
        std::cout << "    - 优化算法复杂度\n";
        std::cout << "    - 使用更高效的数学库\n";
        
        // Roofline模型绘制数据
        std::cout << "\nRoofline模型关键点:\n";
        std::cout << "  峰值性能线: " << theoreticalGFLOPS << " GFLOPS (水平线)\n";
        std::cout << "  内存带宽线: Performance = " << memoryBandwidth << " × AI (斜率=" << memoryBandwidth << ")\n";
        std::cout << "  交点 (Ridge Point): (" << boundaryAI << ", " << theoreticalGFLOPS << ")\n";
        
        if (device < deviceCount - 1) {
            std::cout << "\n" << std::string(60, '=') << "\n";
        }
    }
    
    return 0;
}
