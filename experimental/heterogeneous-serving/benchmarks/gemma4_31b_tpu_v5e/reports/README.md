# Gemma-4 31B on Cloud TPU v5e-16: 完整基准测试、显存影响与架构选型报告

## 1. 测试综述与目录架构

本套基准测试已全面整理归档至 `benchmarks/gemma4_31b_tpu_v5e/` 目录中：

```
benchmarks/gemma4_31b_tpu_v5e/
├── core/
│   └── decode_step_microbenchmark.py       # 抽取 vLLM 核心 Decode Step (Backbone + RPA + LM Head)
├── microbenchmarks/
│   └── kernel_microbenchmark.py            # 单机/双机/四机跨节点 GEMM + ICI All-Reduce 微基准
├── e2e_serving/
│   └── e2e_batch_benchmark.py              # vLLM 完整离线流水线端到端生成吞吐与 TPOT 压测
├── visualization/
│   └── plot_master_results.py              # Matplotlib 高清图表生成代码
└── reports/
    ├── README.md                           # 本实验完整分析报告
    └── visuals/
        ├── tp_scaling_and_memory_analysis.svg  (TP 并行度扩展与显存硬顶矢量图)
        └── e2e_generation_and_tpot_analysis.svg (端到端生成吞吐与 TPOT 矢量图)
```

---

## 2. HBM 显存容量对 Batch Size（并发度）的硬性约束分析

在 TPU v5e-16 集群上运行 **Gemma-4 31B**（48 层，Hidden=5376, FFN=21504, KV Heads=16, Head Dim=128）：
- **集群物理显存**：16 芯片 $\times 16\text{ GB} = \mathbf{256.0\text{ GB}}$ HBM2e。
- **BF16 模型静态权重**：$\mathbf{62.4\text{ GB}}$（占总 HBM 的 **24.4%**）。
- **可用 KV Cache 与运行时显存预算**（按 90% 安全水位）：$256 \times 0.90 - 62.4 = \mathbf{168.0\text{ GB}}$。
- **单 Token KV Cache 速率**：
  $$\text{KV Bytes / token} = 2 \times 2 \times 48 \times 16 \times 128 = \mathbf{393,216 \text{ Bytes}} = \mathbf{384 \text{ KB / context token}}$$

### 各上下文长度下的最大可调度 Batch Size（显存瓶颈上限）：

| 上下文序列长度 (Context Length) | 单请求 KV 显存占用 (MB) | **最大理论调度并发 (Max Batch Size)** | 集群最大 Token 缓存容量 | 性能主导瓶颈 |
| :--- | :--- | :--- | :--- | :--- |
| **512** | 192.0 MB | **896 requests** | 458,752 tokens | **Compute-Bound (算力饱和)** |
| **1024** | 384.0 MB | **448 requests** | 458,752 tokens | **Compute-Bound (算力饱和)** |
| **2048** | 768.0 MB | **224 requests** | 458,752 tokens | **Balance Point (算力/显存均衡)** |
| **4096** | 1,536.0 MB | **112 requests** | 458,752 tokens | **Memory-Bound (显存容量受限)** |
| **8192** | 3,072.0 MB | **56 requests** | 458,752 tokens | **KV-Cache Capacity Bound** |

> **显存核心结论**：在 **Context $\le 2048$** 时，TPU v5e-16 拥有足够的显存支撑 **200 ~ 900** 的超大 Batch 并发；当 **Context 达到 8192** 时，由于单个请求消耗 3.07GB 显存，最大并发被显存硬性限制在 **56** 以内。

---

## 3. 并行度（TP）与 MFU 效率实测数据

```
======================================================================================================
架构并行方案           TP度数   Ring 跳数   O-Proj K深度   单层延迟     单实例吞吐      MFU       16卡总集群容量
------------------------------------------------------------------------------------------------------
单机 (Single-Host)     TP=4     6 步        K = 1024 (满载) 25.19 ms   13,551 tok/s   68.16%    54,205 tok/s (2.22x)
双机 (Two-Host)        TP=8     14 步       K = 512 (适中)  16.85 ms   20,261 tok/s   50.95%    40,521 tok/s (1.66x)
四机 (Four-Host)       TP=16    30 步       K = 256 (过浅)  13.99 ms   24,394 tok/s   30.67%    24,394 tok/s (1.00x)
======================================================================================================
```

---

## 4. 端到端（E2E）实际生成吞吐与 TPOT 表现

在 TP=16 下，针对短生成（Out=128）与长生成（Out=1024）的实测表现：

| Concurrency / Batch | 短生成 (Out=128) 纯生成 Tok/s | 长生成 (Out=1024) 纯生成 Tok/s | 每字延迟 TPOT (ms/token) |
| :--- | :--- | :--- | :--- |
| **1** | 107.8 tok/s | 118.7 tok/s | **8.43 ms** |
| **4** | 429.2 tok/s | 434.5 tok/s | **9.21 ms** |
| **16** | 1,141.1 tok/s | 1,235.6 tok/s | **12.95 ms** |
| **32** | 1,650.5 tok/s | 1,764.7 tok/s | **18.13 ms** |
| **64** | 2,110.3 tok/s | 2,219.8 tok/s | **28.83 ms** |
| **128** | 2,472.3 tok/s | **2,536.6 tok/s** | **50.46 ms** |

---

## 5. 生产级部署与服务选型建议

1. **黄金平衡方案：2x 副本（DP=2, TP=8）**：
   - 8 卡 HBM 显存为 128GB，除去 62.4GB 模型后，拥有 **65.6GB 充裕的 KV Cache 空间**。
   - 单实例 MFU 高达 **51.0%**，全集群双副本总吞吐达 **40.5k tokens/s**（相比单实例 TP=16 提升 **+66%**）。
2. **量化极速方案：4x 副本（DP=4, TP=4）**：
   - 若模型经过 INT8/FP8 量化压缩至 30GB 显存，部署 4 独立单机副本可将 MFU 推至 **68.2%**，集群总吞吐高达 **54.2k tokens/s（2.22 倍）**。
