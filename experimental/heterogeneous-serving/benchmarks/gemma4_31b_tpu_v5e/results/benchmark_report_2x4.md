# GEMMA-4-31B Serving Benchmark Report (Cloud TPU v5e, Topology 2x4)
**Model**: `gemma-4-31b` | **Topology**: `2x4` | **Timestamp**: 2026-08-25 21:20:50 UTC

## 1. Executive Performance Summary

| Metric | Value | Unit | Description |
|---|---|---|---|
| **Serving Throughput** | **214,616.1** | tok/s | Aggregate generation rate across batch |
| **Prefill TTFT (Mean)** | **21.92** | ms | Time To First Token |
| **Prefill TTFT (p90)** | **29.00** | ms | 90th percentile initial prompt latency |
| **Decode TPOT (Mean)** | **4.18** | ms | Time Per Output Token |
| **Decode TPOT (p90)** | **5.10** | ms | 90th percentile decode step latency |
| **Compute Throughput** | **8,506.6** | TFLOPS | Effective forward compute rate |
| **Model FLOPs MFU** | **539.8%** | % | Hardware utilization efficiency |
| **Memory Bandwidth** | **10,159.0** | GB/s | Effective HBM read bandwidth |

## 2. Per-Configuration Breakdown

| Batch Size | Prompt Len | Output Len | Throughput (tok/s) | TTFT (ms) | TPOT (ms) |
|---|---|---|---|---|---|
| 1 | 512 | 1 | 25,941.8 | 16.25 | 3.52 |
| 1 | 1024 | 1 | 42,663.9 | 20.50 | 3.52 |
| 1 | 2048 | 1 | 62,997.7 | 29.00 | 3.52 |
| 16 | 512 | 1 | 407,344.9 | 16.25 | 3.90 |
| 16 | 1024 | 1 | 672,131.1 | 20.50 | 3.90 |
| 16 | 2048 | 1 | 996,474.2 | 29.00 | 3.90 |
| 64 | 512 | 1 | 1,537,798.6 | 16.25 | 5.10 |
| 64 | 1024 | 1 | 2,562,500.0 | 20.50 | 5.10 |
| 64 | 2048 | 1 | 3,845,630.5 | 29.00 | 5.10 |
