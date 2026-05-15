# LLM Deploy Prediction

## Objective
Give an analysis on how many chips and sharding strategies serve the best performance under latency constraints for Qwen3 Coder like models and P/D disaggregation on ironwood.

## Folder Structure
- `src/`: Contains the source code for the project.
  - `roofline/`: Contains scripts for calculating theoretical roofline based on model config and hardware spec.
  - `microbenchmarks/`: Contains microbenchmarks for GMM, RPA kernels, and ragged_all_to_all communication.
  - `benchmarks/`: Contains end-to-end model benchmarks to verify the trend matches the prediction.




