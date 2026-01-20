# High-Performance Pneumonia Detection & Localization (HPC Project)

## 🚀 Project Overview
This project applies **Deep Learning** and **High-Performance Computing (HPC)** to detect and localize pneumonia in chest X-ray (CXR) images. It leverages a hybrid architecture using **CNNs** for classification and **YOLOv8** for bounding-box localization.

The system was engineered to run on the **Northeastern Discovery HPC Cluster**, utilizing **PyTorch Distributed Data Parallel (DDP)** and **Fully Sharded Data Parallel (FSDP)** to scale training across multi-GPU nodes (NVIDIA A100/P100).

## 📊 Key Results
* **Accuracy:** Achieved **87.2%** classification accuracy on the RSNA Pneumonia Detection Challenge dataset.
* **Throughput:** Optimized training pipeline using parallel data loading and mixed-precision (AMP) to maximize GPU utilization.
* **Scalability:** Benchmarked distributed training strategies (DDP vs. FSDP) to analyze communication overhead vs. memory efficiency.

## 🛠️ Methodologies & Tech Stack

### 1. Data Pipeline
* **Dataset:** RSNA Pneumonia Detection Challenge (30,000+ DICOM images).
* **Preprocessing:** Custom OpenCV pipeline for contrast enhancement (CLAHE) and resizing (640x640).
* **Parallel I/O:** Utilized `Dask` and `NumWorkers` to prevent CPU-bottlenecks during GPU training.

### 2. Model Architecture
* **Localization:** **YOLOv8**, fine-tuned with custom anchor boxes for thoracic opacities.
* **Classification:** Custom CNN backbone with dropout regularization to prevent overfitting on clinical data.

### 3. Distributed Engineering (HPC)
* **DDP (Distributed Data Parallel):** Implemented gradient synchronization via the `NCCL` backend for multi-GPU training.
* **FSDP (Fully Sharded Data Parallel):** Sharded model parameters across GPUs to reduce peak memory consumption, allowing for larger batch sizes.
* **Mixed Precision:** Integrated `torch.cuda.amp` to reduce VRAM usage by ~40%.

## 📉 Performance Analysis
We conducted extensive profiling to compare Serial (CPU) vs. Parallel (GPU) training execution:
* **Observation:** Distributed training significantly reduced epoch time compared to serial baselines.
* **Trade-offs:** Identified synchronization overhead (Amdahl's Law) when scaling beyond 4 GPUs for smaller model architectures (YOLOv8 Nano).
* **Optimization:** FSDP proved superior for memory management, while DDP offered the highest raw throughput for this specific model size.

## 💻 Environment & Reproducibility
* **Cluster:** Open OnDemand (OOD) HPC Environment.
* **Job Scheduler:** Slurm Workload Manager.
* **Frameworks:** PyTorch 2.0, Ultralytics, CUDA 12.x, Dask.

---
*Note: This repository contains the source code tailored for HPC submission. Jupyter notebooks in the `notebooks/` directory demonstrate the inference logic and distributed training setup.*
