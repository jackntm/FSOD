# Prompt Driven Few Shot Object Detection (FSOD)

This study presents a Few-Shot Object Detection (FSOD) approach designed to address the challenge of extremely limited reference data[cite: 16]. The system operates under a strict protocol where each target class is defined by a support set of only $N \le 3$ reference images.

## 👥 Team Members
Developed by a team from the Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT):
* **Nguyễn Tiến Minh** - 2352755 
* **Tán Khánh Phong** - 2352911 
* **Nguyễn Hồ Quang Khải** - 2352538 
* **Hồ Hồng Phúc Nguyên** - 2352824 
* **Phạm Trần Gia Phú** - 2352921 

## 🚀 System Architecture
The proposed methodology follows a three-stage processing architecture that synergizes Transfer Learning and Metric Learning.



### 1. Pre-processing & Scene Segmentation
* **Algorithm**: Uses a Mean Absolute Difference (MAD) mechanism to detect visual changes between consecutive grayscale frames.
* **Optimization**: Effectively eliminates temporal redundancy by extracting only high-variance Key-Frames.
* **Parameters**: Configured with a `DIFF_THRESHOLD` of 5.0 and a `MIN_SCENE_FRAMES` of 50 to ensure stability.

### 2. High-Level Recognition
* **Proposal Generation**: Leverages **SAM 2** (`sam2.1_hiera_large`) for zero-shot masking to isolate potential entities within a frame.
* **Feature Extraction**: Utilizes **DINOv2** (`dinov2-base`) as a high-generalization backbone to encode objects into feature vectors.
* **Matching Mechanism**: Calculates the **Cosine Similarity** between candidate crops and prototype vectors from reference images.
* **Noise Reduction**: Reference images undergo automated background removal via `rembg` to improve matching accuracy.

### 3. Temporal Stitching
* **Trajectory Construction**: Discrete frame-level detections are consolidated into continuous tracks.
* **Algorithm**: Merges adjacent temporal segments ($Start_{next} = End_{curr} + 1$) to simulate stable object tracking.
* **Output**: Serializes results into a standardized JSON format containing `video_id` and consolidated annotations.

## 🎮 Interactive Demo
We have prepared a ready-to-run Google Colab notebook. This environment is pre-configured with all necessary dependencies, including **SAM 2**, **DINOv2**, and **Rembg**.

* **Google Colab Link**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GGQo42qq8UUeVVOKuwvPKYsRBhvMApwo?usp=sharing)

> [!IMPORTANT]
>The Colab notebook requires a **GPU runtime** (T4 or better) to handle the Foundation Models efficiently.

## 🛠 Installation

### 1. Prerequisites
* **OS**: Windows/Linux/macOS
* **Python**: 3.10+
* **Hardware**: NVIDIA GPU with CUDA support is highly recommended for SAM 2 and DINOv2 inference.

### 2. Setup
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/jackntm/FSOD.git](https://github.com/jackntm/FSOD.git)
cd FSOD
pip install -r requirements.txt

