# Prompt Driven Few Shot Object Detection (FSOD)

[cite_start]This study presents a Few-Shot Object Detection (FSOD) approach designed to address the challenge of extremely limited reference data[cite: 16]. [cite_start]The system operates under a strict protocol where each target class is defined by a support set of only $N \le 3$ reference images[cite: 16, 160].

## 👥 Team Members
[cite_start]Developed by a team from the Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT)[cite: 1, 14]:
* [cite_start]**Nguyễn Tiến Minh** - 2352755 [cite: 4]
* [cite_start]**Tán Khánh Phong** - 2352911 [cite: 4]
* [cite_start]**Nguyễn Hồ Quang Khải** - 2352538 [cite: 4]
* [cite_start]**Hồ Hồng Phúc Nguyên** - 2352824 [cite: 4]
* [cite_start]**Phạm Trần Gia Phú** - 2352921 [cite: 4]

## 🚀 System Architecture
[cite_start]The proposed methodology follows a three-stage processing architecture that synergizes Transfer Learning and Metric Learning[cite: 16, 65].



### 1. Pre-processing & Scene Segmentation
* [cite_start]**Algorithm**: Uses a Mean Absolute Difference (MAD) mechanism to detect visual changes between consecutive grayscale frames[cite: 18, 70, 73].
* [cite_start]**Optimization**: Effectively eliminates temporal redundancy by extracting only high-variance Key-Frames[cite: 18, 143, 145].
* [cite_start]**Parameters**: Configured with a `DIFF_THRESHOLD` of 5.0 and a `MIN_SCENE_FRAMES` of 50 to ensure stability[cite: 166, 178, 179].

### 2. High-Level Recognition
* [cite_start]**Proposal Generation**: Leverages **SAM 2** (`sam2.1_hiera_large`) for zero-shot masking to isolate potential entities within a frame[cite: 19, 85, 164].
* [cite_start]**Feature Extraction**: Utilizes **DINOv2** (`dinov2-base`) as a high-generalization backbone to encode objects into feature vectors[cite: 19, 90, 165].
* [cite_start]**Matching Mechanism**: Calculates the **Cosine Similarity** between candidate crops and prototype vectors from reference images[cite: 21, 95, 96].
* [cite_start]**Noise Reduction**: Reference images undergo automated background removal via `rembg` to improve matching accuracy[cite: 20, 92, 134, 202].

### 3. Temporal Stitching
* [cite_start]**Trajectory Construction**: Discrete frame-level detections are consolidated into continuous tracks[cite: 22, 100, 103].
* [cite_start]**Algorithm**: Merges adjacent temporal segments ($Start_{next} = End_{curr} + 1$) to simulate stable object tracking[cite: 107, 109, 223].
* [cite_start]**Output**: Serializes results into a standardized JSON format containing `video_id` and consolidated annotations[cite: 112, 224].

## 🛠 Installation

### 1. Prerequisites
* **OS**: Windows/Linux/macOS
* **Python**: 3.10+
* [cite_start]**Hardware**: NVIDIA GPU with CUDA support is highly recommended for SAM 2 and DINOv2 inference[cite: 169].

### 2. Setup
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/jackntm/FSOD.git](https://github.com/jackntm/FSOD.git)
cd FSOD
pip install -r requirements.txt