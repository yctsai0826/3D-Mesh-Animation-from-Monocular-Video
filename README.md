# 3D Mesh Animation from Monocular Video via Explicit Kinematic Retargeting


## 🚀 Pipeline Overview


1. 
**Tracking & Initialization:** Extracts long-term 2D point tracks from video and generates a 3D rest skeleton from the input mesh.


2. 
**Semantic Correspondence:** Employs DINOv2 zero-shot features and the Hungarian algorithm to map 3D joints to 2D video tracks without manual labels.


3. 
**3D Motion Lifting:** A Spatio-Temporal Graph Neural Network (ST-GNN) resolves depth ambiguity. It processes a tensor of shape $F \times K \times 3$ to lift 2D tracks into full 3D kinematics.


4. 
**Voxel-Based Skinning:** Converts the mesh into a watertight volumetric proxy to compute robust weights, ensuring artifact-free deformations.



## Installation

### Environment Setup

```bash
conda create -n MeshDeform python=3.10
conda activate MeshDeform
pip install -r requirements.txt

```

### Quick Start (Inference)

Place your input video and your static mesh in `data/{name}/`, then run step 1~3

The final rigged asset will be saved in `outputs/`.

