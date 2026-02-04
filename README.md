---
title: GSAS-II Impurity Detector
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# 🔬 GSAS-II Impurity Detector

This application leverages **GSAS-II** for automated crystallographic impurity phase discovery in powder diffraction data.

### 🌟 Key Features
- **16GB RAM Environment**: Hosted on Hugging Face Spaces for stability.
- **Hierarchical Progress Tracking**: Real-time pass-by-pass status.
- **Interactive Results**: Dynamic plots and CSV data visualization.
- **ML-Guided Refinement**: Automated candidate screening and selection.

### 🚀 Running Locally
If you want to run this locally:
1. Install [Pixi](https://pixi.sh).
2. Clone the repository.
3. Run `pixi run streamlit run app.py`.

### 📦 Deployment
Deploying to Hugging Face Spaces:
1. Create a new Space.
2. Select **Docker** as the SDK.
3. Choose **Blank** or **Streamlit** (doesn't matter as we use the Dockerfile).
4. Connect this GitHub repository.
