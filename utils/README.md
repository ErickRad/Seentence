# SEENTENCE

*Transforming Images into Captions with AI Power*

![Last Commit](https://img.shields.io/badge/last%20commit-yesterday-blue)
![Python](https://img.shields.io/badge/python-100%25-blue)
![Languages](https://img.shields.io/badge/languages-1-blue)

---

### Built with the tools and technologies:

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/-CUDA-76B900?logo=nvidia&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C)
![Pandas](https://img.shields.io/badge/-Pandas-150458)
![NumPy](https://img.shields.io/badge/-NumPy-013243)
![tqdm](https://img.shields.io/badge/-tqdm-yellow)

![Pillow](https://img.shields.io/badge/-Pillow-blue)

---

## 📚 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)

---

## 🧠 Overview

**Seentence** is a powerful developer toolkit designed for building sophisticated image captioning systems. It streamlines data management, model training, and caption generation within a cohesive, modular framework.

### Why Seentence?

This project aims to simplify the development of scalable, high-quality vision-language models. The core features include:

- 🦾 **Powerful Architecture**: Combines Transformers to Vision, Encoder and Decoder for best results
results.
- 🧵 **Efficient Data Handling**: Loads and caches large datasets like MSCOCO, ensuring fast access and consistency.
- 🎯 **End-to-End Workflow**: Supports training, inference, and model management with configurable hyperparameters.
- 📊 **Progress Visualization**: Tracks training metrics with integrated plotting tools for performance monitoring.
- 🛠️ **Custom Data Pipelines**: Facilitates preprocessing, tokenization, and batching tailored for image captioning tasks, allowing you to create your own vocabulary.

---

## 🚀 Getting Started

### 📦 Pre-requisites

- Python 3.8+
- PyTorch
- torchvision
- CUDA Toolkit (optional for GPU acceleration)
- numpy
- pandas
- tqdm
- matplotlib
- nltk
- Pillow

### 🔧 Installation

```bash
git clone https://github.com/ErickRad/Seentence.git

Optional: 
python -m venv Seentence

cd Seentence
pip install -r requirements.txt

---

## 🧪 Usage 

### Create captions for images

- Paste your image in the Seetence directory

```bash
python captionize.py --image-path <your_image.jpg> --num-caption 5 (Default: 1) 


