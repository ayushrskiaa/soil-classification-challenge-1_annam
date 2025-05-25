# Soil Classification Challenge

This repository contains the code and resources for the Soil Classification Challenge, as part of the Annam.ai IIT Ropar hackathon.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Train](#how-to-train)
- [How to Infer](#how-to-infer)
- [Preprocessing & Postprocessing](#preprocessing--postprocessing)
- [Team](#team)
- [License](#license)

---

## Overview

The goal of this project is to classify soil images into one of four categories: Alluvial soil, Black Soil, Clay soil, and Red soil. The solution uses a ResNet18-based deep learning model implemented in PyTorch.

---

## Project Structure

```
challenge-1/
│
├── requirements.txt
├── data/
│   ├── download.sh
│   └── ... (dataset files)
├── docs/
│   └── cards/
│       ├── architecture.png
│       └── ml-metrics.json
├── notebooks/
│   ├── inference.ipynb
│   └── training.ipynb
└── src/
    ├── config.py
    ├── dataset.py
    ├── postprocessing.py
    ├── predict.py
    ├── preprocessing.py
    └── train.py
```

---

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd soil-classification-challenge-template/challenge-1
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   ```sh
   cd data
   bash download.sh
   cd ..
   ```

---

## How to Train

To train the model, run:

```sh
python src/train.py
```

This will train a ResNet18 model and save the weights to `model.pth`.

---

## How to Infer

To generate predictions on the test set, run:

```sh
python src/predict.py
```

This will create a `submission.csv` file with the predicted soil types for the test images.

---

## Preprocessing & Postprocessing

- **Preprocessing:**  
  See [`src/preprocessing.py`](src/preprocessing.py) for any data preprocessing steps.

- **Postprocessing:**  
  See [`src/postprocessing.py`](src/postprocessing.py) for postprocessing steps after inference.

---

## Team

- **Team Name:** KrishiSetu
- **Members:** Dnyandeep Chute, Ayush Kumar, Suyash Mishra, Krish Kalgude, Yash Verma

---

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

---
