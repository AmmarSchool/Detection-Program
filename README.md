﻿# Detection Program (Capstone Project - NG-06)

This repository contains the **Detection Program** developed as part of the NG-06 Capstone Project at Toronto Metropolitan University. The goal of the project is to detect **anomalies in Augmented Reality (AR) environments** using AI-powered feature reconstruction and heatmap visualization, offering real-time feedback via a modern UI.

---

## 🧠 Project Overview

This system utilizes a **convolutional autoencoder** trained to reconstruct normal visual features. When anomalies (objects or conditions not seen during training) appear in a live camera feed, the reconstruction error is visualized as a **heatmap overlay**, allowing users to immediately identify abnormal areas.

This technique is particularly effective for identifying defective objects or unexpected items in controlled environments such as smart factories, healthcare facilities, or AR simulations.

---

## 📷 Key Features

- **Live Camera Feed Selection**  
  Choose any connected webcam via dropdown and begin detection in real-time.

- **Enable/Disable AI Detection**  
  Toggle anomaly detection on or off with a single button.

- **Heatmap Overlay**  
  Visual representation of anomalies based on reconstruction error—high-error areas are highlighted using a color-coded heatmap.

- **Capture Image**  
  Take and save a snapshot of the current video frame.

- **CustomTkinter UI**  
  A sleek, dark-themed modern interface for user interaction.

---

## 🗂 Folder Structure

```
Detection-Program/
│
├── models/
│   └── decoder.h5               # Pre-trained autoencoder model (tracked via Git LFS)
│
├── DetectionProgram.py          # Main Python script with UI and detection logic
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files/folders
├── .gitattributes               # Git LFS tracking config
└── README.md                    # This file
```

---

## ⚙️ Technologies Used

- **Python 3.10**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **CustomTkinter**
- **Pillow**
- **Git LFS**

---

## 🚀 How to Run

1. **Clone the repository**

```bash
git clone https://github.com/AmmarSchool/Detection-Program.git
cd Detection-Program
```

2. **Create and activate a virtual environment**

```python -m venv venv310
source venv310/Scripts/activate     # On Windows
```

3. **Install dependencies**

```bash 
pip install -r requirements.txt
```

4. Run the detection program

```bash
python DetectionProgram.py
```

💡 Notes

    The file models/decoder.h5 is tracked via Git LFS. Make sure you have Git LFS installed before cloning the repository:

    git lfs install

    If anomalies are not being detected or heatmaps look incorrect, ensure the camera is working and positioned in a stable lighting environment. Also verify that the .h5 model is trained properly and loaded successfully.

🧪 Use Case & Application

This detection tool serves as a foundational module in our larger AR anomaly detection pipeline. Future integration plans include:

    Overlaying results in AR headsets.

    Real-time communication between rovers and ground control (Lab 3+).

    Dataset collection via Unity3D for extended training and testing.

🧑‍💻 Team NG-06 Members

    Syed Ammar Ali — Lead Developer for UI and Model Integration
        Led the development of the detection program’s user interface and implemented the integration of the pretrained model with the live webcam pipeline. Also contributed to overall system design, testing, and GitHub deployment.

    Usba Gohir — Lead on Autoencoder Design & Pretraining
        Designed and developed the autoencoder architecture using Google Colab. Trained the anomaly detection model and produced the pretrained .h5 files used in the Detection Program. Also supported research efforts on AR pipeline strategies and machine learning methodology.

    Seyed Hamid Javaheri — Lead on Unity Dataset Generation
        Spearheaded the creation of Unity-based 3D environments to simulate real-world scenes for anomaly injection. Generated high-quality image datasets used for model training. Provided insights on Unity-to-Python pipeline compatibility.

    Abdul Aziz Ibrahim — UI Refinement and Research Assistant
        Assisted in refining the Detection Program’s UI for better user experience and visual consistency. Supported both the Unity-based dataset development and the research phase of autoencoder architecture, contributing to the technical design and evaluation strategies.

📬 Contact

For inquiries or collaboration:
📧 syedammar.work@gmail.com
📧 usba.gohir@torontomu.ca
📧 seyedhamidjavaheri@gmail.com
📧 a4ibrahim@torontomu.ca

🏫 Toronto Metropolitan University — Engineering Design Project 2025
📜 License

This project is for educational purposes under the TMU EDP guidelines. No commercial use is permitted without written consent from the NG06 Team Members.
