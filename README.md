# 🚗 Car Object Detection with YOLOv8

A Deep Learning project to detect and localize **cars** in street and traffic images using the **YOLOv8** model. Built using Python, OpenCV, Ultralytics YOLO, and deployed via a **Streamlit** web app.

## 📌 Project Overview
This project focuses on creating an object detection system for identifying cars in images from traffic environments using the YOLOv8 model. The model was trained on custom-annotated data and deployed with an interactive UI.

---

## 🚀 Tech Stack
- **Python**
- **YOLOv8 (Ultralytics)**
- **OpenCV**
- **Streamlit**
- **Pandas / NumPy**
- **Matplotlib / PIL**
- **MLflow** (optional for experiment tracking)

---

## 🎯 Objectives
- Train a YOLOv8 model to detect vehicles in traffic scenes
- Evaluate the model using mAP@0.5, Precision, Recall, and FPS
- Deploy the detection interface using Streamlit
- Optionally track experiments with MLflow

---

## 📁 Dataset
- Annotated CSV file containing bounding boxes: `image, xmin, ymin, xmax, ymax`
- Converted to YOLO format: `class_id x_center y_center width height`
- Data folder structure:
