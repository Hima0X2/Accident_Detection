# 🚗 Road Accident Detection using YOLOv8

This project is part of my **Final Year Thesis**, aimed at developing a real-time, vision-based accident detection system using **YOLOv8**. By leveraging deep learning and computer vision, the system detects potential road accidents in video feeds with high accuracy, helping to enhance road safety through early detection and response.


## 📌 Project Overview

- **Objective**: Detect road accidents from video or real-time feeds using YOLOv8 object detection.
- **Framework**: PyTorch-based YOLOv8 model trained and fine-tuned for accident scenarios.
- **Input**: Images/Video files.
- **Output**: Annotated video frames with bounding boxes and labels for accident detection.


## 🛠️ Key Technologies Used

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn (for visualization)
- Roboflow (for data annotation)
- Google Colab


## 🧪 Dataset

- Dataset collected from public accident footage sources and manually annotated using **Roboflow** / **LabelImg**.
- Split into train, validation, and test sets.
- Classes: `accident`, `vehicle`, etc.


## 📊 Evaluation Metrics

- **mAP (Mean Average Precision)**
- **Precision**
- **Recall**
- **F1 Score**

## 📸 Sample Output

#### User Interface
<img width="581" height="389" alt="image" src="https://github.com/user-attachments/assets/97d1827f-63b6-4c00-91d1-300a027ae27c" />

#### Analysis video result 

<img width="748" height="373" alt="image" src="https://github.com/user-attachments/assets/2448bd12-9149-4dea-80d4-3709a50771ae" />

## 📚 Final Year Thesis

This project was completed as part of my **Bachelor’s Final Year Thesis** under the title:  
**"YOLOv8 for Road Accident Detection: A Vision-Based Deep Learning Model"**

## 📌 Future Improvements

- Integrate alert system (email/SMS)
- Integrate with live cameras
- Deploy as a mobile app
- Improve accuracy with larger annotated datasets

## 🧑‍💻 Author

**Sanjida Akter Samanta**   
Noakhali Science and Technology University

