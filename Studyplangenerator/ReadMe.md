# Study Plan Generator

## 1. Project Overview
The **Study Plan Generator** is a machine learning–based tool that helps students generate personalized study strategies by analyzing prerequisite relationships between topics and predicting study effort required.  
It combines **graph-based dependency trees** with **machine learning models** to suggest effective learning paths and estimate study hours.  

Additionally, a **Student Lifestyle Efficiency Predictor** is included, which evaluates whether a student’s daily routine (study, sleep, social activities, etc.) leads to efficient academic outcomes.  

---

## 2. Dataset

### a) Prerequisite Graph Data
- **Dataset Name:** *Educational Prerequisite Annotations*  
- **Source:** [MIT Prerequisite Corpus (Educational Data Mining research)](https://people.cs.umass.edu/~miyyer/pubs/2016_naacl_prereq.pdf)  
- **Files Used:**  
  - [`prerequisite_annotations.csv`](https://raw.githubusercontent.com/rajarshd/Github-Data/master/prerequisite_annotations.csv) → Contains prerequisite relationships between academic topics.  
  - [`topics_to_resources.csv`](https://raw.githubusercontent.com/rajarshd/Github-Data/master/topics_to_resources.csv) → Maps topic IDs to topic names.  
This data is used to build **directed graphs** (`networkx`) representing prerequisite → dependent relationships.  

### b) Student Lifestyle Data
- **[`student_lifestyle.csv`](https://raw.githubusercontent.com/ansh941/Machine-Learning-Datasets/master/student_lifestyle.csv)** → Contains details of students’ daily habits and GPA.  

---

## 3. Methodology

### A. Study Plan Tree Generation
1. Build a **directed graph** of topics (`networkx`).  
2. Implement two recursive tree functions:  
   - **Prerequisite Tree** → Topics required before studying a given target topic.  
   - **Dependency Tree** → Topics dependent on the given target topic.  
3. Display tree hierarchy using **Streamlit**.  

### B. Target Variable
- For lifestyle prediction: **Efficiency (0/1)**

---

## 6. Application (Streamlit UI)

### Training Mode
- Add new topics and retrain **SVM model**.  

### Prediction Mode
- Select a topic → See **prerequisites & dependencies (DETAILED STUDY PLAN)**.  
- Predict study hours (**SVR – Support Vector Regressor**).  
- Predict student efficiency (**SVC – Support Vector Classifier**) from lifestyle input.  
