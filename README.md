# Mental_Health-App (Hindi Input)

This project is a Hindi language-based sentiment analysis tool that classifies user input (sentence or paragraph) into **Positive**, **Negative**, or **Neutral** sentiment. If the sentiment is found to be negative, the system also provides helpful **stress-relief suggestions**.

---

## 🧠 Features

- **Supports Hindi text input**
- **Three sentiment categories**: Positive, Negative, Neutral
- **TF-IDF + Naive Bayes model**
- **Simple web interface using Gradio**
- **Stress-relief recommendations if sentiment is negative**

---

## 📸 Screenshot

![image](https://github.com/user-attachments/assets/919df53a-725e-41b7-b3f6-db60d38c2a1b)


---

## 🛠️ Tech Stack

- Python
- Pandas, scikit-learn
- Regex for Hindi text cleaning
- Gradio for web-based UI

---

## 📂 File Structure
├── sen_1k.csv # Dataset (Hindi text + labels) ├── app.py # Main application code ├── README.md # This file └── d794115c-0876...png # Screenshot image


---

## 🚀 How to Run
```bash
1.Install dependencies:

pip install pandas scikit-learn gradio


2.Make sure all files are in the same directory:

app.py

sen_1k.csv

README.md

3.Run the application:
app.py

