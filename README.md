# 🎓🧠 Student Performance ANN Predictor 🧠🎓

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=2E97F7&center=true&vCenter=true&width=1000&lines=Deep+Learning+Grade+Prediction;UCI+Student+Performance+Dataset;TensorFlow+%2B+Keras+ANN+Architecture;Robust+Sklearn+Preprocessing+Pipeline;Interactive+Streamlit+Web+App)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://student-performance-ann-regreappr-project.streamlit.app/)

### 🚀 **Predict your final academic grade (G3) using an End-to-End Deep Learning System** 📊

### 🧠 Demographic Factors × Study Habits × ANN = **Precision Grade Forecasting** 🎯

---

## 🌟 **WHAT IS THIS?** 🌟

<table>
<tr>
<td>

### 🎯 **The Mission**

This project builds a robust **AI-powered regression system** to predict a student's final grade (**G3**) based on the **UCI Student Performance Dataset** (Math + Portuguese). It analyzes how demographic, social, and study-related factors influence academic success.

**Think of it as:**
- 🧠 **Brain** = TensorFlow Artificial Neural Network (ANN)
- 📊 **Input** = Study time, failures, family support, etc.
- 🔮 **Output** = Final Grade Prediction (0-20 scale)

</td>
<td>

### 🔥 **Key Features**

✅ **End-to-End Pipeline** from raw CSV to deployed web app
✅ **Robust Preprocessing** handles missing data, scaling & encoding
✅ **Advanced ANN Architecture** with Dropouts & Early Stopping
✅ **Feature Selection** strategy (excludes G1/G2 to prevent leakage)
✅ **State Persistence** using `.keras` model & `.joblib` pipeline
✅ **Interactive UI** built with Streamlit for real-time inference
✅ **Mixed Data Support** handles numeric & categorical inputs

**UseCase Applications:**
- 🏫 **Educators** - Identify at-risk students early
- 🎓 **Students** - Understand impact of study habits
- 📊 **EdTech** - Personalized learning recommendations

</td>
</tr>
</table>

---

## 🛠️ **TECH STACK** 🛠️

![Tech Stack](https://skillicons.dev/icons?i=python,tensorflow,sklearn,streamlit,pandas)

| **Category** | **Technologies** |
|--------------|------------------|
| 🐍 **Language** | Python 3.10+ |
| 🧠 **Deep Learning** | TensorFlow, Keras (Sequential API) |
| 📊 **Preprocessing** | Scikit-learn (ColumnTransformer, Pipeline) |
| 🎨 **Frontend** | Streamlit |
| 💾 **Data Handling** | Pandas, NumPy, Joblib |
| 📈 **Data Source** | UCI Machine Learning Repository |

---

## 📂 **PROJECT STRUCTURE** 📂

```
🎓 Student-Performance-ANN/
│
├── 📁 assets/                           # UI Assets (images, banners)
├── 📁 app.py                            # 🚀 Main Streamlit Application
├── 📁 main.ipynb                        # 📓 Training Notebook (EDA + Modeling)
├── 📦 requirements.txt                  # Dependency list
├── 🧠 student_grade_ann_best.keras      # Best trained ANN model
├── 🔧 preprocessor.joblib               # Saved Scikit-learn transformation pipeline
├── 📋 feature_columns.json              # Schema of input features
├── 📊 student-mat.csv                   # Mathematics dataset
├── 📊 student-por.csv                   # Portuguese language dataset
└── 📖 README.md                         # Project documentation
```

---

## 🚀 **QUICK START** 🚀

### **Step 1: Clone the Repository** 📥

```bash
git clone https://github.com/your-username/student-performance-ann.git
cd student-performance-ann
```

### **Step 2: Install Dependencies** 📦

```bash
pip install -r requirements.txt
```

### **Step 3: Run the App** 🎯

```bash
streamlit run app.py
```

### **Step 4: Open in Browser** 🌐

The app will automatically open at: **`http://localhost:8501`**

---

## 🧪 **HOW IT WORKS** 🧪

```mermaid
graph LR
    A[UCI Datasets] --> B[Data Merging]
    B --> C[Preprocessing Pipeline]
    C --> D[ANN Training]
    D --> E[Model & Artifacts Saving]
    E --> F[Streamlit Web App]
    F --> G[User Prediction]
```

### **The AI Pipeline:**

1️⃣ **Data Ingestion** → Merges Math (`student-mat.csv`) and Portuguese (`student-por.csv`) datasets.
2️⃣ **Preprocessing Engine** (`ColumnTransformer`):
   - **Numeric**: Median Imputation → Standard Scaling
   - **Categorical**: Most Frequent Imputation → One-Hot Encoding
3️⃣ **Neural Network Architecture**:
   - **Input Layer**: Matches processed feature dimensions
   - **Hidden Layers**: Dense layers with ReLU activation
   - **Regularization**: Dropout layers (0.25-0.30) to prevent overfitting
   - **Output Layer**: Single Linear neuron for regression
4️⃣ **Training**: Adam Optimizer, MSE Loss, EarlyStopping callbacks.
5️⃣ **Deployment**: Loads saved `.keras` model and `.joblib` pipeline to serve predictions.

---

## 📊 **DATASET & FEATURES** 📊

The model treats **30+ input features** to determine academic success:

| **Feature Category** | **Examples** |
|----------------------|-------------|
| 🏠 **Demographics** | `age`, `sex`, `address` (urban/rural), `famsize` |
| 📚 **School Info** | `school` (GP/MS), `reason` for choosing school |
| 📖 **Study Habits** | `studytime`, `failures`, `absences`, `schoolsup` |
| 👪 **Family Context** | `Medu` (Mother's edu), `Fjob`, `famsup`, `famrel` |
| 🍻 **Lifestyle** | `freetime`, `goout`, `Dalc` (weekday alcohol), `health` |

> **Note:** Interim grades `G1` and `G2` are intentionally excluded in some training configurations to create a purely predictive model based on student characteristics rather than past performance.

---

## 👨‍💻 **CONNECT WITH ME** 👨‍💻

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github)](https://github.com/mayank-goyal09)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mayank-goyal-mg09/)

**Mayank Goyal**
📊 Data Scientist | 🤖 Deep Learning Enthusiast | 🐍 Python Developer

---

## ⭐ **SHOW YOUR SUPPORT** ⭐

Give a ⭐️ if this project inspired your next AI application!

### 🎓 **Built with TensorFlow & ❤️** 🎓
​
