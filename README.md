# 🚀 Pipelyne  
### **A Production-Ready Machine Learning Framework in Rust**

![Rust](https://img.shields.io/badge/Rust-1.70+-orange)
![License](https://img.shields.io/badge/License-MIT-blue)
![Status](https://img.shields.io/badge/Build-Passing-brightgreen)

---

## ⭐ Overview

**Pipelyne** is a **full-stack machine learning framework built entirely in Rust**, featuring multiple ML algorithms from scratch, a complete data processing pipeline, a professional CLI, model persistence, and interactive demos.

It is designed for:
- Machine Learning learners  
- Rust enthusiasts  
- Developers who want a clean, extensible ML architecture  
- Portfolio showcase projects  

---

## ✨ Features

### 🔢 **4 Machine Learning Algorithms (From Scratch)**
- **Linear Regression** (w/ R² score)
- **K-Means Clustering**
- **Decision Tree Classifier/Regressor**
- **Perceptron Neural Network**

### 🔄 **Modular Pipeline System**

---

## ⭐ Overview

**Pipelyne** is a **full-stack machine learning framework built entirely in Rust**, featuring multiple ML algorithms from scratch, a complete data processing pipeline, a professional CLI, model persistence, and interactive demos.

It is designed for:
- Machine Learning learners  
- Rust enthusiasts  
- Developers who want a clean, extensible ML architecture  
- Portfolio showcase projects  

---

## ✨ Features

### 🔢 **4 Machine Learning Algorithms (From Scratch)**
- **Linear Regression** (w/ R² score)
- **K-Means Clustering**
- **Decision Tree Classifier/Regressor**
- **Perceptron Neural Network**

### 🔄 **Modular Pipeline System**
```

CSV → Loader → Scaler → Split → Trainer → Predictor → Output

````
- Trait-based design  
- Pluggable pipeline steps  
- Highly extensible  

### 💻 **Production-Ready CLI**
```bash
pipelyne train -d data.csv -m linear_regression -o model.json
pipelyne predict -m model.json -i "6,7,8"
pipelyne demo all_models
````

### 📁 **Data Processing Suite**

* CSV loader (with header support)
* Standard scaling (normalization)
* Train/test splitting
* X/y separation

### 🎓 **Educational Demos**

* 8 fully interactive demos
* Model comparisons
* Pipeline visualizations

---

## ⚡ Quick Start

### 1️⃣ Install

```bash
cargo install pipelyne
```

### 2️⃣ Create a dataset

```bash
echo "x,y
1,3
2,5
3,7" > data.csv
```

### 3️⃣ Train a model

```bash
pipelyne train -d data.csv -m linear_regression -o model.json
```

### 4️⃣ Predict values

```bash
pipelyne predict -m model.json -i "4,5,6"
# Output → [9.0, 11.0, 13.0]
```

---

## 🏗️ Architecture

```
┌───────────┐   ┌────────────┐   ┌──────────────┐   ┌─────────────┐
│ CSV Loader│ → │ Standardizer│ → │ Train/Test   │ → │  Model       │
│           │   │ (Scaling)   │   │ Splitter     │   │ Training     │
└───────────┘   └────────────┘   └──────────────┘   └─────────────┘
```

### Core Components

* **PipelineStep Trait** — Build any pipeline operation
* **Model Trait** — Train, predict, and serialize models
* **CLI Engine** — Subcommands for training, prediction, demos
* **JSON Persistence** — Save/load ML models instantly

---

## 🎮 Demos

Try built-in interactive demos:

```bash
pipelyne demo all_models       # Compare all algorithms
pipelyne demo pipeline         # Watch pipeline in action
pipelyne demo save_load        # Learn model persistence
pipelyne demo decision_tree
pipelyne demo perceptron
pipelyne demo linear_regression
pipelyne demo kmeans
```

---

## 📦 File Formats

| Component    | Format         |
| ------------ | -------------- |
| Input Data   | CSV            |
| Saved Models | JSON           |
| Demo Outputs | Console / JSON |

---

## 📊 Project Stats

```
Algorithms:        4
Pipeline Steps:    5
CLI Commands:      6
Demos:             8
LOC:               ~2000
Dependencies:      5 (csv, serde, clap, rand, ordered-float)
```

---

## 🧱 Folder Structure

```
pipelyne/
│
├── src/
│   ├── algorithms/
│   │   ├── linear_regression.rs
│   │   ├── kmeans.rs
│   │   ├── decision_tree.rs
│   │   └── perceptron.rs
│   │
│   ├── pipeline/
│   │   ├── loader.rs
│   │   ├── scaler.rs
│   │   ├── splitter.rs
│   │   └── traits.rs
│   │
│   ├── cli/
│   │   ├── train.rs
│   │   ├── predict.rs
│   │   ├── demo.rs
│   │   └── main.rs
│
├── models/
│   └── *.json
│
└── README.md
```

---

## 🧠 Why Pipelyne Stands Out

| Feature                 | Typical ML Repo | **Pipelyne**                    |
| ----------------------- | --------------- | ------------------------------- |
| From-scratch algorithms | ❌ No            | ✅ Yes                           |
| CLI tool                | ❌ Rare          | ✅ Full CLI w/ subcommands       |
| Pipeline system         | ❌ None          | ✅ Modular + extensible          |
| JSON persistence        | ❌ Manual        | ✅ Built-in                      |
| Language                | Python          | **Rust (performance + safety)** |
| Educational demos       | ❌ Few           | ✅ 8 demos                       |

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ❤️ Contributing

Pull requests are welcome—submit bug fixes, new algorithms, or better demos!

---

## ⭐ Show Your Support

If you like this project, give it a **star ⭐ on GitHub** to help others discover it!

---

```
```
