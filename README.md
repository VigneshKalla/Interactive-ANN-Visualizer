# Interactive ANN Visualizer with Agentic AI Support

## Abstract

This project presents an **interactive educational system for visualizing Artificial Neural Network (ANN) training dynamics**. Built using **PyTorch** and **Streamlit**, the application enables users to configure neural network architectures, train models on both predefined and custom datasets, and observe decision boundary evolution across training epochs. An optional **Agentic AI assistant powered by Google Gemini** provides contextual explanations of model behavior, architectural choices, and loss trends. The system is designed to support intuitive understanding of neural networks through visual and experimental learning.

---

## Keywords

Artificial Neural Networks, Deep Learning Education, PyTorch, Streamlit, Model Visualization, Agentic AI, Gemini, Interpretability

---

## 1. Introduction

Understanding how neural networks learn remains challenging for many beginners due to their abstract mathematical nature. Traditional learning approaches often emphasize theory or final metrics without exposing intermediate learning behavior.

This project addresses this gap by providing:

* Visual representations of ANN learning
* Epoch-wise decision boundary evolution
* Loss-based performance analysis
* AI-assisted explanations grounded in model context

The application serves as an **educational and exploratory tool** rather than a production training system.

---

## 2. System Architecture

### 2.1 Technologies Used

* **PyTorch** – Neural network modeling and training
* **Streamlit** – Interactive web interface
* **scikit-learn** – Dataset generation
* **Matplotlib** – Visualization
* **Google Gemini (optional)** – Agentic AI explanations

---

### 2.2 User Interface Design

The interface is divided into two major components:

1. **Sidebar (Control Panel)**
   Dataset selection, model configuration, training parameters, and API key input.

2. **Main Canvas (Visualization Area)**
   Displays decision boundary evolution, loss curves, and analysis results.

This separation ensures clarity and minimizes cognitive overload.

---

## 3. Methodology

### 3.1 Dataset Configuration

The system supports two dataset workflows:

#### Predefined Datasets

* Moons
* Circles
* Classification
* XOR

#### Custom Dataset Creation

* Number of samples
* Number of classes
* Noise level
* Class imbalance (weights)

All datasets are generated in 2D space to support intuitive visualization.

---

### 3.2 ANN Architecture

The neural network is dynamically constructed based on user input:

* Variable number of hidden layers
* Configurable neurons per layer
* Activation functions:

  * ReLU
  * Sigmoid
  * Tanh
  * Linear

The model supports both **classification** and **regression** tasks.

---

### 3.3 Training Process

* Gradient-based optimization using backpropagation
* User-defined learning rate and epochs
* Training and test loss tracked per epoch

Snapshots of model predictions are captured at:

* Initial state
* 25% training progress
* 75% training progress
* Final epoch

---

## 4. Visualization and Analysis

### 4.1 Decision Boundary Evolution

A 2×2 grid displays how decision boundaries change as training progresses, providing insight into:

* Non-linear feature learning
* Model capacity
* Convergence behavior

### 4.2 Loss Curves

* Training loss
* Test loss

These metrics enable identification of:

* Underfitting
* Overfitting
* Convergence stability

---

## 5. Agentic AI Assistance (Optional)

An integrated AI assistant powered by **Google Gemini** provides contextual explanations based on:

* Selected dataset
* Model architecture
* Training parameters
* Observed loss trends

Example queries include:

* Why did test loss increase?
* How does depth affect learning?
* What role does regularization play?

The chatbot has access to the **entire project context**, enabling informed responses.

---

## 6. Installation and Execution

### 6.1 Clone Repository

```bash
git clone [https://github.com/<your-username>/<your-repository-name>.git](https://github.com/VigneshKalla/Interactive-ANN-Visualizer.git)
cd <your-repository-name>
```

---

### 6.2 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 6.3 Run Application

```bash
streamlit run app.py
```

The application runs locally at `http://localhost:8501`.

---

## 7. Educational Outcomes

This project demonstrates:

* Why linear models fail on non-linear problems (e.g., XOR)
* How network depth improves representational power
* Effects of learning rate and regularization
* Relationship between architecture complexity and generalization

It promotes **learning through experimentation and visualization**.

---

## 8. Limitations

* Designed for 2D datasets only
* Not optimized for large-scale training
* Intended for educational use, not production deployment

---

## 9. Future Work

Potential extensions include:

* Additional datasets (spirals, blobs)
* Dropout and batch normalization
* Early stopping
* Model export and reuse
* Multi-dimensional feature visualization

---

## 10. Project Structure

```
project/
├── app.py
├── model.py
├── trainer.py
├── datasets.py
├── visualizer.py
├── chatbot.py
├── requirements.txt
└── README.md
```

---

## 11. Acknowledgements

The author sincerely acknowledges **Saxon K. Sha Sir** for continuous technical guidance, **Lakshmi Vangapandu Mam** for clear and encouraging mentorship, and **Raghu Ram Aduri Sir** for valuable managerial support.

Gratitude is extended to **Innomatics Research Labs**, and to **Vishwanath Nyathani Sir (Founder)** and **Kalpana Katiki Reddy Mam (Co-Founder)** for establishing a strong learning ecosystem that enables real-world exposure and applied learning.

---

## 12. License

This project is released for **educational and academic use**.


* Convert this into a **research-paper-style project report**
* Add **mathematical formulation section**
* Write an **evaluation/experiments section**
* Create a **conference-poster abstract**
* Align it with **IEEE / ACM format**

Just tell me which one.
