# **🔗 Interactive ANN Visualizer**

An educational web application that visualizes how Artificial Neural Networks (ANNs) learn, built with **PyTorch**, **Streamlit**, and powered by an **AI agent (Gemini)** for contextual explanations.

---

## **What This Does**

This application lets you:

1. **Configure and train neural networks** with custom architectures
2. **Visualize decision boundaries** evolving across 4 key training stages (0%, 25%, 75%, 100%)
3. **Track training/test loss** to understand convergence and overfitting
4. **Ask an AI assistant** to explain model behavior, architecture choices, and training dynamics

### Key Features

- **Predefined datasets**: moons, circles, classification, XOR
- **Custom datasets**: Configure samples, classes, noise, and imbalance
- **Flexible ANN architecture**: Variable hidden layers, neurons, and activation functions
- **Agentic AI chatbot**: Context-aware explanations using Gemini

---

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Google Gemini API key (optional, for chatbot only)

---

##  🔗 Installation

### 1. Clone or Download

```bash
# If using git
git clone <https://github.com/VigneshKalla/Interactive-ANN-Visualizer.git>
cd <project-directory>

# Or download and extract the ZIP file
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `torch` - PyTorch for neural networks
- `streamlit` - Web application framework
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `scikit-learn` - Dataset generation and utilities
- `google-generativeai` - Gemini API for chatbot

### 3. Get Gemini API Key (Optional)

The chatbot requires a Google Gemini API key:

1. Go to [Google AI Studio]([https://makersuite.google.com/app/apikey](https://aistudio.google.com/welcome?utm_source=google&utm_medium=cpc&utm_campaign=Cloud-SS-DR-AIS-FY26-global-gsem-1713578&utm_content=text-ad&utm_term=KW_google%20ai%20studio&gad_source=1&gad_campaignid=23417416052&gbraid=0AAAAACn9t66B_Lp12xWHm_axM7EtLkNGl&gclid=CjwKCAiAqKbMBhBmEiwAZ3UboGzc5ZHfPLj7CuCwQ5j59F4QYcEGB619UOKKFcT9vca8O7uHknHWCxoCk0IQAvD_BwE))
2. Sign in with your Google account
3. Create a new API key
4. Copy it for use in the application

**Note:** The core training and visualization features work without an API key. Only the chatbot requires it.

---

##  Usage

### Starting the Application

```bash
streamlit run app.py
```

This will:
- Start a local web server
- Open your default browser to `http://localhost:8501`
- Display the application interface

### Step-by-Step Workflow

#### 1. **Configure the Sidebar**

**Dataset Selection:**
- Choose "Predefined Datasets" (moons, circles, classification, XOR)
- Or "Create Custom Dataset" (configure samples, classes, noise, imbalance)

**Model Architecture:**
- Set number of hidden layers (0-5)
- Set neurons per layer (2-128)

**Training Configuration:**
- Epochs: How many times to iterate through the data (10-1000)
- Learning rate: Step size for weight updates (0.0001-1.0)
- Activation function: ReLU, Sigmoid, Tanh, or Linear
- Problem type: Classification or Regression

#### 2. **Preview Dataset (Optional)**

Click "Preview Dataset" to see the data distribution before training.

#### 3. **Train the Model**

Click "Train Model" to start training. The application will:
- Generate the dataset
- Build the neural network
- Train for the specified epochs
- Capture snapshots at 0%, 25%, 75%, and 100% progress

#### 4. **Analyze Results**

**Left Panel - Decision Boundary Evolution:**
- 2×2 grid showing how decision boundaries change during training
- Each subplot shows the model's classification at a different stage

**Right Panel - Loss Curves:**
- Training loss (blue): How well the model fits training data
- Test loss (red): How well it generalizes to unseen data
- Summary metrics: Final and best test loss

#### 5. **Ask the Chatbot (Optional)**

If you provided a Gemini API key:
- Click "Auto-Analyze Training" for automatic insights
- Or type questions in the text box:
  - "Why did my test loss increase after epoch 50?"
  - "How does adding more hidden layers help?"
  - "What does L2 regularization do?"
  - "Why is ReLU better than Sigmoid for this dataset?"

The chatbot has full context:
- Your dataset configuration
- Model architecture
- Training parameters
- Loss curves and metrics

#### 6. **Experiment**

Click "Train New Model" to reset and try different configurations.

---

## 📂 Project Structure

```
project/
├── app.py              # Main Streamlit application
├── model.py            # PyTorch ANN definition
├── trainer.py          # Training loop with snapshots
├── datasets.py         # Dataset generators
├── visualizer.py       # Plotting functions
├── chatbot.py          # Gemini-powered AI assistant
├── requirements.txt    # Dependencies
└── README.md           # This file
```

### Module Descriptions

**`app.py`**
- Streamlit UI and user interaction
- Orchestrates dataset → model → training → visualization workflow
- Manages chatbot interface

**`model.py`**
- `FlexibleANN` class: Configurable neural network
- Support for variable layers, neurons, activations
- Regularization calculation (L1/L2)

**`trainer.py`**
- `ANNTrainer` class: Handles training loop
- Captures model snapshots at key epochs
- Tracks training/test losses
- Supports classification and regression

**`datasets.py`**
- Predefined dataset generators (moons, circles, classification, XOR)
- Custom dataset creation with configurable parameters
- Visualization-friendly 2D data

**`visualizer.py`**
- Decision boundary plotting
- 2×2 snapshot grid generation
- Loss curve plotting
- Dataset preview plots

**`chatbot.py`**
- `ANNAssistant` class: Gemini-powered explanations
- Context-aware prompting with full project information
- Automatic training analysis
- Code-level explanations on request

---

## 🔗 Educational Concepts

### Why This Matters

This application bridges the gap between:
- **Theory**: What neural networks should do
- **Implementation**: How they're built in code
- **Behavior**: What actually happens during training

### Key Learning Points

**1. Architecture Impact**
- **Zero hidden layers** = Linear model (can't solve XOR)
- **One hidden layer** = Can approximate any continuous function (universal approximation)
- **Multiple layers** = Hierarchical feature learning

**2. Activation Functions**
- **ReLU**: Fast, works well for deep networks, but can "die" (neurons output zero forever)
- **Sigmoid**: Outputs [0,1], but suffers from vanishing gradients in deep networks
- **Tanh**: Outputs [-1,1], zero-centered (better than sigmoid)
- **Linear**: No non-linearity, only for regression or debugging

**3. Training Dynamics**
- **Decreasing train/test loss**: Good, model is learning
- **Train loss low, test loss high**: Overfitting (model memorizes training data)
- **Both losses high**: Underfitting (model is too simple or not trained enough)
- **Test loss plateaus**: Model reached its capacity

**4. Dataset Characteristics**
- **Moons**: Non-linear but separable with simple boundary
- **Circles**: Requires depth (one circle inside another)
- **XOR**: Classic non-linear problem, impossible for single-layer networks
- **Custom**: Test effects of noise, imbalance, and multi-class complexity

---

##  Troubleshooting

### Common Issues

**1. Import Errors**
```bash
ModuleNotFoundError: No module named 'torch'
```
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

**2. Chatbot Not Working**
```
Error communicating with Gemini: API key not valid
```
**Solution:**
- Check your API key is correct
- Ensure you have internet connection
- Verify the API key has proper permissions

**3. Visualization Not Showing**
- Ensure you clicked "Train Model" before expecting results
- Check browser console for errors
- Try refreshing the page

**4. Training Takes Too Long**
- Reduce epochs (try 50 instead of 100)
- Reduce number of samples (try 100 instead of 200)
- Reduce model complexity (fewer layers/neurons)

---

## 🔗  Customization Ideas

### Extend the Project

1. **Add New Datasets**
   - Edit `datasets.py` to add more predefined datasets
   - Example: spirals, blobs, multi-modal distributions

2. **Add Dropout Regularization**
   - Modify `model.py` to include `nn.Dropout` layers
   - Add UI controls in `app.py`

3. **Add More Activation Functions**
   - LeakyReLU, ELU, SELU
   - Update `model.py` and `app.py`

4. **Batch Normalization**
   - Add `nn.BatchNorm1d` between layers
   - Helps with training stability

5. **Early Stopping**
   - Stop training when test loss stops improving
   - Prevents overfitting automatically

6. **Export Trained Models**
   - Add button to save model weights
   - Allow loading pre-trained models

---

##  Example Experiments

### Experiment 1: Understanding XOR

**Goal:** See why single-layer networks fail on XOR

**Setup:**
1. Dataset: XOR
2. Hidden layers: 0 (linear model)
3. Train for 100 epochs

**Expected Result:** Decision boundary will be a straight line, unable to separate XOR pattern

**Then Try:**
- Hidden layers: 1 with 4 neurons
- Watch boundary become non-linear and solve XOR

### Experiment 2: Overfitting vs Regularization

**Goal:** See how regularization prevents overfitting

**Setup:**
1. Dataset: Moons with high noise (0.3)
2. Hidden layers: 3, neurons: 64 (very complex model)
3. No regularization
4. Train for 200 epochs

**Expected Result:** Training loss very low, test loss high (overfitting)

**Then Try:**
- Add L2 regularization (strength 0.01)
- Watch the gap between train/test loss shrink

### Experiment 3: Learning Rate Impact

**Goal:** See how learning rate affects convergence

**Setup:**
1. Dataset: Circles
2. Learning rate: 0.001 (very slow)

**Expected Result:** Loss decreases very slowly, may not converge in 100 epochs

**Then Try:**
- Learning rate: 0.1 (fast)
- May oscillate or diverge

**Optimal:**
- Learning rate: 0.01
- Smooth convergence

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Add new features
- Improve documentation
- Create additional visualizations
- Share interesting experiment results

---

## 📝 License

This project is for educational purposes. Feel free to use and modify as needed.

---

##  Acknowledgments

- **PyTorch** for the deep learning framework
- **Streamlit** for the simple web app interface
- **scikit-learn** for dataset generators
- **Google Gemini** for the AI assistant
- **The neural network research community** for foundational concepts

---

##  Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments (heavily documented)
3. Ask the AI chatbot (it knows the codebase!)
4. Open an issue in the repository

---

## 🔗  Acknowledgements

I would like to express my sincere gratitude to **Saxon K. Sha Sir** for his continuous technical guidance, **Lakshmi Vangapandu Mam** for her clear, supportive, and encouraging mentorship, and **Raghu Ram Aduri Sir** for his valuable managerial support and strategic direction throughout this project.

I am also thankful to **Innomatics Research Labs** for providing a structured, growth-oriented learning environment. Special thanks to **Vishwanath Nyathani Sir** (Founder) and **Kalpana Katiki Reddy Mam** (Co-Founder) for building a strong learning ecosystem that enables real-world exposure and practical opportunities.


---

**Happy Learning!**

*Remember: The best way to understand neural networks is to experiment. Try different configurations, observe what happens, and ask the chatbot to explain. That's what this tool is for.*
