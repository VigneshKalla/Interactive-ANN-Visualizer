"""
Interactive ANN Visualizer - Main Application

This is the entry point for the Streamlit web application.
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import torch
# Import our custom modules
from datasets import get_predefined_dataset, create_custom_dataset
from model import FlexibleANN
from trainer import ANNTrainer
from visualizer import plot_snapshot_grid, plot_loss_curves, plot_dataset_preview
from chatbot import ANNAssistant, analyze_loss_trend

# Page configuration
st.set_page_config(page_title="Interactive ANN Visualizer", page_icon="💡",
                   layout="wide", # "centered"
                   initial_sidebar_state="collapsed")

def initialize_session_state():
    
    if "trained" not in st.session_state:
        st.session_state.trained = False
    
    if "training_results" not in st.session_state:
        st.session_state.training_results = None
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    
    if "dataset_preview" not in st.session_state:
        st.session_state.dataset_preview = None
    
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None
    
    if "current_model_config" not in st.session_state:
        st.session_state.current_model_config = None
    # Initialize chat history for popup-like experience
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False

def render_onboarding():
  
    st.title("💡:violet[Interactive ANN Visualizer]", anchor=None, help=None, 
             width="stretch", text_alignment="center")
    st.subheader(":grey[Understand How Neural Networks Learn Through Visualization]", 
                 width="stretch", text_alignment="center", divider="grey"); st.space(size="xxsmall")
    
# Introduction
    st.markdown("""
    <div style="text-align: center;">
        <h4>What This Application Does</h4>
        <p>This interactive tool lets you <b>visualize how Artificial Neural Networks (ANNs) learn</b> by showing you:</p>
        <ul style="list-style-position: inside; display: inline-block; text-align: left;">
            <li>📊 <b>Decision Boundaries</b>: See how the model separates different classes in 2D space</li>
            <li>📈 <b>Learning Evolution</b>: Watch how boundaries change from random initialization to trained state</li>
            <li>📉 <b>Loss Curves</b>: Track training and test performance over time</li>
            <li>⚛ <b>AI Explanations</b>: Ask an AI assistant to explain what's happening</li>
        </ul>
        <h4>🚀 How to Use</h4>
        <ol style="list-style-position: inside; display: inline-block; text-align: left;">
            <li><b>Configure the sidebar</b> (left) to choose your dataset and model architecture</li>
            <li><b>Set training parameters</b> like epochs, learning rate, and regularization</li>
            <li><b>Click Train</b> to watch the model learn</li>
            <li><b>Explore results</b> through visualizations and loss curves</li>
            <li><b>Ask the chatbot</b> (optional) to understand behavior in depth</li>
        </ol>
        <h4>🎓 Learning Goals</h4>
        <p>By experimenting with different configurations, you'll understand:</p>
        <ul style="list-style-position: inside; display: inline-block; text-align: left;">
            <li>Why deeper networks can solve more complex problems</li>
            <li>How regularization prevents overfitting</li>
            <li>Why some datasets need non-linear activation functions</li>
            <li>How learning rate affects convergence speed</li>
        </ul>
        <br><b>👈 Get started by configuring the sidebar!</b>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():

    st.sidebar.markdown("## ⚙️ Configuration"); st.sidebar.divider()
    # API Key Section
    st.sidebar.markdown("#### 🔑 API Key")
    
    api_key = st.sidebar.text_input(
        "Google Gemini API Key", type="password", placeholder="Paste your API key here",
        help="Get your API key from https://aistudio.google.com/?project=gen-lang-client-0807966959")
    st.sidebar.caption("Required for chatbot functionality")

    # Initialize chatbot if API key is provided
    if api_key and st.session_state.chatbot is None:
        try:
            st.session_state.chatbot = ANNAssistant(api_key)
            st.sidebar.success("✔ Chatbot initialized!")
        except Exception as e:
            st.sidebar.error(f"✘ Failed to initialize chatbot: {e}")

    st.sidebar.markdown("---")
    
    # Dataset Selection
    st.sidebar.markdown("### 🧪 Dataset Configuration")
    
    dataset_mode = st.sidebar.radio(
        "Select Dataset Mode",
        ["Predefined Datasets", "Custom Dataset"],
        help=(
        "Predefined: Ready-made benchmark datasets (Moons, Circles, XOR...)\n\n"
        "Custom: Generate your own dataset by tuning parameters"),key="dataset_mode")
    
    config = {"dataset_mode": dataset_mode}
    
    if dataset_mode == "Predefined Datasets":
        config.update(render_predefined_dataset_config())
    else:
        config.update(render_custom_dataset_config())
    
    st.sidebar.markdown("---")
    
    # Model Architecture
    st.sidebar.markdown("### 🏗️ Model Architecture")
    # Number of hidden layers
    config["hidden_layers"] = st.sidebar.slider(
        "Number of Hidden Layers",
        min_value=0, max_value=20, value=2, help="0 = Linear model (no hidden layers)")
    
    # Neurons per layer (only if hidden layers > 0)
    config["neurons_per_layer"] = st.sidebar.slider(
        "Neurons per Hidden Layer",
        min_value=6, max_value=152, value=16, step=4,
        help="Higher = more capacity, slower training, higher overfitting risk")
    
    st.sidebar.markdown("---")
    
    # Training Configuration
    st.sidebar.markdown("### 🎯 Training Configuration")
    
    config["epochs"] = st.sidebar.number_input(
        "Epochs", min_value=10, max_value=10000, value=100, step=10,
        help="Number of times to iterate through the dataset")
    
    learning_rates = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3,10]
    config["learning_rate"] = st.sidebar.selectbox(
        "Learning Rate", learning_rates, index=5,
        help="Step size for weight updates (higher = faster but less stable)")
    
    config["activation"] = st.sidebar.selectbox(
        "Activation Function", ["ReLu", "Sigmoid", "Tanh", "Linear"],
        help="ReLU: Most common | Sigmoid/Tanh: Classic | Linear: No non-linearity")
    
    config["regularization"] = st.sidebar.selectbox(
        "Regularization", ["None", "L1", "L2"],
        help="Prevents overfitting by penalizing large weights")
    
    if config["regularization"] != "None":
        config["reg_strength"] = st.sidebar.number_input(
            "Regularization Strength",
            min_value=0.0001, max_value=1.0, value=0.01, step=0.001, format="%.4f",
            help="Higher values = stronger regularization")
    else:
        config["reg_strength"] = 0.0
    
    config["problem_type"] = st.sidebar.selectbox(
        "Problem Type", ["Classification", "Regression"],
        help="Classification: Discrete classes | Regression: Continuous values")
    
    return config; st.sidebar.markdown("---")

def render_predefined_dataset_config():
    """Render configuration for predefined datasets."""
    dataset_name = st.sidebar.selectbox("**Select Dataset**",
        ["Moons", "Circles", "Classification", "XOR"],
        help="Each dataset tests different model capabilities")
    
    n_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=50, max_value=5000, value=200, step=50)
    
    noise = st.sidebar.slider(
        "Noise Level",
        min_value=0.0, max_value=0.5, value=0.1, step=0.05,
        help="Higher noise makes the problem harder")
    
    return {"dataset_type": "predefined", "dataset_name": dataset_name,
            "n_samples": n_samples, "noise": noise}

def render_custom_dataset_config():
    """Render configuration for custom datasets."""
    n_samples = st.sidebar.slider(
        "Number of Samples", min_value=50, max_value=5000, value=200, step=50)
    
    n_classes = st.sidebar.slider(
        "Number of Classes", min_value=2, max_value=5, value=2,
        help="2 = Binary classification, 3+ = Multi-class")
    
    noise = st.sidebar.slider("Noise Level", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    
    # Class weights (optional advanced feature)
    use_imbalance = st.sidebar.checkbox(
        "Use Class Imbalance", value=False,
        help="Create unbalanced dataset (useful for testing robustness)")
    
    class_weights = None
    if use_imbalance:
        st.sidebar.markdown("**Class Weights** (will be normalized)")
        class_weights = []
        for i in range(n_classes):
            weight = st.sidebar.slider(
                f"Class {i} Weight",
                min_value=0.1, max_value=5.0, value=1.0,step=0.1)
            class_weights.append(weight)
    
    return {"dataset_type": "custom", "n_samples": n_samples,
        "n_classes": n_classes, "noise": noise, "class_weights": class_weights}

def generate_dataset(config):
 
    if config["dataset_type"] == "predefined":
        X, y = get_predefined_dataset(
            config["dataset_name"], n_samples=config["n_samples"], noise=config["noise"])
    else:
        X, y = create_custom_dataset(
            n_samples=config["n_samples"], n_classes=config["n_classes"], noise=config["noise"],
            class_weights=config["class_weights"])
    
    return X, y

def train_model(config, X, y):
   
    # Determine input/output sizes
    input_size = X.shape[1]  # Should be 2 for visualization
    
    if config["problem_type"].lower() == "classification":
        output_size = len(np.unique(y))
    else:
        output_size = 1
    
    # Create model
    model = FlexibleANN(
        input_size=input_size,
        hidden_layers=config["hidden_layers"],
        neurons_per_layer=config["neurons_per_layer"],
        output_size=output_size,
        activation=config["activation"])
    
    # Store model config for chatbot
    model_config = {
        "input_size": input_size,
        "hidden_layers": config["hidden_layers"],
        "neurons_per_layer": config["neurons_per_layer"],
        "output_size": output_size,
        "activation": config["activation"],
        "total_params": model.count_parameters()}
    
    # Create trainer
    trainer = ANNTrainer(model, problem_type=config["problem_type"].lower())
    
    # Train
    reg_type = None if config["regularization"] == "None" else config["regularization"].lower()
    
    results = trainer.train(
        X, y,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        reg_type=reg_type,
        reg_strength=config["reg_strength"])
    
    # Store everything for visualization and chatbot
    results["model"] = model
    results["model_config"] = model_config
    results["X"] = X
    results["y"] = y
    results["config"] = config
    
    return results

def render_training_results(results):

    st.markdown("### 📊 Training Results")
    
    # Create two columns: left for snapshots, right for losses
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### 🎯 Decision Boundary Evolution")
        st.markdown("*Watch how the model's understanding improves across epochs*")
        
        # Generate 2x2 grid
        fig_snapshots = plot_snapshot_grid(
            snapshots=results["snapshots"],
            snapshot_epochs=results["snapshot_epochs"],
            model_class=FlexibleANN,
            model_config={
                "input_size": results["model_config"]["input_size"],
                "hidden_layers": results["model_config"]["hidden_layers"],
                "neurons_per_layer": results["model_config"]["neurons_per_layer"],
                "output_size": results["model_config"]["output_size"],
                "activation": results["model_config"]["activation"]},
            X=results["X"],
            y=results["y"])
        
        st.pyplot(fig_snapshots)
    
    with col2:
        st.markdown("#### 📈 **Loss Curves**")
        st.markdown("**Training vs Test Loss**")
        
        fig_loss = plot_loss_curves(results["train_losses"], results["test_losses"])
        
        st.pyplot(fig_loss)
        
        # Add summary metrics
        st.markdown("---")
        st.markdown("#### 📊 **Summary Metrics**")
        
        final_train_loss = results["train_losses"][-1]
        final_test_loss = results["test_losses"][-1]
        best_test_loss = min(results["test_losses"])
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Final Train Loss",  f"{final_train_loss:.4f}")
        
        with metric_col2:
            st.metric("Final Test Loss", f"{final_test_loss:.4f}")
        
        with metric_col3:
            st.metric("Best Test Loss", f"{best_test_loss:.4f}")

def render_chatbot_interface(results):
  
    if st.session_state.chatbot is None:
        st.info("💡 **Add your Gemini API key in the sidebar to enable the AI chatbot!**")
        return
    
    st.markdown("---")
    st.markdown("### ⚛ **AI Assistant**")
    st.markdown("*Ask questions about your model, dataset, or training results*")
    
    # Create context for chatbot
    dataset_info = {
        "type": results["config"].get("dataset_name", "Custom"),
        "n_samples": len(results["X"]),
        "n_classes": len(np.unique(results["y"])),
        "n_features": results["X"].shape[1],
        "details": f"Noise: {results["config"].get("noise", "N/A")}"}
    
    training_results_info = {
        "epochs": results["config"]["epochs"],
        "learning_rate": results["config"]["learning_rate"],
        "regularization": results["config"]["regularization"],
        "problem_type": results["config"]["problem_type"],
        "initial_train_loss": results["train_losses"][0],
        "final_train_loss": results["train_losses"][-1],
        "initial_test_loss": results["test_losses"][0],
        "final_test_loss": results["test_losses"][-1],
        "best_test_loss": min(results["test_losses"]),
        "loss_trend": analyze_loss_trend(results["test_losses"])}
    
    context_info = {
        "dataset_info": dataset_info,
        "model_config": results["model_config"],
        "training_results": training_results_info}
    
    # Quick analysis button
    if st.button("### 📊 Auto-Analyze Training"):
        with st.spinner("Analyzing..."):
            analysis = st.session_state.chatbot.analyze_training(
                results["train_losses"], results["test_losses"])
            st.markdown(analysis)
    
    st.markdown("---")
    
    # Chat interface
    with st.expander("💬 **Open Chat Interface**", expanded=False):
        st.markdown("**Ask me anything about your training:**")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.markdown(f"**Q:** {question}")
                with st.chat_message("assistant"):
                    st.markdown(answer)
            st.markdown("---")
        
        # Chat input - FIX: Using st.chat_input instead of incorrect method
        user_question = st.chat_input(
            placeholder="Why did my test loss increase? How does architecture affect learning?",
            key="chatbot_input")
        
        if user_question:
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.ask(user_question, context_info)
                
                # Add to chat history
                st.session_state.chat_history.append((user_question, response))
                
                # Rerun to display new message
                st.rerun()
        
        # Clear chat button
        if st.session_state.chat_history and st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

def main():
    """Main application entry point.""" 
    # Initialize
    initialize_session_state()
    # Render sidebar and get configuration
    config = render_sidebar()
    # Main canvas
    if not st.session_state.trained:
        # Show onboarding
        render_onboarding()
        
        # Add dataset preview button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🔍︎ Preview Dataset", use_container_width=True):
                with st.spinner("Generating dataset..."):
                    X, y = generate_dataset(config)
                    st.session_state.current_dataset = (X, y)
                    
                    fig = plot_dataset_preview(X, y, title=f"Dataset Preview: {config.get("dataset_name", "Custom")}")
                    st.pyplot(fig)
        
        # Train button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🚀 Train Model", use_container_width=True, type="primary"):
                with st.spinner("Training... This may take a moment."):
                    # Generate dataset
                    X, y = generate_dataset(config)
                    
                    # Train model
                    results = train_model(config, X, y)
                    
                    # Store results
                    st.session_state.training_results = results
                    st.session_state.trained = True
                    
                    # Trigger rerun to show results
                    st.rerun()
    
    else:
        # Show training results
        render_training_results(st.session_state.training_results)
        
        # Show chatbot interface
        render_chatbot_interface(st.session_state.training_results)
        
        # Reset button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🔄 Train New Model", use_container_width=True):
                st.session_state.trained = False
                st.session_state.training_results = None
                st.rerun()

if __name__ == "__main__":
    main()