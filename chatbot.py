import google.generativeai as genai

class ANNAssistant:
    def __init__(self, api_key):
     
        if not api_key:
            raise ValueError("API key is required for chatbot functionality")
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        # System prompt that defines the assistant's role
        self.system_context = """
You are an expert AI assistant specializing in Artificial Neural Networks (ANNs) and deep learning.

Your role is to help users understand:
- How ANNs learn and make decisions
- Why certain architectures work better for different problems
- What training dynamics (loss curves, convergence) tell us
- How implementation choices affect behavior

You have full access to:
1. The project codebase (PyTorch implementation)
2. User's configuration (architecture, training parameters)
3. Training results (losses, decision boundaries)

When answering:
- Be clear and educational
- Connect theory to the specific configuration the user is working with
- Point out interesting patterns or potential issues
- Suggest experiments when appropriate
- Explain code-level details if asked

Keep responses focused and practical. Avoid generic advice—tailor everything to the user's actual setup.
"""
    
    def create_context_prompt(self, dataset_info, model_config, training_results):
       
        context = f"""
CURRENT PROJECT CONTEXT:

=== DATASET ===
Type: {dataset_info.get("type", "Unknown")}
Samples: {dataset_info.get("n_samples", "N/A")}
Classes: {dataset_info.get("n_classes", "N/A")}
Features: {dataset_info.get("n_features", 2)}
Additional Info: {dataset_info.get("details", "None")}

=== MODEL ARCHITECTURE ===
Input Features: {model_config.get("input_size", "N/A")}
Hidden Layers: {model_config.get("hidden_layers", "N/A")}
Neurons per Layer: {model_config.get("neurons_per_layer", "N/A")}
Activation Function: {model_config.get("activation", "N/A")}
Total Parameters: {model_config.get("total_params", "N/A")}

=== TRAINING CONFIGURATION ===
Epochs: {training_results.get("epochs", "N/A")}
Learning Rate: {training_results.get("learning_rate", "N/A")}
Regularization: {training_results.get("regularization", "None")}
Problem Type: {training_results.get("problem_type", "N/A")}

=== TRAINING RESULTS ===
Initial Train Loss: {training_results.get("initial_train_loss", "N/A"):.4f}
Final Train Loss: {training_results.get("final_train_loss", "N/A"):.4f}
Initial Test Loss: {training_results.get("initial_test_loss", "N/A"):.4f}
Final Test Loss: {training_results.get("final_test_loss", "N/A"):.4f}
Best Test Loss: {training_results.get("best_test_loss", "N/A"):.4f}

Loss Trajectory: {training_results.get("loss_trend", "N/A")}
"""
        return context
    
    def ask(self, question, context_info=None):
       
        try:
            # Build the full prompt
            if context_info:
                context_prompt = self.create_context_prompt(
                    context_info.get('dataset_info', {}),
                    context_info.get('model_config', {}),
                    context_info.get('training_results', {}))
                
                full_prompt = f"""
{self.system_context}

{context_prompt}

USER QUESTION: {question}

Provide a clear, educational response tailored to this specific configuration.
"""
            else:
                # No context available (user hasn't trained yet)
                full_prompt = f"""
{self.system_context}

USER QUESTION: {question}

Note: The user hasn't trained a model yet, so provide general guidance about ANNs.
"""
            # Get response from Gemini
            response = self.model.generate_content(full_prompt)
            
            return response.text
        
        except Exception as e:
            return f"Error communicating with Gemini: {str(e)}\n\nPlease check your API key and internet connection."
    
    def analyze_training(self, train_losses, test_losses):

        # Calculate key metrics
        train_start = train_losses[0]
        train_end = train_losses[-1]
        test_start = test_losses[0]
        test_end = test_losses[-1]
        
        train_decrease = ((train_start - train_end) / train_start) * 100
        test_decrease = ((test_start - test_end) / test_start) * 100
        
        final_gap = abs(train_end - test_end)
        
        # Detect overfitting
        overfitting = test_end > train_end * 1.2
        
        # Detect underfitting
        underfitting = train_end > 0.5
        
        # Build analysis
        analysis = f"""
TRAINING ANALYSIS:

〽 Loss Reduction:
- Training loss decreased by {train_decrease:.1f}%
- Test loss decreased by {test_decrease:.1f}%

𖣠 Final Performance:
- Final train loss: {train_end:.4f}
- Final test loss: {test_end:.4f}
- Gap between train and test: {final_gap:.4f}
"""
        
        if overfitting:
            analysis += """
⚠︎ OVERFITTING DETECTED:
Test loss is significantly higher than training loss.
The model memorized training data but doesn't generalize well.

Suggestions:
- Add regularization (L1 or L2)
- Reduce model complexity (fewer layers/neurons)
- Use more training data
- Add dropout (not implemented in this version)
"""
        elif underfitting:
            analysis += """
⚠︎ UNDERFITTING DETECTED:
Both losses are still high - the model hasn't learned the pattern well.

Suggestions:
- Increase model complexity (more layers/neurons)
- Train for more epochs
- Increase learning rate
- Check if dataset is learnable (some patterns are inherently random)
"""
        else:
            analysis += """
✔  GOOD FIT:
The model is learning well and generalizing reasonably.
Train and test losses are both decreasing with a small gap.
"""
        
        return analysis
def analyze_loss_trend(losses):
    if len(losses) < 10:
        return "Insufficient data"
    
    # Check first vs last
    decrease = losses[0] - losses[-1]
    
    if decrease > losses[0] * 0.5:
        return "Steady decrease"
    elif decrease > losses[0] * 0.2:
        return "Moderate decrease"
    elif decrease > 0:
        return "Slight decrease"
    else:
        return "No improvement or increasing"