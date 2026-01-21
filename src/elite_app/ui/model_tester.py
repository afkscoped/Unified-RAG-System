"""
Model Tester - Interactive testing interface for fine-tuned models
"""

import streamlit as st
from pathlib import Path
import json


def render_model_tester():
    """Render model testing interface"""
    st.title("Model Tester")
    st.markdown("Test your fine-tuned models interactively")
    
    # Model selection
    st.subheader("Load Fine-Tuned Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Find available fine-tuned models
        models_dir = Path("./data/fine_tuned_models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        available_models = list(models_dir.glob("*/"))
        
        if available_models:
            model_paths = {m.name: str(m) for m in available_models}
            
            selected_model = st.selectbox(
                "Select fine-tuned model",
                options=list(model_paths.keys()),
                help="Choose a model to test"
            )
            
            model_path = model_paths[selected_model]
            
            # Show model info
            config_file = Path(model_path) / "training_config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                
                with st.expander("Model Configuration"):
                    st.json(config)
        else:
            st.info("No fine-tuned models found. Train a model in the Training Dashboard first.")
            selected_model = None
            model_path = None
    
    with col2:
        if selected_model and st.button("Load Model", use_container_width=True):
            with st.spinner("Loading model..."):
                try:
                    # Load base model if not already loaded
                    if st.session_state.current_model_loaded is None:
                        st.warning("Loading base model first...")
                        model_manager = st.session_state.model_manager
                        
                        # Try to infer base model from config
                        base_model_name = "TinyLlama-1.1B"  # Default to smallest for testing
                        
                        model, tokenizer = model_manager.load_model(base_model_name)
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.current_model_loaded = base_model_name
                    
                    # Load inference engine with adapter
                    from src.elite_app.core.inference_engine import InferenceEngine
                    
                    inference_engine = InferenceEngine(
                        base_model=st.session_state.model,
                        tokenizer=st.session_state.tokenizer,
                        adapter_path=model_path
                    )
                    
                    st.session_state.inference_engine = inference_engine
                    st.session_state.loaded_adapter_path = model_path
                    
                    st.success("Model loaded successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
    st.markdown("---")
    
    # Testing interface
    if st.session_state.get('inference_engine') is not None:
        st.subheader("Interactive Testing")
        
        # Test mode selection
        test_mode = st.radio(
            "Test Mode",
            ["Single Prompt", "Chat Conversation"],
            horizontal=True
        )
        
        if test_mode == "Single Prompt":
            render_single_prompt_test()
        else:
            render_chat_test()
    else:
        st.info("Load a fine-tuned model to start testing")
        
        # Show example prompts
        st.markdown("### Example Test Prompts")
        st.code('''
### Instruction:
Explain the benefits of fine-tuning language models.

### Instruction:
Write a Python function to calculate factorial.

### Instruction:
What are the key differences between supervised and unsupervised learning?
        ''')


def render_single_prompt_test():
    """Render single prompt testing interface"""
    
    # Prompt input
    prompt = st.text_area(
        "Enter your prompt",
        height=150,
        placeholder="### Instruction:\nYour question or task here...\n\n### Response:\n",
        help="Use the instruction format your model was trained on"
    )
    
    # Generation parameters
    with st.expander("Generation Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_tokens = st.slider("Max Tokens", 50, 1024, 256)
        with col2:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        with col3:
            top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)
    
    # Generate button
    if st.button("Generate", type="primary", use_container_width=True):
        if not prompt:
            st.warning("Please enter a prompt")
        else:
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.inference_engine.generate(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    
                    # Display response
                    st.markdown("### Generated Response")
                    st.markdown(response)
                    
                    # Save to history
                    if 'test_history' not in st.session_state:
                        st.session_state.test_history = []
                    
                    st.session_state.test_history.append({
                        'prompt': prompt,
                        'response': response,
                        'params': {
                            'max_tokens': max_tokens,
                            'temperature': temperature,
                            'top_p': top_p
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Generation error: {str(e)}")
    
    # Show history
    if st.session_state.get('test_history'):
        st.markdown("---")
        st.subheader("Test History")
        
        for i, item in enumerate(reversed(st.session_state.test_history[-5:])):
            with st.expander(f"Test {len(st.session_state.test_history) - i}"):
                st.markdown("**Prompt:**")
                st.code(item['prompt'])
                st.markdown("**Response:**")
                st.markdown(item['response'])


def render_chat_test():
    """Render chat conversation testing interface"""
    
    # Initialize chat history
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat history
    st.markdown("### Conversation")
    
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_messages:
            if msg['role'] == 'user':
                st.chat_message("user").write(msg['content'])
            else:
                st.chat_message("assistant").write(msg['content'])
    
    # Chat input
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Add user message
        st.session_state.chat_messages.append({
            'role': 'user',
            'content': user_input
        })
        
        # Generate response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.inference_engine.chat(
                    messages=st.session_state.chat_messages,
                    max_new_tokens=256
                )
                
                # Add assistant response
                st.session_state.chat_messages.append({
                    'role': 'assistant',
                    'content': response
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Clear chat button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()
