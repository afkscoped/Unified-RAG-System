"""
Export Manager - Export and deploy fine-tuned models
"""

import streamlit as st
from pathlib import Path
import shutil
import zipfile
import json
from datetime import datetime
import pandas as pd


def render_export_manager():
    """Render export and deployment interface"""
    st.title("Export & Deploy")
    st.markdown("Export your fine-tuned models for deployment")
    
    # Find available models
    models_dir = Path("./data/fine_tuned_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    available_models = list(models_dir.glob("*/"))
    
    if not available_models:
        st.info("No fine-tuned models found. Train a model first in the Training Dashboard.")
        return
    
    # Model selection
    st.subheader("Select Model to Export")
    
    model_paths = {m.name: str(m) for m in available_models}
    
    selected_model = st.selectbox(
        "Choose model",
        options=list(model_paths.keys()),
        help="Select a fine-tuned model to export"
    )
    
    model_path = Path(model_paths[selected_model])
    
    # Display model info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Model Information")
        
        config_file = model_path / "training_config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.metric("Training Epochs", config.get('num_epochs', 'N/A'))
                st.metric("LoRA Rank", config.get('lora_r', 'N/A'))
            
            with info_col2:
                st.metric("Learning Rate", f"{config.get('learning_rate', 0):.2e}")
                st.metric("Batch Size", config.get('batch_size', 'N/A'))
            
            with st.expander("Full Configuration"):
                st.json(config)
        
        # Model files
        st.markdown("#### Model Files")
        files = list(model_path.glob("*"))
        
        file_info = []
        for f in files:
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                file_info.append({
                    'File': f.name,
                    'Size (MB)': f"{size_mb:.2f}"
                })
        
        if file_info:
            st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Quick Actions")
        
        # Model size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        st.metric("Total Size", f"{total_size / (1024 * 1024):.2f} MB")
        
        # Creation date
        st.metric("Created", datetime.fromtimestamp(model_path.stat().st_ctime).strftime("%Y-%m-%d"))
    
    st.markdown("---")
    
    # Export options
    st.subheader("Export Options")
    
    tab1, tab2, tab3 = st.tabs(["Save Archive", "Integration Code", "Usage Guide"])
    
    with tab1:
        render_archive_export(model_path, selected_model)
    
    with tab2:
        render_integration_guide(model_path, selected_model)
    
    with tab3:
        render_usage_guide(model_path, selected_model)


def render_archive_export(model_path: Path, model_name: str):
    """Render archive export interface"""
    st.markdown("### Create Downloadable Archive")
    
    st.markdown("""
    Export your model as a ZIP archive containing:
    - LoRA adapter weights
    - Training configuration
    - Tokenizer files
    - README with usage instructions
    """)
    
    archive_name = st.text_input(
        "Archive name",
        value=f"{model_name}_export",
        help="Name for the ZIP file"
    )
    
    include_readme = st.checkbox("Include README.md", value=True)
    
    if st.button("Create Archive", type="primary"):
        with st.spinner("Creating archive..."):
            try:
                # Create exports directory
                exports_dir = Path("./data/exports")
                exports_dir.mkdir(parents=True, exist_ok=True)
                
                archive_path = exports_dir / f"{archive_name}.zip"
                
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add all model files
                    for file in model_path.rglob("*"):
                        if file.is_file():
                            arcname = file.relative_to(model_path.parent)
                            zipf.write(file, arcname)
                    
                    # Add README if requested
                    if include_readme:
                        readme_content = generate_readme(model_name, model_path)
                        zipf.writestr(f"{model_name}/README.md", readme_content)
                
                st.success(f"Archive created: {archive_path}")
                
                # Provide download button
                with open(archive_path, 'rb') as f:
                    st.download_button(
                        label="Download Archive",
                        data=f,
                        file_name=f"{archive_name}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"Error creating archive: {str(e)}")


def render_integration_guide(model_path: Path, model_name: str):
    """Render integration instructions"""
    st.markdown("### Integration with HuggingFace Transformers")
    
    st.markdown("Use your fine-tuned model with the standard HuggingFace libraries:")
    
    code = f'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model (use your original base model)
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Replace with your base model
    load_in_4bit=True,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./{model_name}",  # Path to your exported model
    torch_dtype=torch.float16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./{model_name}")

# Generate
prompt = """### Instruction:
Your question here

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
'''
    
    st.code(code, language='python')
    
    st.markdown("---")
    
    st.markdown("### Merge Adapter with Base Model")
    
    st.info("For production deployment, you can merge the LoRA adapter with the base model:")
    
    merge_code = '''
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model (full precision for merging)
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "./your_model")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")

# Now you have a standard model without adapters
'''
    
    st.code(merge_code, language='python')


def render_usage_guide(model_path: Path, model_name: str):
    """Render usage guide and best practices"""
    st.markdown("### Usage Guide & Best Practices")
    
    st.markdown("""
    #### Recommended Use Cases
    
    Your fine-tuned model is optimized for:
    - Instruction following based on your training data
    - Domain-specific tasks from your dataset
    - Maintaining consistent output format
    
    #### Optimal Generation Settings
    
    For best results, use these generation parameters:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Factual/Precise Tasks:**
        - Temperature: 0.3 - 0.5
        - Top-p: 0.85 - 0.95
        - Max tokens: 128 - 512
        """)
    
    with col2:
        st.markdown("""
        **Creative Tasks:**
        - Temperature: 0.7 - 1.0
        - Top-p: 0.90 - 0.95
        - Max tokens: 256 - 1024
        """)
    
    st.markdown("---")
    
    st.markdown("""
    #### Prompt Template
    
    Use the same prompt format as training:
    """)
    
    st.code('''
### Instruction:
[Your clear, specific instruction here]

### Response:
[Model generates response here]
''')
    
    st.markdown("---")
    
    st.markdown("""
    #### Tips for Best Results
    
    - Keep prompts similar in style to training data
    - Use clear, specific instructions
    - Adjust temperature based on task requirements
    - Monitor output quality and iterate if needed
    - Consider additional fine-tuning for new domains
    """)


def generate_readme(model_name: str, model_path: Path) -> str:
    """Generate README content for model"""
    
    # Load config if available
    config_file = model_path / "training_config.json"
    config_text = "Configuration not available"
    
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            config_text = json.dumps(config, indent=2)
    
    readme = f"""# {model_name}

Fine-tuned language model created with QLoRA Training Studio.

## Model Details

- **Created**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Training Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Framework**: HuggingFace Transformers + PEFT

## Training Configuration

```json
{config_text}
```

## Usage

### Load Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model (replace with your base model)
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    load_in_4bit=True,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./{model_name}")
tokenizer = AutoTokenizer.from_pretrained("./{model_name}")
```

### Generate Text

```python
prompt = "### Instruction:\\nYour question here\\n\\n### Response:\\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Recommended Settings

- **Temperature**: 0.7 (adjust based on task)
- **Top-p**: 0.9
- **Max tokens**: 256-512

---

Created with QLoRA Training Studio
"""
    
    return readme
