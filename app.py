

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re

# Streamlit app title
st.title("Code Generation and Debugging with Phi-2")

# Model loading
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "srikanthccc/phi2-mbpp-debug-merged"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Handle model loading errors
try:
    model, tokenizer = load_model_and_tokenizer()
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Post-process code
def postprocess_code(code, expected_func_name="is_palindrome", expected_param="s"):
    try:
        code_body = code.split("Completion:")[1].strip()
        code_body = re.split(r'(Test:|Exercise|Output:)', code_body)[0].strip()
        code_body = re.sub(r'def is_palindrome\([^)]+\)', f'def is_palindrome({expected_param})', code_body)
        lines = code_body.replace(":", ":\n    ").replace(" return ", "\n    return ").strip()
        lines = re.sub(r'\[:\s*:\s*-1\]', '[::-1]', lines)
        lines = re.sub(r'\[:::-1\]', '[::-1]', lines)
        lines = re.sub(r'\.reverse\(\)', '[::-1]', lines)
        lines = re.sub(r'\bstring\b', expected_param, lines)
        if not lines.startswith("def "):
            lines = f"def {expected_func_name}({expected_param}):\n    " + lines
        parts = lines.split(":\n", 1)
        if len(parts) > 1:
            return f"def {expected_func_name}({expected_param}):\n    {parts[1].strip()}"
        return lines
    except Exception as e:
        return code_body

# User input
prompt_type = st.selectbox("Select task:", ("Generate Code", "Debug Code"))
user_input = st.text_area("Enter your prompt:", height=150)

if prompt_type == "Generate Code":
    prompt = f"Prompt: {user_input}\nCompletion:"
else:
    prompt = f"Prompt: Fix this buggy code: {user_input}. Convert to lowercase, remove spaces, store the result in a variable, and compare it with its reverse using slice notation.\nCompletion:"

if st.button("Generate"):
    with st.spinner("Generating code..."):
        try:
            result = generator(prompt, max_new_tokens=150, do_sample=False)
            generated_code = result[0]['generated_text']
            processed_code = postprocess_code(generated_code)
            st.subheader("Generated Code:")
            st.code(generated_code, language="python")
            st.subheader("Post-processed Code:")
            st.code(processed_code, language="python")
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")