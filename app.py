import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
from get_context_from_api import infer_context_from_trends
import os
import gdown
import zipfile

API_KEY = "bbc5ace3aed5f02bcbd6affac66a496ed78fb1dd31f96cdbfffdcabf39bb0d0a"

# Google Drive file ID and paths
file_id = "15p_VKppO-VDLXHv6tspumyfRLCIqP_yN"
zip_path = "t5-query-rewriter-final.zip"
model_folder = "t5-query-rewriter-final"

# Download if the folder doesn't exist
if not os.path.exists(model_folder):
    if not os.path.exists(zip_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)
    # Unzip the model
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")


tokenizer = T5Tokenizer.from_pretrained(model_folder)
model = T5ForConditionalGeneration.from_pretrained(model_folder)

def sanitize(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def make_keywords_block(context_field, keep_top=8):
    if isinstance(context_field, list):
        kws = [sanitize(str(x)) for x in context_field if str(x).strip()]
    elif isinstance(context_field, str) and context_field.strip():
        kws = [sanitize(context_field)]
    else:
        kws = []
    if keep_top is not None:
        kws = kws[:keep_top]
    return "[" + ", ".join(kws) + "]"

def build_input(question, context_field, keep_top_keywords=8, strong=False):
    header = (
        "Instruction: Rewrite the question to be stand-alone. Ensure all HIGH-PRIORITY keywords appear if natural."
        if strong else
        "Instruction: Rewrite the question to be stand-alone. Prefer to include the given keywords if they are relevant."
    )
    kw_title = "HIGH-PRIORITY KEYWORDS:" if strong else "Keywords:"
    kw_block = make_keywords_block(context_field, keep_top_keywords)
    
    return "\n".join([
        header, "",
        kw_title, kw_block, "",
        "Question:", sanitize(question)
    ])

st.title("Query Rewriter")

question = st.text_input("Enter your question:")

# Slider to choose number of rewrites
num_rewrites = st.slider("Number of rewrites", min_value=1, max_value=5, value=3)

if question:
    context_result = infer_context_from_trends(question, API_KEY)
    context_keywords = context_result["top_related"]
    model_input = build_input(question, context_keywords)

    inputs = tokenizer(model_input, return_tensors="pt")

    # Generate rewrites
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=max(5, num_rewrites),  # beams >= num_return_sequences
        num_return_sequences=num_rewrites,
        early_stopping=True
    )

    st.subheader(f"Top {num_rewrites} Rewritten Questions:")
    for i, output in enumerate(outputs):
        rewritten_question = tokenizer.decode(output, skip_special_tokens=True)
        st.write(f"{i+1}. {rewritten_question}")
