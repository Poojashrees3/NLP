import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import pandas as pd

# Load model and tokenizer
@st.cache_resource
def load_model_pipeline():
    model = BertForSequenceClassification.from_pretrained("models/Bert_trainer")
    tokenizer = BertTokenizer.from_pretrained("models/Bert_trainer")
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)

pipeline = load_model_pipeline()

@st.cache_resource
def load_text_generator_model():
    model_name = "gpt2"
    return pipeline("text-generation",
                    model=GPT2LMHeadModel.from_pretrained(model_name),
                    tokenizer=GPT2Tokenizer.from_pretrained(model_name))

generator = load_text_generator_model()

#generator = pipeline('text-generation', model='gpt2')
def generate_response(category, question, max_length=100):
    prompt = f"{category.upper()} INTERVIEW\nQ: {question}\nA:"
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]['generated_text'].split('A:')[-1].strip()


st.title("🎤 Transcript Classifier with BERT and Text Generation")
st.write("""
This app allows you to:
- Classify interview transcripts into categories.
- Generate AI-based answers for a selected interview category and question.
""")

# Set up Streamlit UI
st.title("🎤 Transcript Classifier with BERT")
transcript = st.text_area("Paste interview transcript here:")

if st.button("Classify Transcript"):
    if transcript.strip():
        # Make prediction
        result = pipeline(transcript)[0]
        
        # Extract predicted label and score
        predicted_label = result['label']  # e.g., 'LABEL_1'
        confidence_score = result['score']
        
        # Extract the numeric part of the label (e.g., 'LABEL_1' -> 1)
        predicted_label_int = int(predicted_label.split('_')[-1]) + 1  # Add +1 to the label
        
        # Show the updated predicted label with confidence
        st.success(f"Predicted Label: {predicted_label_int} (Confidence: {confidence_score:.2f})")
    else:
        st.warning("Please enter a transcript.")


# 2. Question-and-Answer Generator
st.header("🧠 Interview Q&A Generator")

categories = ["Cricket", "Football", "Tennis", "Basketball"] 
selected_category = st.selectbox("Select interview category:", categories)
question = st.text_input("Enter your interview question:")

if st.button("Generate Answer"):
    if question.strip():
        answer = generate_response(selected_category, question)
        st.success(f"**AI Response:** {answer}")
    else:
        st.warning("Please enter a question.")
