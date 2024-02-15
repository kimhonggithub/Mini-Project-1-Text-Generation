
import streamlit as st
from backoff_model import  build_language_model,generate_text_backoff
from interpolation_model import build_interpolation_language_model, lamdas_and_k,generate_text_iplt
from data_pro import read_corpus,split_corpus_to_token

# Streamlit UI
st.title('Language Model Selection')

# Input text area for user to input seed text
seed_text = st.text_area('Enter Seed Text:', '')

# Select language model type
model_type = st.radio("Select Language Model:", ('LM1 (Backoff Method)', 'LM2 (Interpolation Method)'))

# Button to generate text based on input
if st.button('Generate Text'):
    # Read and preprocess corpus from file
    corpus = read_corpus('khmer_food.txt')  # Read your corpus from file
    train_tokens, val_tokens = split_corpus_to_token(corpus)
    
    vocab_size = 5000
    vocab = set(train_tokens)
    
    # Determine which language model to build
    if model_type == 'LM1 (Backoff Method)':
        # Build LM1 language model (backoff method)
        model = build_language_model(train_tokens)

        # Generate text based on input
        generated_text = generate_text_backoff(model, seed_text, length=100,vocab=vocab)
    else:
        # Build LM2 language model (interpolation method)
        # Placeholder for hyperparameter experimentation
      
        lamda_value, k_value = lamdas_and_k(train_tokens,val_tokens,5000)
        model = build_interpolation_language_model(train_tokens, k_value, val_tokens)

        # Generate text based on input
        
        generated_text = generate_text_iplt(model, seed_text, length=100,vocab=vocab)

 
    # Display generated text
    st.write('Generated Text:')
    st.write(generated_text)
  

