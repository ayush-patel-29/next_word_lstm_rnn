import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Load the trained model
model = load_model('next_word_lstm.h5')
# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
max_sequence_len = 14 # This should match the max_sequence_len used during training

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Next Word Prediction using LSTM RNN")
user_input = st.text_input("Enter a sequence of words:")
if user_input:
    next_word = predict_next_word(model, tokenizer, user_input, max_sequence_len)
    st.write(f"Predicted next word: {next_word}")