# Next Word LSTM RNN

This project predicts the next word in Shakespearean text using an LSTM RNN.

## Folder Structure

```
LSTM_RNN/
├── app.py
├── lstm_rnn.ipynb
├── next_word_lstm.h5
├── requirements.txt
├── tokenizer.pickle
├── data/
    └── hamlet.txt
```

## Setup

1. **Clone the Repository**
   ```sh
   git clone https://github.com/ayush-patel-29/next_word_lstm_rnn.git
   cd next_word_lstm_rnn
   ```

2. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**
   ```sh
   streamlit run app.py
   ```