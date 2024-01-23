
import numpy as np # la
import pandas as pd #dp
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
import re
from nltk.corpus import stopwords
import tensorflow as tf
from nltk.stem.porter import PorterStemmer
import streamlit as st
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM,Bidirectional
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint



ps = PorterStemmer()

corpus = []
vocab_size = 10000
sentence_length = 50
embedding_vector_features=150


model_file = 'jsw3rel.h5'
if os.path.exists(model_file):
    model2 = load_model(model_file)
else:
    df = pd.read_csv('/Users/josephinemiller/Desktop/FHS_model/WELFake.csv')
    df = df.dropna()
    df.drop(columns=['Unnamed: 0'],inplace=True)
    X = df.drop(columns=['label'])
    y = df['label']

    messages = X.copy()
    messages.reset_index(inplace=True)
    nltk.download('stopwords')


    for i in range(0, len(messages)):
        
        # We are substituting everything apart from (a-z, A-Z) with a " " (space)
        review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
        
        review = review.lower()
        review = review.split()
        
        # if a word is not in Stop Words,then only we will add it to review (list/array)
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

    max_length = max(len(sentence.split()) for sentence in corpus)

    print("Maximum sentence length:", max_length)

    onehot_repr=[one_hot(words,vocab_size) for words in corpus] 

    embedded_docs = pad_sequences(onehot_repr,padding='pre',maxlen=sentence_length)


    model2 = Sequential()
    model2.add(Embedding(vocab_size, embedding_vector_features, input_length=sentence_length))
    model2.add(Bidirectional(LSTM(200))) 
    model2.add(Dropout(0.2))
    model2.add(Dense(1,activation='sigmoid'))
    model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    checkpoint_path = "training_1/cp.ckpt"
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    X_final = np.array(embedded_docs)
    y_final = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
    model2.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3, batch_size=120)
    model2.save(model_file)

def classify_text(input_text):
    # Preprocessing steps
    review = re.sub('[^a-zA-Z]', ' ', input_text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)

    # One hot representation and padding
    onehot_repr = one_hot(review, vocab_size)
    padded_sentence = pad_sequences([onehot_repr], padding='pre', maxlen=sentence_length)

    # Predicting the class
    prediction = model2.predict(padded_sentence)
    return (prediction > 0.5).astype(int), prediction

def main():

    language = st.sidebar.selectbox("Choose your language（选择您的语言）", ["English", "中文"], index=0 if st.session_state['language'] == 'English' else 1)
    st.session_state['language'] = language
    
    if st.session_state['language'] == 'English':
        st.title("Misinformation Analyzer")
    
        # User input text box
        user_input = st.text_area("Enter the article title to check if it's real or fake, remember this model is not fully accurate and should not be used as the only fake-article-checking method", "")
    
        # Button to classify the text
        if st.button("Classify"):
            if user_input:
                # Classify the input text
                classification,prediction = classify_text(user_input)
                slay = prediction*100
                slay1 = "{:.2f}".format(slay[0][0])
                slay_str=str(slay1)
                
                
                # Display the result
                if classification == 1:
                    st.warning("May be unreliable!")
                    st.warning(f'News is {slay_str}% Fake!')
                elif classification == 0:
                    st.success("Most likely reliable!")
                    st.success(f'News is {slay_str}% Fake!')
                else:
                    st.error("Error in classification")
            else:
                st.error("Please enter some text to classify")
    elif st.session_state['language'] == '中文':
        st.title("不实信息分析器")
    
        # User input text box
        user_input = st.text_area("输入英文新闻内容以检测其真伪（请记住这个模型的检测并不确保100%的精准度，并且不应该被用作是唯一的虚假文章的检测方式）。", "")
    
        # Button to classify the text
        if st.button("分类"):
            if user_input:
                # Classify the input text
                classification,prediction = classify_text(user_input)
                slay = prediction*100
                slay1 = "{:.2f}".format(slay[0][0])
                slay_str=str(slay1)
                
                
                # Display the result
                if classification == 1:
                    st.warning("分析结果可能是文章不可靠!")
                    st.warning(f'{slay_str}%的新闻内容是虚假的!')
                elif classification == 0:
                    st.success("分析结果可能是文章可靠!")
                    st.success(f'{slay_str}%的新闻内容是虚假的!')
                else:
                    st.error("分类错误")
            else:
                st.error("请输入文字并分类")
        

# Run the app
if __name__ == '__main__':
    main()
    
