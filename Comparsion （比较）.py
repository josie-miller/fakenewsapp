#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 08:46:17 2024

@author: josephinemiller
"""
import streamlit as st
from nltk.corpus import stopwords
from collections import Counter
    
def compare_texts(text1, text2):
    # Convert texts to sets of words, excluding stopwords
    stop_words = set(stopwords.words('english'))
    words1 = {word.lower() for word in text1.split() if word.lower() not in stop_words}
    words2 = {word.lower() for word in text2.split() if word.lower() not in stop_words}

    # Find common words
    common_words = words1.intersection(words2)
    return common_words


def main():
   
    # Language selection widget
    language = st.sidebar.selectbox("Choose your language（选择您的语言）", ["English", "中文"], index=0 if st.session_state['language'] == 'English' else 1)
    st.session_state['language'] = language
    
    if st.session_state['language'] == 'English':
        st.title("Article Comparison")
        text35 = st.write("This system will show all similar words between two articles.", "")

        # Text input
        text1 = st.text_area("Enter first text", "")
        text2 = st.text_area("Enter second text", "")

        # Button to start comparison
        if st.button("Compare Texts"):
            if text1 and text2:
                # Function to process and compare texts
                similar_words = compare_texts(text1, text2)
                st.write("Similar words:")
                similar_words_str = ', '.join(similar_words)
                st.write(similar_words_str)
            else:
                st.error("Please enter text in both fields")
        
    elif st.session_state['language'] == '中文':
        st.title("文本比较")
        text35 = st.write("此系统将显示两篇文章之间的所有相似词语。","")
        # Text input
        text1 = st.text_area("输入第一段英文文本", "")
        text2 = st.text_area("输入第二段英文文本", "")

        # Button to start comparison
        if st.button("比较文本"):
            if text1 and text2:
                # Function to process and compare texts
                similar_words = compare_texts(text1, text2)
                st.write("相似的词语:")
                similar_words_str = ', '.join(similar_words)
                st.write(similar_words_str)
            else:
                st.error("请在两个字段中输入文本")
        
if __name__ == "__main__":
    main()
