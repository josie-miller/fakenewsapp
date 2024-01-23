#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 08:52:24 2024

@author: josephinemiller
"""

import streamlit as st

def main():
    # Initialize language in session state if not already set
    if 'language' not in st.session_state:
        st.session_state['language'] = 'English'  # Default language
    
    # Language selection in sidebar
    lang = st.sidebar.selectbox("Choose your language（选择您的语言）", ["English", "中文"], index=0 if st.session_state['language'] == 'English' else 1)
    st.session_state['language'] = 'English' if lang == "English" else '中文'
    
    if st.session_state['language'] == 'English':

        st.title("Misinformation Detector")
        # Define your questions and answers
        q_and_a = {
            "What is Misinformation?": "Misinformation is information that is false, inaccurate, or misleading, but is often shared without any intention to deceive. It can arise from misunderstandings, errors in reporting, or the misinterpretation of facts.",
            "Why is it important to be aware of misinformation?": "Learning about misinformation is crucial because it helps develop critical thinking and media literacy skills. By understanding how misinformation spreads and its impact, we can better evaluate the accuracy of information we encounter, make informed decisions, and contribute to a more truthful and informed society. ",
            "How does the analysis work?": "Our analysis utilizes a Bidirectional LSTM neural network model designed for classifying news. This model is trained using the WELFake dataset from Kaggle. When processing text, it first undergoes NLP preprocessing steps before the model predicts a numerical value. If the text receives a 'fake' rating exceeding 50%, it is categorized as 'unreliable.'",
            "How accurate is the analysis?": "The accuracy of our model largely depends on the nature of the text being analyzed. Primarily trained on formal news and online articles, its performance may diminish with more informal text. The inherent complexity of neural networks makes it challenging to pinpoint exactly what influences the model's decisions. Consequently, its sensitivity to different text types, apart from news and online articles, is somewhat uncertain.",
            "Who are we?": "We are Joe, Josh, Josie, Katie, and Hebe, a team of five high school students from Avenues the World School. As part of our J-Term project, we have committed ourselves to raising awareness about online misinformation. This initiative represents our collective effort in this endeavor.",
        }
    
        questions = list(q_and_a.keys())
        for i in range(0, len(questions), 3):  # Loop to create rows
            row_questions = questions[i:i+3]
            cols = st.columns(3)
            for col, question in zip(cols, row_questions):
                with col:
                    st.write(question)  # Display the question
                    st.write(q_and_a[question]) 
    elif st.session_state['language'] == '中文':
        st.title("不实信息检测器")
        # Define your questions and answers
        q_and_a = {
            "什么是不实信息？": "不实信息指的是错误，不准确或具有误导性的信息，虽然它们的欺骗性通常是在人们无意识的状态下造成的。这些信息可以来自误解，报道性错误，或是对于事实的错误解读。",
            "为什么对不实信息的警觉非常重要？": "研究不实信息是非常重要的，因为这可以帮助人们提升它们的批判性思维和媒体识读能力（认识社交媒体运作方式的能力）。通过了解不实信息的传播方式及其影响，我们可以更好的评估我们遇到的信息的精准度，做出有依据的决策，并且为创造一个更加真实和理智的社会做出贡献。",
            "我们的分析方式是什么？": "我们的分析利用了一个为分类新闻而设计的双向LSTM神经网络模型，这个模型使用来自kaggle的WELFake数据集进行训练。在处理文本时，它首先经过自然语言处理（NLP)的预处理步骤，然后模型将预测一个数值。如果文本获得的“假”评级超过50%，则被归类为“不可靠”。",
            "我们的分析有多精准？": "我们模型的准确性在很大程度上取决于被分析文本的性质。由于主要针对正式新闻和在线文章进行训练，对于更非正式的文本，其性能可能会降低。神经网络的内在复杂性使对于影响模型决策的因素的识别变得困难，其性能可能会降低。",
            "我们是谁": "我们是Joe, Josh, Josie, Katie, 和Hebe, 一个由五名爱文世界学校的高中生组成的小组。作为我们的J-Term项目的一部分，我们致力于唤起人们对网上不实信息的警觉的工作。这项工作包含了我们小组共同的努力与心血。"
        }
    
        # Create a 2x3 grid for questions and answers
        questions = list(q_and_a.keys())
        for i in range(0, len(questions), 3):  # Loop to create rows
            row_questions = questions[i:i+3]
            cols = st.columns(3)
            for col, question in zip(cols, row_questions):
                with col:
                    st.text(question)  # Display the question
                    st.write(q_and_a[question])  # Display the answer


if __name__ == "__main__":
    main()
