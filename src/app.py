import streamlit as st
from markov_chain.markov_core import MarkovTextGenerator

default = "И он начал" #ему объяснять
userInput = st.text_input("Start of Markov Chain", value=default, label_visibility="hidden")

if st.button("Generate"):

    generator = MarkovTextGenerator(order=len(userInput.split()))
    generator.load_default_dataset()
    output = generator.generate_from_seed(userInput)

    st.write(output)