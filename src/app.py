import streamlit as st
from markov_chain.markov_core import MarkovTextGenerator

st.html("""
    <script defer src="https://stat.gehrman.me/script.js" data-website-id="025ade9a-2a56-494f-a07a-21d6e4b07019"></script>
""")

default = "И он начал" #ему объяснять
userInput = st.text_input("Start of Markov Chain", value=default, label_visibility="hidden")

if st.button("Generate"):
    st.html('<script>umami.track("generate-clicked")</script>')
    
    generator = MarkovTextGenerator(order=len(userInput.split()))
    generator.load_default_dataset()
    output = generator.generate_text(userInput)
    if generator.used_random_start:
        st.warning("Такого старта нет в датасете, сгенерировали случайный текст.")

    st.write(output)
