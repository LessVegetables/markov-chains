import streamlit as st
from markov_chain.markov_core import MarkovTextGenerator
from markov_chain.reader import read_texts_from_folder


default = "И он начал ему объяснять"
userInput = st.text_input("label", value=default)


### this must not be here
from pathlib import Path
#
data_folder = Path(__file__).resolve().parents[1] / "data" / "processed"
included_files = [
    "turgenev_mumu.txt",
    "turgenev_dvoryanskoe_gnezdo.txt",
    #"turgenev_nov.txt",
]
#
text = ""
for file_name in included_files:
    text += read_texts_from_folder(str(data_folder / file_name))
#
#
### remove this once read_texts_from_folder becomes part of the MarkovTextGenerator.__init__()

if st.button("Generate"):
    generator = MarkovTextGenerator(order=len(userInput.split())) ### this also must not need to have `order=...`
    generator.train_from_text(text) ### rm this too
    output = generator.generate(start_state=(userInput.split())) ### this needs to be a simple `start_state=userInput`

    st.write(output)