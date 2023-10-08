from model import LlamaModel, build_inputs
import streamlit as st
from streamlit_chat import message
import argparse

@st.cache_resource
def create_model():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument(
        '-m',
        '--model_id',
        required=True,
        type=str,
        help='local model path or remote model id'
    )
    args = parser.parse_args()
    return LlamaModel(args.model_id)

with st.spinner("Loading models"):
    ov_llama = create_model()

if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("## Configuration")

    max_tokens = st.number_input("max_tokens",
                                 min_value=1,
                                 max_value=500,
                                 value=200)
    temperature = st.number_input("temperature",
                                  min_value=0.1,
                                  max_value=4.0,
                                  value=0.8)
    top_p = st.number_input("top_p", min_value=0.1, max_value=1.0, value=0.8)
    top_k = st.number_input("top_k", min_value=1, max_value=500, value=20)

    if st.button("Reset the chat"):
        st.session_state.message = ""
        st.session_state.history = []

st.markdown("## OpenVINO Chatbot based on Llama2")

history: list[tuple[str, str]] = st.session_state.history

if len(history) == 0:
    st.caption("Please enter your question")

for idx, (question, answer) in enumerate(history):
    message(question, is_user=True, key=f"history_question_{idx}")
    st.write(answer)
    st.markdown("---")

next_answer = st.container()

question = st.text_area(label="Message", key="message")

if st.button("Send") and len(question.strip()):
    with next_answer:
        message(question, is_user=True, key="message_question")
        with st.spinner("Preparing the response"):
            with st.empty():
                prompt = build_inputs(history, question)
                for answer in ov_llama.generate_iterate(
                        prompt,
                        max_generated_tokens=max_tokens,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                ):
                    st.write(answer)
        st.markdown("---")

    st.session_state.history = history + [(question, answer)]