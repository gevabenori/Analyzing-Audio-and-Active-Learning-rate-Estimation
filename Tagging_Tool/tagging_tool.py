import pandas as pd
import streamlit as st

st.set_page_config(page_icon="⚙️", page_title="Tagging Tool")


def main():
    st.sidebar.radio("Choose who is the king", ('BROS', 'YUVI'))
    st.title("Video and audio tagging tool")
    st.warning("TEST!!!")


if __name__ == "__main__":
    main()
