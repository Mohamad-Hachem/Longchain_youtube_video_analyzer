import streamlit as st
import langchain_helper as lch
import textwrap

st.title('Youtube Video Analyzer')

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="What is the youtube video url?",
            max_chars=50
        )
        query = st.sidebar.text_area(
            label="ask me anything about the video",
            max_chars=100,
            key="query"
        )

        submit_button = st.form_submit_button(label='Submit')

if query and youtube_url:
    db = lch.create_vector_db_from_youtube_url(youtube_url)
    response = lch.get_response_from_query(db, query, 4)
    st.subheader("Answer:")
    st.text(response['answer'])

    #st.text(textwrap.fill(response, width=80))
