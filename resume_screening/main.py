# main - By: limon - Sun Jun 2 2024

import streamlit as st
from utils import *
import uuid
from IPython.display import display


def main():
    st.set_page_config(page_title="Resume Screening")
    st.title("Resume Screening")
    
    job_description = st.text_area("Please write your desired job description to help find the best candidate...", key="1")
    pdf = st.file_uploader("Upload resumes, please upload only PDF files", type=["pdf"], accept_multiple_files=True)

    submit = st.button("Analyze resume and give proper feedback")

    if submit:
        with st.spinner('Please Wait...'):
            st.session_state['unique_id'] = uuid.uuid4().hex

            model = init_model()
            final_docs_list = make_docs(pdf, st.session_state['unique_id'])

            st.write("*Resumes uploaded* :" + str(len(final_docs_list)))

            summaries = []

            for doc in final_docs_list:
                summary = generate_summary(job_description, doc, model)
                query_embedding = embed_content(summary)
                summary_embedding = embed_content(summary)


                st.markdown(f"<h4 style='color: orange;'>{doc.metadata['name']}</h4>", unsafe_allow_html=True)
                st.write(summary)
                st.write("\n\n") 

if __name__ == "__main__":
    main()
