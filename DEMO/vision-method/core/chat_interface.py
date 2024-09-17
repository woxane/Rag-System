import streamlit as st
from chatbot import Chatbot
import re

dotenv_path = ".env"


class ChatInterface:
    def __init__(self):
        self.chatbot = Chatbot()

    def display_chat(self, messages):
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

    def run(self):
        st.title("PDF Helper")

        # Initialize session state for pdf_texts and messages if not already present
        if "files_id" not in st.session_state:
            st.session_state.files_id = []

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "file_names" not in st.session_state:
            st.session_state.file_names = []

        # Custom CSS for hover effect
        st.markdown(
            """
            <style>
            /* General styles for the hover container */
            .hover-container {
                display: inline-block;
                position: relative;
                color: #1f77b4; /* Streamlit's blue color */
                font-weight: bold;
            }

            /* Styles for the hover content */
            .hover-content {
                visibility: hidden;
                background-color: #f0f2f6; /* Light grey background to match Streamlit theme */
                color: #333; /* Dark text color for readability */
                text-align: left; /* Default alignment for LTR text */
                border-radius: 4px; /* Rectangular corners */
                padding: 10px;
                position: absolute;
                z-index: 1;
                top: 30px;
                left: 0;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Slight shadow for depth */
                min-width: 400px; /* Minimum width to ensure it is wider */
                max-width: 600px; /* Maximum width to keep it within reasonable bounds */
                width: auto; /* Let width adapt based on content and constraints */
                direction: ltr; /* Default direction for LTR languages */
            }

            /* Adjust hover content based on text direction */
            .hover-container[dir="rtl"] .hover-content {
                direction: rtl; /* Text direction for RTL languages */
                text-align: right; /* Align text to the right for RTL languages */
                left: auto; /* Position from the right for RTL languages */
                right: 0; /* Align the content to the right edge */
            }

            /* Show the hover content on hover */
            .hover-container:hover .hover-content {
                visibility: visible;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        with st.sidebar:
            st.header("Upload PDF Files")
            current_files = []
            uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
            uploaded_ids = [file.file_id for file in uploaded_files]

            if uploaded_files:
                for file_id in st.session_state.files_id:
                    if file_id not in uploaded_ids:
                        # This file has deleted
                        self.chatbot.delete_pdf(file_id=file_id)
                        st.session_state.files_id.remove(file_id)

                for file in uploaded_files:
                    current_files.append(file.name)
                    if file.file_id not in st.session_state.files_id:
                        # New file uploaded
                        self.chatbot.save_pdf(file)
                        st.session_state.files_id.append(file.file_id)
                        st.session_state.file_names.append(file.name)

                        st.success("PDF files uploaded successfully!")
            else:
                # Delete last remaining id
                for file_id in st.session_state.files_id:
                    self.chatbot.delete_pdf(file_id=file_id)

                st.session_state.files_id = []

            st.write("File status:")
            for file_name in st.session_state.file_names:
                if uploaded_files and file_name in current_files:
                    st.write(f"{file_name}")
                else:
                    st.write(f"{file_name} - Removed")

        self.display_chat(st.session_state.messages)

        # Accept user input
        if user_input := st.chat_input("Type your message here ..."):

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                completion = self.chatbot.get_response(query=user_input, history=st.session_state.messages, stream=False)

                for response in completion:
                    full_response += response
                    message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)

                message_placeholder.markdown(full_response, unsafe_allow_html=True)

                references = self.chatbot.get_formatted_references()

                for idx, reference in enumerate(references, 1):
                    full_response += '<div class="hover-container">\n' \
                                     f"   <b>Reference {idx}</b>\n" \
                                     '   <div class="hover-content">\n' \
                                     f"        {reference}" \
                                     "   </div>\n" \
                                     "</div>\n"

                    message_placeholder.markdown(full_response, unsafe_allow_html=True)

            st.session_state.messages.append({'role': 'user', 'content': user_input})
            st.session_state.messages.append({'role': 'assistant', 'content': full_response})


if __name__ == "__main__":
    chat_interface = ChatInterface()
    chat_interface.run()
