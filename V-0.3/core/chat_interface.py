import streamlit as st
from chatbot import Chatbot

dotenv_path = ".env"


class ChatInterface:
    def __init__(self):
        self.chatbot = Chatbot()

    def display_chat(self, messages):
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def run(self):
        st.title("PDF Helper")

        # Initialize session state for pdf_texts and messages if not already present
        if "files_id" not in st.session_state:
            st.session_state.files_id = []

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "file_names" not in st.session_state:
            st.session_state.file_names = []

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
                chat_history = '\n'.join([f"{message['role']}:{message['content']}" for
                                         message in st.session_state.messages])
                completion = self.chatbot.get_response(query=user_input, history=chat_history, stream=False)

                for response in completion:
                    full_response += response
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

            st.session_state.messages.append({'role': 'user', 'content': user_input})
            st.session_state.messages.append({'role': 'assistant', 'content': full_response})


if __name__ == "__main__":
    chat_interface = ChatInterface()
    chat_interface.run()
