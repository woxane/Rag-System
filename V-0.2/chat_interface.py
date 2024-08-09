import streamlit as st

class ChatInterface:
    def __init__(self, document_processor, vectorizer, milvus_handler, chatbot):
        self.document_processor = document_processor
        self.vectorizer = vectorizer
        self.milvus_handler = milvus_handler
        self.chatbot = chatbot

    def display_chat(self, messages):
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    def respond(self, user_input):
        query_vector = self.vectorizer.vectorize([user_input])
        search_results = self.milvus_handler.search_vectors(query_vector)
        relevant_texts = [res['entity'].get("text") for res in search_results[0]]
        context = "\n\n".join(relevant_texts)
        prompt = self.chatbot.create_prompt(context, user_input)
        st.session_state.messages.append({'role': 'user', 'content': user_input, 'rag_prompt': prompt})


    def run(self):
        st.title("PDF Helper")

        # Initialize session state for pdf_texts and messages if not already present
        if "files_id" not in st.session_state:
            st.session_state.files_id= []

        if "messages" not in st.session_state:
            st.session_state.messages = []

        with st.sidebar:
            st.header("Upload PDF Files")
            uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
            uploaded_ids = [file.file_id for file in uploaded_files]

            if uploaded_files:
                for id in st.session_state.files_id:
                    if id not in uploaded_ids:
                        #This file has deleted
                        #TODO: create delete rows with group_id
                        st.session_state.files_id.remove(id)

                for file in uploaded_files:
                    if file.file_id not in st.session_state.files_id:
                        #New file uploaded
                        chunks = self.document_processor.load_pdf(file)
                        vectors = self.vectorizer.vectorize(chunks)
                        self.milvus_handler.save_vectors(vectors, chunks)

                        st.success("PDF files uploaded successfully!")
            else:
                #Delete last remaining id
                #TODO: create delete rows with group_id
                st.session_state.files_id = []

        self.display_chat(st.session_state.messages)

        # Accept user input
        if user_input := st.chat_input("Type your message here ..."):
            self.respond(user_input)

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                messages = [{"role": message["role"], "content": message["rag_prompt"]
                if message["role"] == "user" else message["content"]} for message in st.session_state.messages]
                completion = self.chatbot.get_response(messages)

                for response in completion:
                    if response.choices[0].delta.content:
                        full_response += response.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)


            st.session_state.messages.append({'role': 'assistant', 'content': full_response})