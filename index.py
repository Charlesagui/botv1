import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Initialize the chat model
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)

# Define a prompt template for the chat
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name="messages")
])

# Combine the prompt template with the chat model
chain = prompt_template | chat_model

# Initialize the in-memory chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = ChatMessageHistory()

def send_message():
    user_input = st.session_state.user_input
    if user_input:
        # Add user message to history
        st.session_state['chat_history'].add_user_message(user_input)
        
        # Invoke the chat model
        response = chain.invoke({"messages": st.session_state['chat_history'].messages})

        # Extract content from AIMessage
        if isinstance(response, AIMessage):
            bot_response = response.content
        else:
            bot_response = "Sorry, I didn't understand that."

        # Add AI response to history
        st.session_state['chat_history'].add_ai_message(bot_response)

        # Clear the input box after sending the message
        st.session_state.user_input = ""

def main():
    # Title and Header
    st.title("Intelligent Chatbot with LangChain and OpenAI")
    st.subheader("This chatbot uses LangChain and OpenAI's GPT to answer your questions.")

    # Display the conversation history
    if st.session_state['chat_history'].messages:
        st.text_area("Conversation History", value="\n".join(
            message.content for message in st.session_state['chat_history'].messages), height=250, key='history')

    # User input
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    # Ensure that the text_input widget takes the empty string when cleared
    user_input = st.text_input("Type your question or comment and press enter:", 
                               key="user_input", 
                               on_change=send_message, 
                               value=st.session_state.user_input)

    # Button to manually send the message
    if st.button("Send"):
        send_message()

if __name__ == "__main__":
    main()