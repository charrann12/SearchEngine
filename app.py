import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun ## final one is to search from browser
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
import os 
from dotenv import load_dotenv

#Tools(Arxiv and Wikipedia)
arxiv_api_wrapper = ArxivAPIWrapper(tok_k_results = 1,doc_content_chars_max = 250)
arxiv = ArxivQueryRun(api_wrapper = arxiv_api_wrapper)

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results = 1,doc_content_chars_max = 250)
wiki = WikipediaQueryRun(api_wrapper = wiki_api_wrapper)

search = DuckDuckGoSearchRun(name = 'Search')


st.title("Langchain - Chat with search")

## Sidebar settings 
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API Key here", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assisstant", "content":"Hi I am a Chatbot who can search the web. How can i help you ?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder = "What is Machine Learning"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key = api_key, model_name = "meta-llama/llama-4-scout-17b-16e-instruct", streaming = True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools, llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors = True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts = False)
        response = search_agent.run(st.session_state.messages, callbacks = [st_cb])
        st.session_state.messages.append({"role":"assisstant", "content":response})
        st.write(response)
