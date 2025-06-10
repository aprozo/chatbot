import streamlit as st
import os

# CRITICAL: Set environment variables BEFORE importing any ML libraries
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import torch early and set to CPU mode
try:
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(-1)  # Disable CUDA
except ImportError:
    pass

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="STAR chat", page_icon=":star:")
st.image("STAR-logo-trans.gif")
st.title("STAR chat")

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

@st.cache_resource
def get_embedding_function():
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_function


embedding_function = get_embedding_function()
text_splitter  = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

with st.sidebar:
    # choose on number of top documents to retrieve as a slider
    database = st.radio(
            "Choose Database",
            [ "Pinecone (Cloud)","Chroma (Local)"],
            key="visibility",
            disabled=True,
            horizontal=True,
        )
    useParentDocument = False
    if database == "Chroma (Local)":
        useParentDocument = st.checkbox("Use Parent document retriever ( top 5 full documents are passed to LLM instead of chunks - use when a specific topic is needed in detail)", value=False)

    if useParentDocument:
        top_k = 5
    else:
        top_k = st.slider("Number of chunks to retrieve", min_value=10, max_value=200,  value=100, step=5)


    search_type = st.selectbox(
    'Data search type:',
    ('Everything', 'arxiv', 'theses', 'starnotes','mattermost'),
    )
    
    if search_type=='Everything':
        search_type = None

    filter_dict={}
    if search_type:
        filter_dict = {"type": search_type}


if useParentDocument:
    database="Chroma (Local)"
    
index_name="general"
try:
    pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
    index = pc.Index(index_name)
except Exception as e:
    st.error(f"Failed to initialize Pinecone index: {e}")
    st.stop()



if database == "Chroma (Local)":
    st.warning("switch to cloud")
else:
    vectorestore = PineconeVectorStore(index_name=index_name, embedding=embedding_function, pinecone_api_key = st.secrets["pinecone_api_key"])
    retriever = vectorestore.as_retriever(search_kwargs={
            'k': top_k,
            'filter': filter_dict
                                        })    

openai_api_key=st.secrets["openai_api_key"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["langsmith_api_key"]
os.environ["LANGSMITH_TRACING"] = "true"

llm = ChatOpenAI(model_name="gpt-4o-mini", streaming=True, openai_api_key=openai_api_key)

system_prompt  = """
You are an expert on the STAR experiment, a high-energy nuclear physics experiment at the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory. \
Your task is to answer questions specifically related to the STAR experiment, its software, technologies, and related topics.  \
Refrain any other topics by saying you will not answer questions about them and Exit right away here. DO NOT PROCEED. \
You are not allowed to use any other sources other than the provided search results and chat history. \

Generate a comprehensive, and informative answer strictly within 200 words or less for the \
given question based solely on the provided search results (urls and content) and chat history. You must \
only use information from the provided search results and chat history. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. You should use bullet points in your answer for readability. Make sure to break down your answer into bullet points.\
You should not hallicunate nor build up any references, do not use any text within html block <url> and </url> except when citing in the end.  \
Make sure not to repeat the same context. Be specific to the exact questions. Take you time.\

Here is the response template:
---
# Response template 

- Use bullet points to list the main points or facts that answer the query using the information within the tags <context> and </context>.  
- After answering, analyze the respective source links provided within <url> and </url> and keep only the unique links for the next step. Try to minimize the total number of unique links with no more than 10 unique links for the answer.
- You will strictly use no more than 10 most unique links for the answer.
- Use bulleted list of superscript numbers within square brackets to cite the sources for each point or fact. The numbers should correspond to the order of the sources which will be provided in the end of this reponse.
- End with a closing remark and a list of sources with their respective URLs as a bullet list explicitly with full links which are enclosed between <url> and </url>.\
---
Here is how an response would look like. Reproduce the same format for your response:
---
# Example response

Hello, here are some key points:

- The STAR (Solenoidal Tracker at RHIC) experiment is a major high-energy nuclear physics experiment conducted at the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory[^1].
- The primary research goal of STAR is to study the properties of the quark-gluon plasma (QGP), a state of matter thought to have existed just after the Big Bang, by colliding heavy ions at nearly the speed of light[^2].
- STAR utilizes a variety of advanced detectors to measure the thousands of particles produced in these collisions, including the Time Projection Chamber (TPC), the Barrel Electromagnetic Calorimeter (BEMC), and the Muon Telescope Detector (MTD)[^3].
- Key findings from STAR include evidence for the QGP's near-perfect fluidity, the discovery of the "chiral magnetic effect," and insights into the spin structure of protons[^1].

[^1]: Three Particle Correlations from STAR https://arxiv.org/abs/0704.0220v1
[^2]: Description of Reaction Plane Correlated Triangular Flow in Au+Au Collisions with the STAR Detector at RHIC https://drupal.star.bnl.gov/STAR/files/CRacz_Dissertation_v5.pdf
[^3]: Fluctuations of charge separation perpendicular to the event plane and local parity violation in sqrt sNN =200 GeV Au+Au collisions at RHIC https://arxiv.org/abs/1302.3802v3
---

Where each of the references is taken from the corresponding <url> html block in the context. \
Strictly do not repeat the same link. Use numbers to cite the sources. If it happens that a link has already been used, just use previously used reference in the text.\

Don't try to make up an answer.\
Make sure to highlight the most important key words in bold font. Don't repeat any context nor points in the answer.\
You not sure anout the answer, just say that "I am not surem but here is the closest retrieved context:".\
You may provide first 5 relevant context sources.\
Anything between the following `context`  html blocks is retrieved from a knowledge \
bank. The context is numbered based on its knowledge retrival and increasing cosine similarity index. \
Make sure to consider the order in which context appear. It is an increasing order of cosine similarity index.\
The contents are formatted in latex or markdown, you need to remove any special characters and latex formatting before cohercing the points to build your answer.\
You can use latex commands if necessary.\
You will strictly cite no more than 10 unqiue citations at maximum from the context below.\
Make sure these citations have to be relavant and strictly do not repeat the context in the answer.

<context>
    {context}
</context>

REMEMBER: If there is no relevant information within the context or chat history, just say "Hmm, I'm \
not sure." or greet back. Don't try to make up an answer.\
Question: {input}
"""
### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. This question will be retrieved "
    "using an embedding database for RAG. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
contextualize_q_llm = llm.with_config(tags=["contextualize_q_llm"])

history_aware_retriever = create_history_aware_retriever(
    contextualize_q_llm, retriever, contextualize_q_prompt
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
custom_document_prompt = PromptTemplate(
        input_variables=["page_content","url", "type"],  # Replace "your_variable_name" with your actual variable name
        template="type:{type} <url>{url}</url>\n Text:{page_content}"  # Customize your template
    )
question_answer_chain = create_stuff_documents_chain(
                                                    llm, qa_prompt,
                                                    document_prompt = custom_document_prompt
                                                    )

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hi there, you can ask any questions regarding STAR experiment and its infrastructure, any other questions will not be answered :)")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if input_question := st.chat_input("Ask a question, e.g. What are TPC operational principles?.."):
    st.chat_message("human").write(input_question)
    config = {"configurable": {"session_id": "any"}}
    # Note: new messages are saved to history automatically by Langchain during run
    with st.spinner("Retrieving information..."):
            response = conversational_rag_chain.invoke({"input": input_question}, config)
    st.chat_message("ai").write(response["answer"])
