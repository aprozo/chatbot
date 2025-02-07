import chainlit as cl
import pinecone
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
import os
from chainlit import user_session
from langchain_community.vectorstores import Pinecone
from langchain.memory import ConversationBufferWindowMemory
from pinecone import Pinecone as PineconeClient
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.retrievers import MultiQueryRetriever
from sentence_transformers import CrossEncoder

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import pinecone


# 1. Initialize Pinecone (replace with your actual API key, environment, and index name)
from dotenv import load_dotenv
load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone



# Initialize models
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Initialize Pinecone
# Initialize environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = "general"
pc = PineconeClient(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Initialize Pinecone vector store
index_name = "YOUR_INDEX_NAME"
vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings, pinecone_api_key = st.secrets["pinecone_api_key"])

# Define filter options
FILTER_OPTIONS = ["arxiv", "starnotes", "theses", "mattermost"]


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


Context: {context}
Chat History: {chat_history}

Question: {question}
Answer:
"""


QA_PROMPT = PromptTemplate(template=system_prompt, input_variables=["context", "chat_history", "question"])

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# Streamlit UI
st.title("STAR Experiment RAG Chatbot")
st.sidebar.header("Filters")

# Type filter
selected_types = st.sidebar.multiselect(
    "Filter by content type:",
    options=FILTER_OPTIONS,
    default=FILTER_OPTIONS
)

# Search mode toggle
single_query_mode = st.sidebar.checkbox("Single-Query Search Mode")

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)

# Custom retriever with metadata filtering
def get_retriever():
    return vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"type": {"$in": selected_types}} if selected_types else {}
        }
    )

# Initialize retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=get_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about the STAR experiment:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.spinner("Analyzing STAR documentation..."):
        try:
            if single_query_mode:
                # Direct retrieval mode
                results  = vectorstore.similarity_search_with_score(
                    prompt,
                    k=10,
                    filter={"type": {"$in": selected_types}} if selected_types else {}
                )
                
                response = "\n\n".join([
                    f"Type: ({doc.metadata['type']}): {doc.page_content}\n"
                    f"URL: {doc.metadata.get('url', 'N/A')}\nScore: {score}"
                    for doc, score in results
                ])
            else:
                # Conversational RAG
                result = qa_chain({"question": prompt})
                response = result["answer"]
                
        except Exception as e:
            response = f"Error processing request: {str(e)}"

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


# Reranking section (add your reranking implementation here)
# Example using Cohere reranker:
from langchain.retrievers import CohereRerank

reranker = CohereRerank()
reranked_docs = reranker.rerank(query, docs)

