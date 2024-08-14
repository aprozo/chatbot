import streamlit as st
from langchain.schema import StrOutputParser
from langchain_core.runnables import  RunnablePassthrough
from langchain.schema.runnable import RunnableMap
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


from langchain.retrievers import ParentDocumentRetriever
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

from dotenv import load_dotenv
load_dotenv()
import os

st.set_page_config(page_title="STAR chat", page_icon=":star:")
st.image("STAR-logo-trans.gif")
st.title("STAR chat")
st.write("This is a simple app to demonstrate the STAR chat system")

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
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

if useParentDocument:
    database="Chroma (Local)"


pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
index_name = "arxiv-papers-md"
index = pc.Index(index_name)


if database == "Chroma (Local)":
    st.warning("switch to cloud")
else:
    vectorestore = PineconeVectorStore(index_name=index_name, embedding=embedding_function, pinecone_api_key = st.secrets["pinecone_api_key"])
    retriever = vectorestore.as_retriever(search_kwargs={"k": top_k})    


openai_api_key=st.secrets["openai_api_key"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["langsmith_api_key"]
os.environ["LANGSMITH_TRACING"] = "true"

llm = ChatOpenAI(model_name="gpt-4o-mini", streaming=True, openai_api_key=openai_api_key)

system_prompt  = """
You are an expert on the STAR experiment, a high-energy nuclear physics experiment at the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory. \
Your task is to answer questions specifically related to the STAR experiment, its findings, technologies, and related topics.  \
Refrain any other topics by saying you will not answer questions about them and Exit right away here. DO NOT PROCEED. \
You are not allowed to use any other sources other than the provided search results. \

Generate a comprehensive, and informative answer strictly within 200 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. You should use bullet points in your answer for readability. Make sure to break down your answer into bullet points.\
You should not hallicunate nor build up any references, Use only the `context` html block below and do not use any text within <ARXIV_ID> and </ARXIV_ID> except when citing in the end.  \
Make sure not to repeat the same context. Be specific to the exact question asked for.\

Here is the response template:
---
# Response template 

- Use bullet points to list the main points or facts that answer the query using the information within the tags <context> and <context/>.  
- After answering, analyze the respective source links provided within <ARXIV_ID> and </ARXIV_ID> and keep only the unique links for the next step. Try to minimize the total number of unique links with no more than 10 unique links for the answer.
- You will strictly use no more than 10 most unique links for the answer.
- Use bulleted list of superscript numbers within square brackets to cite the sources for each point or fact. The numbers should correspond to the order of the sources which will be provided in the end of this reponse. Note that for every source, you must provide a URL.
- End with a closing remark and a list of sources with their respective URLs as a bullet list explicitly with full links which are enclosed in the tag <ARXIV_ID> and </ARXIV_ID> respectively.\
---
Here is how an response would look like. Reproduce the same format for your response:
---
# Example response

Hello, here are some key points:

- The STAR (Solenoidal Tracker at RHIC) experiment is a major high-energy nuclear physics experiment conducted at the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory[^1].
- The primary research goal of STAR is to study the properties of the quark-gluon plasma (QGP), a state of matter thought to have existed just after the Big Bang, by colliding heavy ions at nearly the speed of light[^2].
- STAR utilizes a variety of advanced detectors to measure the thousands of particles produced in these collisions, including the Time Projection Chamber (TPC), the Barrel Electromagnetic Calorimeter (BEMC), and the Muon Telescope Detector (MTD)[^3].
- Key findings from STAR include evidence for the QGP's near-perfect fluidity, the discovery of the "chiral magnetic effect," and insights into the spin structure of protons[^4].

[^1]: https://arxiv.org/abs/0704.0220v1
[^2]: https://arxiv.org/abs/nucl-ex/0106003
[^3]: https://arxiv.org/abs/1302.3802v3
[^4]: https://arxiv.org/abs/nucl-ex/0603028
---

Where each of the references are taken from the corresponding <ARXIV_ID> in the context. Strictly do not provide title for the references \
Strictly do not repeat the same links. Use the numbers to cite the sources. \

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." or "This is beyond the chat scope" or greet back or politely refuse to answer. Don't try to make up an answer. Write the answer in the form of markdown bullet points.\
Make sure to highlight the most important key words in bold font. Don't repeat any context nor points in the answer.\

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. The context are numbered based on its knowledge retrival and increasing cosine similarity index. \
Make sure to consider the order in which they appear context appear. It is an increasing order of cosine similarity index.\
The contents are formatted in latex, you need to remove any special characters and latex formatting before cohercing the points to build your answer.\
Write your answer in the form of markdown bullet points. You can use latex commands if necessary.
You will strictly cite no more than 10 unqiue citations at maximum from the context below.\
Make sure these citations have to be relavant and strictly do not repeat the context in the answer.

<context>
    {context}
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." or greet back. Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
Question: {input}
"""
def format_docs(docs):
    return f"\n\n".join(f'{i+1}. ' + doc.page_content.strip("\n") 
                        + f"<ARXIV_ID> {doc.metadata['arxiv_id']} <ARXIV_ID/>" 
                        for i, doc in enumerate(docs))

llm = ChatOpenAI(model_name="gpt-4o-mini", streaming=True, openai_api_key=openai_api_key)

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain


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

from langchain_community.document_transformers import LongContextReorder

# Reorder the documents:
# Less relevant document will be at the middle of the list and more
# relevant elements at beginning / end.
reordering = LongContextReorder()

from langchain_core.prompts import PromptTemplate

# Define your custom document prompt

#  document_prompt: Prompt used for formatting each document into a string. Input
#             variables can be "page_content" or any metadata keys that are in all
#             documents. "page_content" will automatically retrieve the
#             `Document.page_content`, and all other inputs variables will be
#             automatically retrieved from the `Document.metadata` dictionary. Default to
#             a prompt that only contains `Document.page_content`.

custom_document_prompt = PromptTemplate(
    input_variables=["page_content", "title" , "arxiv_id"],  # Replace "your_variable_name" with your actual variable name
    template="Title:{title}, <ARXIV_ID>{arxiv_id}<ARXIV_ID/>\n Text:{page_content}"  # Customize your template
)

question_answer_chain = create_stuff_documents_chain(
                                                    llm, qa_prompt,
                                                    document_prompt = custom_document_prompt)


rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


from langchain_core.runnables.history import RunnableWithMessageHistory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")


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


