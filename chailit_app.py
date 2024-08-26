
import chainlit as cl

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv()


from typing import List
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.docstore.document import Document
import chainlit as cl


welcome_message = "Welcome to the Chainlit Pinecone demo! Ask anything about documents you vectorized and stored in your Pinecone DB."

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter  = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

pinecone_api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=pinecone_api_key)
index_name = "arxiv-papers-md"
index = pc.Index(index_name)


vectorestore = PineconeVectorStore(index_name=index_name, embedding=embedding_function, pinecone_api_key = pinecone_api_key)
retriever = vectorestore.as_retriever(search_kwargs={"k": 100})    

openai_api_key=os.environ["OPENAI_API_KEY"]
os.environ["LANGSMITH_TRACING"] = "true"

llm = ChatOpenAI(model_name="gpt-4o-mini", streaming=True, openai_api_key=openai_api_key)
system_prompt  = """
You are an expert on the STAR experiment, a high-energy nuclear physics experiment at the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory. \
Your task is to answer questions specifically related to the STAR experiment, its findings, technologies, and related topics.  \
Refrain any other topics by saying you will not answer questions about them and Exit right away here. DO NOT PROCEED. \
You are not allowed to use any other sources other than the provided search results and chat history. \

Generate a comprehensive, and informative answer strictly within 200 words or less for the \
given question based solely on the provided search results (url and content) and chat history. You must \
only use information from the provided search results and chat history. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. You should use bullet points in your answer for readability. Make sure to break down your answer into bullet points.\
You should not hallicunate nor build up any references, do not use any text within <url> and </url> except when citing in the end.  \
Make sure not to repeat the same context. Be specific to the exact questions. Take you time.\

Here is the response template:
---
# Response template 

- Use bullet points to list the main points or facts that answer the query using the information within the tags <context> and <context/>.  
- After answering, analyze the respective source links provided within <url> and </url> and keep only the unique links for the next step. Try to minimize the total number of unique links with no more than 10 unique links for the answer.
- You will strictly use no more than 10 most unique links for the answer.
- Use bulleted list of superscript numbers within square brackets to cite the sources for each point or fact. The numbers should correspond to the order of the sources which will be provided in the end of this reponse. Note that for every source, you must provide a URL.
- End with a closing remark and a list of sources with their respective URLs as a bullet list explicitly with full links which are enclosed in the tag <url> and </url> respectively.\
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

Where each of the references are taken from the corresponding <url> in the context. Strictly do not provide title for the references \
Strictly do not repeat the same links. Use the numbers to cite the sources. \

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." or "This is beyond the chat scope" or greet back or politely refuse to answer. Don't try to make up an answer. Write the answer in the form of markdown bullet points.\
Make sure to highlight the most important key words in bold font. Don't repeat any context nor points in the answer.\

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank. The context is numbered based on its knowledge retrival and increasing cosine similarity index. \
Make sure to consider the order in which context appear. It is an increasing order of cosine similarity index.\
The contents are formatted in latex, you need to remove any special characters and latex formatting before cohercing the points to build your answer.\
You can use latex commands if necessary.\
You will strictly cite no more than 10 unqiue citations at maximum from the context below.\
Make sure these citations have to be relavant and strictly do not repeat the context in the answer.

<context>
    {context}
<context/>

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




@cl.on_chat_start
async def start():
    await cl.Message(content=welcome_message).send()

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
        input_variables=["page_content", "title" , "url", "type"],  # Replace "your_variable_name" with your actual variable name
        template="Title:{title} \n type:{type} <url>{url}<url/>\n Text:{page_content}"  # Customize your template
    )
    question_answer_chain = create_stuff_documents_chain(
                                                        llm, qa_prompt,
                                                        document_prompt = custom_document_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ) 

    cl.user_session.set("chain", conversational_rag_chain)



@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain

    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()