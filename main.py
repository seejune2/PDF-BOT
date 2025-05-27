
# 환경 변수에서 API 키를 불러오기
import os
from dotenv import load_dotenv
# dotenv 불러오기 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangChain 패키지
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr
from gradio_pdf import PDF

# RAG Chain 구현을 위한 패키지
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# pdf 파일을 읽어서 벡터 저장소에 저장
def load_pdf_to_vector_store(pdf_file, chunk_size, chunk_overlap, similarity_metric):\

    # PDF 파일 로드
    pdf_loader = PyPDFLoader(pdf_file)

    documents = pdf_loader.load()

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
        collection_metadata={'hnsw:space': similarity_metric},
    )

    return vectorstore

# 벡터 저장소에'서 문서를 검색하고 답변을 생성
def retrieval_and_generate_answer(vectorstore, message, temperature=0):

    # RAG Chain 생성
    retriever = vectorstore.as_retriever()

    # prompt 템플릿 정의
    template = '''Answer the question based only on the following context.
    <context>
    {context}
    </context>
    Question: {input}
    '''
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-3.5-turbo", 
                     api_key=OPENAI_API_KEY,
                     temperature=temperature)
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever,document_chain)

    # 답변 생성
    response = rag_chain.invoke({"input": message,})

    return response['answer']

# Gradio 인터페이스에서 사용할 함수
def process_pdf_answer(message, history, pdf_file, chunk_size, chunk_overlap, similarity_metric, temperature):

    vectorstore = load_pdf_to_vector_store(pdf_file, chunk_size, chunk_overlap, similarity_metric)
    answer = retrieval_and_generate_answer(vectorstore, message, temperature)

    return answer

demo = gr.ChatInterface(fn = process_pdf_answer, 
                        additional_inputs = [PDF(label="Upload PDF File"),
                                             gr.Number(label= "Chunk_Size",value=1000),
                                             gr.Number(label= "Chunk_Overlap",value=200),
                                             gr.Dropdown(["cosine", "l2"], label="similarity_metric", value="cosine"),
                                             gr.Slider(label="Temperature", minimum=0, maximum=2, step=0.1, value = 0.5)
                                             ],)


demo.launch()