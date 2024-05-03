from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# loading our env
load_dotenv()

# creating our embedding(a tool that takes unstructured data and transform it to structured)
embeddings = OpenAIEmbeddings()


def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    """
    This function creates a vector database from a youtube video url.
    in other words, it takes the url then
    :param video_url:
    :return FAISS:
    """
    # creating our youtube loader
    youtube_loader = YoutubeLoader.from_youtube_url(video_url)
    # taking transcript from the youtube loader
    transcript = youtube_loader.load()

    # creating our text splitter which will help us reduced the size of our transcript into smaller ones
    # because our model can't handle huge number of tokens at once
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # using text splitter to transform the transcript into smaller parts
    docs = text_splitter.split_documents(transcript)

    # creating a db that holds the docs we created in the form of number representation by using the embeddings
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k):
    """
    This function takes a query and a vector database and returns the result
    :param db:
    :param query:
    :param k:
    :return:
    """
    # we are setting docs to be only docs that are relevant to the question rather than the whole thing
    docs = db.similarity_search(query, k=k)
    # since our token can be up to 4k and each doc is 1k we will join 4 docs tgt to send
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(temperature=0.3)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        you are a helpful Youtube assistant that can answer questions about videos.
        based on the video's transcript.
        
        Answer the following {question}
        by searching the following video transcript: {docs}
        
        only use factual information from the transcript to answer questions.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        your answers should be detailed
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_key="answer")

    response = chain.invoke({'question': query, 'docs': docs_page_content})
    #response = response.replace("\n", " ")
    return response
