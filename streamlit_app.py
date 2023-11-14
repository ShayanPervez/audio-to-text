import os
import streamlit as st
import whisper
from langchain.chains import RetrievalQA
from audio_recorder_streamlit import audio_recorder
from audiorecorder import audiorecorder
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
# from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import SVMRetriever



st.title("Avtarcoach Audio-to-text")

# audio_bytes = audio_recorder("Click to record", "Click to stop recording", neutral_color="#051082", icon_size="2x")
# if audio_bytes:
#     st.audio(audio_bytes, format="audio/wav")

audio = audiorecorder("Click to record", "Click to stop recording")

if len(audio) > 0:
    # To play audio in frontend:
    st.audio(audio.export().read())

    # To save audio to a file, use pydub export method:
    audio.export("audio.wav", format="wav")

    # To get audio properties, use pydub AudioSegment properties:
    st.write(
        f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(r"audio.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
st.write(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions(fp16=False)
result = whisper.decode(model, mel, options)

# print the recognized text
st.write("You Said: ", result.text)
input_text = result.text

st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

st.write("Avtarcoach Response: ")

# Gen AI results

pdf_loader = DirectoryLoader(
    r'C:\Users\shpe1\Downloads\tea_project_text_to_text-main\tea_project_text_to_text-main\pdf_docs', glob="**/*.pdf",
    use_multithreading=True)
docs_loader = DirectoryLoader(
    r'C:\Users\shpe1\Downloads\tea_project_text_to_text-main\tea_project_text_to_text-main\docs', glob="**/*.docx",
    use_multithreading=True)
csv_loader = DirectoryLoader(
    r'C:\Users\shpe1\Downloads\tea_project_text_to_text-main\tea_project_text_to_text-main\docs', glob="**/*.csv",
    use_multithreading=True)
xlsx_loader = DirectoryLoader(
    r'C:\Users\shpe1\Downloads\tea_project_text_to_text-main\tea_project_text_to_text-main\docs', glob="**/*.xlsx",
    use_multithreading=True)
loaders = [pdf_loader, docs_loader, csv_loader, xlsx_loader]

documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitters = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitters.split_documents(documents)
embedding = OpenAIEmbeddings()
# db = FAISS.from_documents(chunks, embedding)
faiss_db = FAISS.from_documents(chunks, embedding)
retriever = faiss_db.as_retriever(search_type='mmr')
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# doc_search =faiss_db.get_relevant_documents(input_text)
# llm = ChatOpenAI(model="gpt-4-1106-preview",temperature =0)
# qa_chain = load_qa_chain(llm=llm,chain_type="stuff")
# qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
response = qa_chain.run(input_text)

st.write(response)
