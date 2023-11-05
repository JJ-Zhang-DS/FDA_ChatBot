import streamlit as st
from llama_index import TreeIndex, VectorStoreIndex, ServiceContext, Document, get_response_synthesizer
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from llama_index.retrievers import VectorIndexRetriever,TreeAllLeafRetriever
from llama_index.query_engine import RetrieverQueryEngine


st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

#@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data/wo_data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=2,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode="compact",)

# assemble query engine

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,) #chat_mode="condense_question", verbose=True)

# here will be the test for example questions

questions = []
answers = []
with open("./questions.txt", 'r') as f:
    for line in f.readlines():
        questions.append(line.split('\n')[0])
with open("./answers.txt", 'r') as f:
    for line in f.readlines():
        answers.append(line.split('\n')[0])

assert len(questions) == len(answers)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# class BertEmbedding(nn.Module):
#     def __init__(self, dropout=0.5):
#         #super(BertClassifier, self).__init__()
#         super(BertEmbedding, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-cased',output_hidden_states=True)
#     def forward(self, input_id, mask):
#         _, pooled_output,_hidden_states = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
#         # Getting embeddings from the final BERT layer
#         token_embeddings = _hidden_states[-1]
#         # Collapsing the tensor into 1-dimension
#         token_embeddings = torch.squeeze(token_embeddings, dim=0)
#         # Converting torchtensors to lists
#         list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]
#         return token_embeddings
# model = BertEmbedding(dropout=0)

model = BertModel.from_pretrained('bert-large-uncased')
cossim_total = []
sorry_words = ["sorry", "not mention", "not provide", "no information", "no data"]

for idx, question in enumerate(questions):
    start = time.time()
    prompt = question
 #   st.session_state.messages.append({"role": "user", "content": prompt})
 #   
 #   with st.chat_message(st.session_state.messages[-1]["role"]):
    with st.chat_message("user"):
 #       st.write(st.session_state.messages[-1]["content"])
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.query(prompt)
            st.write(response.response)
            st.write("Retrieval Time: "+str(round(time.time()-start, 3))+"s")
            res_token = tokenizer(
                [response.response], max_length=128, padding='max_length', truncation=True)
            with torch.no_grad():
                res = model(
                    torch.tensor(res_token['input_ids']), torch.tensor(res_token['attention_mask']))
            res_hidden = res.last_hidden_state[:,0,:]
            st.write("Correct answer: "+ answers[idx])
            std_token = tokenizer(
                [answers[idx]], max_length=128, padding='max_length', truncation=True)
            with torch.no_grad():
                std = model(
                    torch.tensor(std_token['input_ids']), torch.tensor(std_token['attention_mask']))
            std_hidden = std.last_hidden_state[:,0,:]
            cossim = [[1]]
#            for sorry in sorry_words:
#                if sorry in response.response:
#                    cossim = [[0]]
#                    break
            
            if cossim[0][0] == 1:
                cossim = cosine_similarity(res_hidden.numpy(), std_hidden.numpy())
            cossim_total.append(cossim[0][0])
            st.write("Cosine Similarity = " + str(cossim[0][0]))

#            message = {"role": "assistant", "content": response.response}
#            st.session_state.messages.append(message) # Add response to message history

ave = np.mean(cossim_total)
if st.session_state.messages[-1]["role"] == "assistant":           
    with st.chat_message("assistant"):
        with st.spinner("Calculating averate cosine similarity..."):
            st.write("Average Cosine Similarity = "+str(ave))
                     
