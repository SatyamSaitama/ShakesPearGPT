# ShakespearGPT: Seamlessly Integrating Shakespearean Wisdom with Conversational AI

**ShakespearGPT** seamlessly integrates the profound insights of **Shakespeare's literature** with the conversational prowess of **ChatGPT**. Leveraging the capabilities of **ChatGPT's API** and the timeless legacy of **Shakespeare's works**, we embark on a journey of exploration. With the aid of basic retrieval through **Top-K Similarity search**, our approach is fortified by a text file boasting approximately **40,000 lines**, surpassing the limitations of a single context window. By harnessing **Langchain** and constructing a **vector store**, we transcend constraints, furnishing **ChatGPT** with relevant paragraphs or vectors as a knowledge base. This innovative concept extends beyond text files to encompass **PDFs** and large documents, unlocking a realm of boundless possibilities.

## Installations
To get started with ShakespearGPT, follow these installation steps:

```bash
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
!pip install --upgrade langchain openai -q
!pip install unstructured -q
!pip install unstructured[local-inference] -q
!pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2 -q
!apt-get install poppler-utils
!pip install chromadb
!pip install tiktoken
```

## Load Data
Load the Shakespearean text data using the TextLoader utility of Langchain:

```python
from langchain.document_loaders import TextLoader

def load_docs(directory):
    loader = TextLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs('input.txt')
```

## Chunk Your Data
Divide the documents into smaller, manageable pieces using the RecursiveCharacterTextSplitter utility of Langchain:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)
```

## Create Embeddings
Prepare for similarity searches by embedding your documents using OpenAIEmbeddings:

```python
from langchain.embeddings.openai import OpenAIEmbeddings
import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_api_key')
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
```

## Create Vector Store
Create a VectorStore using Chroma:

```python
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(docs, embeddings)
```

## Test Vector Store
Test the vector store with a query:

```python
query = "What did Sebastian say to Antonio about eyelids"
docs = vectorstore.similarity_search(query)
```

## Integrating LLM with Langchain
Integrate the Language Model (LLM) using Langchain utilities:

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
```

Enjoy exploring the seamless integration of Shakespearean wisdom and conversational AI with ShakespearGPT!
