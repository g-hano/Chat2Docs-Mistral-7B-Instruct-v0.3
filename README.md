# Chat2Docs - Mistral 7B Instruct v0.3

Welcome to Chat2Docs, a Retrieval-Augmented Generation (RAG) project designed to facilitate querying over provided documents using advanced language models. This project leverages a hybrid search mechanism to fetch relevant information and answer questions accurately.

![Demo](https://huggingface.co/spaces/Chan-Y/Chat2Docs-Mistral-7B-Instruct-v0.3/resolve/main/demo.gif)

## Features

- **Hybrid Search**: Combines keyword-based and semantic search to retrieve the most relevant information from documents.
- **Multi-Model Support**: Compatible with OpenAI, Ollama, and HuggingFace LLMs.
- **Gradio Interface**: Easy-to-use web interface for interacting with the model, hosted on HuggingFace Spaces.
- **Custom Implementation**: LlamaIndex framework extended with custom hybrid search capabilities.
- **Excel File Support**: Custom solution added to LlamaIndex for reading Excel files.

## Live Demo

Try the Gradio app live on HuggingFace Spaces: [Chat2Docs - Mistral 7B Instruct v0.3](https://huggingface.co/spaces/Chan-Y/Chat2Docs-Mistral-7B-Instruct-v0.3)

## Getting Started

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/g-hano/Chat2Docs-Mistral-7B.git
    cd Chat2Docs-Mistral-7B
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up your API keys for the language models (OpenAI, Ollama, and HuggingFace):
    ```sh
    export OPENAI_API_KEY='your_openai_api_key'
    export HF_TOKEN='your_huggingface_token'
    ```

### Usage

### Simple Usage Example

Here's a simple example to get you started with the hybrid search using this project:

```python
from HybridRetriever import HybridRetriever
from ChatEngine import ChatEngine
from llama_index.retrievers.bm25 import BM25Retriever 
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter

# Initialize the language model
llm = Ollama()  # or HuggingFace() or OpenAI()

# Load documents from the directory
documents = SimpleDirectoryReader("./data").load_data()

# Set up the text splitter
text_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# Create a vector index from the documents
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[text_splitter], embed_model=Settings.embed_model, show_progress=True
)

# Set up the retrievers
bm25_retriever = BM25Retriever(nodes=documents, similarity_top_k=TOP_K, tokenizer=text_splitter.split_text)
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=TOP_K)
hybrid_retriever = HybridRetriever(bm25_retriever=bm25_retriever, vector_retriever=vector_retriever)

# Initialize the chat engine
chat_engine = ChatEngine(hybrid_retriever)

# Ask a question and get a response
response = chat_engine.ask_question("<question here>", llm)
print(response)
```
2. Open your web browser and navigate to the provided local URL to start interacting with the model.

### Hybrid Search Implementation

The hybrid search functionality is implemented by combining keyword-based search with semantic search. The LlamaIndex framework has been extended to support this feature, ensuring that the most relevant information is retrieved from the documents.

### Excel File Support

LlamaIndex does not natively support reading Excel files. A custom solution has been implemented and contributed to the LlamaIndex framework to enable this functionality. The implementation is as follows:

```python
import pandas as pd
import json

from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class ExcelReader(BaseReader):
    def load_data(self, file_path: Path, extra_info: Optional[Dict] = None) -> List[Document]:
        if extra_info is not None:
            if not isinstance(extra_info, dict):
                raise TypeError("extra_info must be a dictionary.")
            
        df = pd.read_excel(file_path)
        # Create a list to store the documents
        documents = []
         
        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            # Convert the row to a dictionary
            row_dict = row.to_dict()
            # Serialize the dictionary to a JSON-formatted string
            json_string = json.dumps(row_dict, indent=4)
            # Create a Document for each JSON string
            documents.append(Document(text=json_string, metadata=extra_info))

        return documents
```
### Supported Models

- **OpenAI GPT-3/GPT-4**
- **Ollama LLMs**
- **HuggingFace LLMs**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please reach out via [Linkedin](https://www.linkedin.com/in/chanyalcin/)

---

Thank you for using Chat2Docs!
