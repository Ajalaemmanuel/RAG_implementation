# RAG_implementation
Implementation of Retrieval-Augmented Generation (RAG) with fine-tuning and data files

  
# Retrieval Augmented Generation (RAG) Implementation

This repository contains an implementation of **Retrieval Augmented Generation (RAG)**, a method that enhances large language models (LLMs) by retrieving external knowledge sources. The repository includes code for fine-tuning a model using RAG and a folder containing data files for testing.

## Table of content
* Overview
* Motivation
* Technical Aspect
* Directory Tree
* References

  
## Overview
Retrieval Augmented Generation is a process that improves the out of a large language model by retrieving information from an external knowledge base and this helps the LLM give the users more accurate, up to date information.

---

## Motivation
Large language models (LLMs) are powerful, but they have limitations but with RAG, we can fix limitations like:
- Hallucination: Hallucination in Ai is when wrong information is being generated by a LLM due to insufficient data source to answer the given prompt and RAG can solve this problem by providing needed insight from the external data source
- Out of Date Information: LLMs are trained on data which can be limited to a certain timeline, this problem can be fixed with RAG by providing a more recent information and helps the LLM give insights on a more accurate data. 

---

## Technical Aspect
The RAG process involves three main steps:
1. **Indexing**: External documents are preprocessed, split into chunks, and encoded into vector representations for efficient retrieval.
2. **Retrieval**: When a query is received, the system retrieves the most relevant document chunks based on semantic similarity.
3. **Generation**: The retrieved documents and the query are combined into a prompt, which is fed into the LLM to generate a response.

### Code Implementation
- I imported the necessary libaries
- Loaded my datasets
- Used a tokenizer to help the model with words
- The documents were retrieved in the data folder
- I then combined the retriever and tokenizer to generate answers
- Fine tuning was performed using the hugging face's transformer library and FAISS for retrieval and I used 5 epochs( this can be changed depending on observed results)
- Saved the finetuned model


# Directory Tree
<pre>
RAG_implementation/
├── README.md
├── data/
│ ├── file1.pdf
│ ├── file2.pdf
│ └── file3.pdf
├── code/
│ ├── Rag_finetune.py
└── .gitignore

</pre>

# References
- https://doi.org/10.48550/arXiv.2312.10997
- https://huggingface.co/docs/transformers/index
- https://github.com/facebookresearch/faiss
- https://github.com/docugami/KG-RAG-datasets
