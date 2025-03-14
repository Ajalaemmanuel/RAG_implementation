import os
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset
import torch

import xml.etree.ElementTree as ET
import pdfplumber

# Step 1: Load the dataset (replace with your data)
def load_data_from_directory(data_dir):
passages = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        text = ""
        
        if filename.endswith(".txt"):  # Read text files
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        
        elif filename.endswith(".pdf"):  # Extract text from PDFs
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        elif filename.endswith(".xml"):
            tree = ET.parse(file_path)
            root = tree.getroot()
            
        
        if text.strip():
            passages.append({"text": text, "title": filename})
    return passages

# Step 2: Set up the RAG model
def setup_rag_model(data_dir):
    # Load passages from the data directory
    passages = load_data_from_directory(data_dir)

    # Initialize the tokenizer, retriever, and model
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-base",
        index_name="custom",
        passages=passages,
        index_path=None,  
    )
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

    return tokenizer, model

# Step 3: Fine-tune the model
def fine_tune_model(model, tokenizer, dataset):

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(5): 
        for batch in dataset["train"]:
            inputs = tokenizer(batch["question"], return_tensors="pt", max_length=512, truncation=True)
            
            labels = tokenizer(batch["answer"], return_tensors="pt", max_length=512, truncation=True)
            
            outputs = model(input_ids=inputs["input_ids"], labels=labels["input_ids"])
            loss = outputs.loss
            
            loss.backward()
            
            # Update model weights
            optimizer.step()
            optimizer.zero_grad()

    # Save the fine tuned model
    model.save_pretrained("fine_tuned_rag_model")
    tokenizer.save_pretrained("fine_tuned_rag_model")

# Step 4: Answer questions interactively
def answer_questions(model, tokenizer):
    
    print("RAG Model is ready! Type 'exit' to exit.")

    
    while True:
        question = input("\nAsk a question: ")
        if question.lower() == "exit":
            break

        # Tokenize the question
        inputs = tokenizer(question, return_tensors="pt")

        # Generate the answer
        outputs = model.generate(input_ids=inputs["input_ids"])
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Answer: {answer}")

# Main function
if __name__ == "__main__":
    # Directory containing the data files
    data_dir = "data/"

    # Step 1: Set up the RAG model
    tokenizer, model = setup_rag_model(data_dir)

    # Step 2: Fine-tune the model

    dataset = load_dataset("wiki_dpr", "psgs_w100_multiple")
    fine_tune_model(model, tokenizer, dataset)

    # Step 3: Answer questions 
    answer_questions(model, tokenizer)
