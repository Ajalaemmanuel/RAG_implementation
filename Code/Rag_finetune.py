import os
import pandas as pd
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import Dataset
import torch
import pdfplumber

# Step 1: Load the Q&A dataset 
def load_qa_dataset(csv_path):
    """
    Load the Q&A dataset from a CSV file.
    """
    df = pd.read_csv(csv_path)
    qa_data = [{"question": row["Question"], "answer": row["Answer"]} for _, row in df.iterrows()]
    return Dataset.from_dict({"question": [item["question"] for item in qa_data],
                             "answer": [item["answer"] for item in qa_data]})

# Step 2: Extract text from files
def load_data_from_directory(data_dir):
    """
    Load passages from PDF files in the data directory.
    """
    passages = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            if text.strip():
                passages.append({"text": text, "title": filename})
    return passages

# Step 3: Set up the RAG model
def setup_rag_model(data_dir):
    """
    Initialize the RAG model and retriever using the data from the directory.
    """
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

# Step 4: Fine-tune the model
def fine_tune_model(model, tokenizer, dataset):
    """
    Fine-tune the RAG model on the Q&A dataset.
    """
    # Set up the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Fine-tuning loop
    for epoch in range(5):
        print(f"Epoch {epoch + 1}/3")
        for batch in dataset:
            # Tokenize the question and answer
            inputs = tokenizer(batch["question"], return_tensors="pt", max_length=512, truncation=True)
            labels = tokenizer(batch["answer"], return_tensors="pt", max_length=512, truncation=True)

            # Forward pass
            outputs = model(input_ids=inputs["input_ids"], labels=labels["input_ids"])
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Loss: {loss.item()}")

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_rag_model")
    tokenizer.save_pretrained("fine_tuned_rag_model")
    print("Fine-tuning complete. Model saved to 'fine_tuned_rag_model'.")

# Step 5: Answer questions 
def answer_questions(model, tokenizer):
    """
    Allow the user to ask questions and get answers from the RAG model.
    """
    print("RAG Model is ready! Type 'exit' to quit.")
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
    # Directory containing the PDF files
    pdf_data_dir = '/content/RAG_implementation/data/docs/files/'

    if not os.path.exists(pdf_data_dir):
        print(f"Directory not found: {pdf_data_dir}")
    else:
        print(f"Directory found: {pdf_data_dir}")
        print("Files in the directory:")
    for filename in os.listdir(pdf_data_dir):
        print(filename)
    
    # Path to the Q&A CSV file
    # qa_csv_path = "RAG_implementation/data/docs/qna_data.csv"  
    # from google.colab import files
    # uploaded = files.upload() 
    qa_csv_path = "qna_data.csv"  

    # Step 1: Load the Q&A dataset
    qa_dataset = load_qa_dataset(qa_csv_path)

    # Step 2: Set up the RAG model
    print("Setting up the RAG model...")
    tokenizer, model = setup_rag_model(pdf_data_dir)

    # Step 3: Fine-tune the model
    print("Fine-tuning the model...")
    fine_tune_model(model, tokenizer, qa_dataset)

    # Step 4: Answer questions interactively
    answer_questions(model, tokenizer)
