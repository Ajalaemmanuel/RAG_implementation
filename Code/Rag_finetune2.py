import os
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


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
                # Split long documents into chunks
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                for i, chunk in enumerate(chunks):
                    passages.append({"text": chunk, "title": f"{filename}_{i}"})
    return passages

# Step 3: Set up the retriever
class FaissRetriever:
    def __init__(self, embedder_model='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(embedder_model)
        self.passages = []
        self.faiss_index = None
        
    def add_passages(self, passages):
        self.passages = passages
        # Generate embeddings for passages
        embeddings = self.embedder.encode([p["text"] for p in passages])
        
        # Create a FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(np.array(embeddings).astype('float32'))
        
    def retrieve(self, query, top_k=3):
        # Generate embedding for the query
        query_embedding = self.embedder.encode([query])
        
        # Search in FAISS index
        scores, indices = self.faiss_index.search(np.array(query_embedding).astype('float32'), top_k)
        
        # Return the most relevant passages
        results = []
        for i in range(len(indices[0])):
            if indices[0][i] < len(self.passages):
                results.append(self.passages[indices[0][i]])
        return results

# Step 4: Set up the generator
class RAGModel:
    def __init__(self, generator_model="t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(generator_model)
        self.retriever = None
        
    def set_retriever(self, retriever):
        self.retriever = retriever
    
    def prepare_input(self, question, retrieved_contexts):
        # Prepare the input for the generator by combining question and contexts
        contexts_text = " ".join([ctx["text"] for ctx in retrieved_contexts])
        combined_input = f"question: {question} context: {contexts_text}"
        return combined_input
    
    def generate(self, question, max_length=150):
        if self.retriever is None:
            raise ValueError("Retriever is not set. Call set_retriever() first.")
        
        # Retrieve relevant contexts
        contexts = self.retriever.retrieve(question)
        
        # Prepare input
        combined_input = self.prepare_input(question, contexts)
        
        # Generate answer
        inputs = self.tokenizer(combined_input, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer, contexts
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(os.path.join(path, "generator"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))

# Step 5: Fine-tune the model
class RAGDataset(TorchDataset):
    def __init__(self, questions, contexts, answers, tokenizer, max_length=1024):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        answer = self.answers[idx]
        
        combined_input = f"question: {question} context: {context}"
        
        input_encodings = self.tokenizer(
            combined_input,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        target_encodings = self.tokenizer(
            answer,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encodings["input_ids"].flatten(),
            "attention_mask": input_encodings["attention_mask"].flatten(),
            "labels": target_encodings["input_ids"].flatten(),
        }

def fine_tune_rag_model(rag_model, qa_dataset, passages, output_dir="fine_tuned_rag_model", epochs=3):
    questions = qa_dataset["question"]
    answers = qa_dataset["answer"]
    
    # For each question, retrieve relevant passages
    retriever = rag_model.retriever
    contexts = []
    for question in questions:
        retrieved = retriever.retrieve(question)
        context_text = " ".join([ctx["text"] for ctx in retrieved])
        contexts.append(context_text)
    
    # Create a PyTorch dataset
    train_dataset = RAGDataset(questions, contexts, answers, rag_model.tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
    )
    
    # Define data collator
    data_collator = DataCollatorForSeq2Seq(rag_model.tokenizer)
    
    # Initialize Trainer
    trainer = Trainer(
        model=rag_model.model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model
    rag_model.save(output_dir)
    print(f"Model saved to {output_dir}")

# Step 6: Answer questions interactively
def answer_questions(rag_model):
    print("RAG Model is ready! Type 'exit' to quit.")
    
    while True:
        question = input("\nAsk a question: ")
        if question.lower() == "exit":
            break
        
        answer, contexts = rag_model.generate(question)
        
        print(f"Answer: {answer}")
        print("\nRetrieved passages:")
        for i, ctx in enumerate(contexts):
            print(f"{i+1}. {ctx['title']}: {ctx['text'][:100]}...")

# Main function
if __name__ == "__main__":
    # Directory containing the PDF files
    pdf_data_dir = "/content/RAG_implementation/data/docs/files/"
    
    # Path to the Q&A CSV file
    qa_csv_path = "/content/RAG_implementation/data/docs/qna_data.csv"
    
    # Step 1: Load the Q&A dataset
    qa_dataset = load_qa_dataset(qa_csv_path)
    
    # Step 2: Load passages from directory
    print("Loading passages from documents...")
    passages = load_data_from_directory(pdf_data_dir)
    
    # Step 3: Set up the retriever
    print("Setting up the retriever...")
    retriever = FaissRetriever()
    retriever.add_passages(passages)
    
    # Step 4: Set up the RAG model
    print("Setting up the generator model...")
    rag_model = RAGModel()
    rag_model.set_retriever(retriever)
    
    # Step 5: Fine-tune the model
    print("Fine-tuning the model...")
    fine_tune_rag_model(rag_model, qa_dataset, passages)
    
    # Step 6: Answer questions interactively
    answer_questions(rag_model)
