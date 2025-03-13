import os
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset
#load my datasets
dataset = load_dataset("wiki_dpr", "psgs_w100_multiple")

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="custom", passages_path="data/")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever) #combine the tokenizer and retriever

# Fine-tuning loop 
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
