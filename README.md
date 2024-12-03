# QA_assesment

# RAG-Based QA Chatbot(English) Using SQuAD_v2 Data  

## Project Overview  
This repository implements a **Retrieval-Augmented Generation (RAG)** based Question Answering (QA) chatbot using the **SQuAD_v2** dataset. The chatbot combines efficient context retrieval and answer generation to provide accurate responses to user queries. The project showcases the process of building a conversational QA system from scratch, including training and deploying a simple interface in Google Colab.  

---

## Features  

### 1. Context Retrieval  
- **Sentence Transformer**: Used to encode entire context paragraphs into dense vector embeddings.  
- **FAISS Index**: Enables efficient and scalable retrieval of the most relevant contexts for a given query.  

### 2. Answer Generation  
- **Fine-Tuned T5 Model**: A T5-small model fine-tuned on a subset of SQuAD_v2 data is used to generate accurate answers from the retrieved context.  

### 3. Chatbot Deployment  
- A simple chatbot interface implemented in Google Colab.  
- Interactive conversation capability to query and receive answers in real time.  

---

## Limitations  
- **Model Size**: Due to resource constraints, a T5-small model was used, and only a small portion of SQuAD_v2 was utilized for fine-tuning.  
- **Deployment**: The project is designed to run entirely in Google Colab and does not include an external API or GUI due to file size limitations on free accounts.  

---

## How It Works  

1. **Data Preprocessing**:  
   - The SQuAD_v2 dataset is preprocessed to structure the input for both the retriever and generator.  

2. **Retriever**:  
   - Context paragraphs are encoded using Sentence Transformer to create embeddings.  
   - A FAISS index is built for these embeddings, enabling fast and accurate context retrieval.  

3. **Generator**:  
   - A T5-small model fine-tuned on SQuAD_v2 data is used to generate answers based on the retrieved context.  

4. **Chatbot Interface**:  
   - The chatbot interface is implemented in Google Colab to allow real-time user interaction.  

---

## Future Improvements  
- **Model Scalability**: Use a larger T5 model and a bigger portion of the SQuAD_v2 dataset for improved performance.  
- **API Integration**: Implement an external API to make the chatbot deployable outside of Colab.  
- **GUI Development**: Design a standalone graphical user interface (GUI) for easier user interaction.  

---

## Getting Started  

### Prerequisites  
- Python 3.8+  
- Google Colab  
- Libraries: Hugging Face Transformers, FAISS, Sentence Transformers  

### Instructions  
1. Clone the repository.  
2. Open the provided notebook in Google Colab.  
3. Follow the steps to train the retriever and generator.  
4. Interact with the chatbot in the Colab interface.  

---
## Fine-Tuned Model  
The fine-tuned T5 model and FAISS index is available for download [here]([https://drive.google.com/file/d/<YOUR-FILE-ID](https://drive.google.com/file/d/1hKHObUp0JnkjrPd0kChy2Kifk7n-QRWX/view?usp=sharing)>) (Google Drive link).  


## Acknowledgments  
This project uses:  
- **Hugging Face Transformers** for T5 implementation and fine-tuning.  
- **Sentence Transformers** for context embedding.  
- **FAISS** for fast nearest neighbor search.  
- **SQuAD_v2 Dataset** for QA training data.  

Feel free to contribute to this project or raise issues for further enhancements!
