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
## Fine-Tuned Model And FAISS Index
The fine-tuned T5 model is available with FAISS Index for download  [here](https://drive.google.com/file/d/1hKHObUp0JnkjrPd0kChy2Kifk7n-QRWX/view?usp=sharing).  
 


## Acknowledgments  
This project uses:  
- **Hugging Face Transformers** for T5 implementation and fine-tuning.  
- **Sentence Transformers** for context embedding.  
- **FAISS** for fast nearest neighbor search.  
- **SQuAD_v2 Dataset** for QA training data.  

Feel free to contribute to this project or raise issues for further enhancements!

# Bengali News Retrieval and Summarization Bot

This repository contains a project designed to scrape news articles from **The Daily Star** and organize them into current and archived news categories. The system uses a FAISS index for efficient similarity-based retrieval and provides an overview of relevant news articles using a fine-tuned BERT-based summarizer. This README describes the methodology, features, and usage of the project.

---

## **Project Overview**

### **Objectives**
1. **News Categorization**:
   - Store current day's news in a DataFrame [use 5th december,2024 as current date . you can scrape and update dataframe].
   - Archive older news by categories (e.g., title, text, URL).
2. **Similarity-Based Search**:
   - Create FAISS indices for chunked news text using custom embeddings and vocabulary.
   - Use word embeddings (trained using Word2Vec) combined with TF-IDF scores to enhance semantic understanding.
3. **User Query Handling**:
   - Determine if a query is for current or archived news.
   - Identify if the user is asking for a title, category, or full text.
4. **Text Summarization**:
   - Summarize retrieved news text to generate concise responses.

### **Key Features**
- **Web Scraping**:
  - Scrapes news articles daily using Beautiful Soup.
  - Automatically categorizes and organizes the data.

- **Custom Embedding & Indexing**:
  - Uses a trained Word2Vec embedder.
  - Combines embeddings with TF-IDF scores for chunked news text.
  - Builds FAISS indices for efficient similarity searches.

- **Query Handling**:
  - Identifies user intent to fetch either current or archived news.
  - Retrieves relevant information based on titles, categories, or full-text matches.

- **Text Summarization**:
  - Summarizes retrieved text to provide an overview of news articles.
  - Fine-tuned BERT model for Bengali text summarization.

---

## **File Structure**

- `scraper.ipynb`: Script for scraping news articles from The Daily Star.
- `Bangla_News_chatbot_QA.ipynb`: The notebook you can run and chat with newsbot.
- `BanglaTextSummarizer.ipynb`: Build the Text summarizer model by fine tuning bert using sentence embedder .


---

## **Dependencies**

- **Python**: 3.7+
- **Libraries**:
  - Beautiful Soup (for web scraping)
  - FAISS (for similarity-based indexing)
  - Transformers (for BERT-based summarization)
  - Gensim (for Word2Vec embeddings)
  - Pandas (for data manipulation)
  - Scikit-learn (for TF-IDF computation)
  - PyTorch (for fine-tuning BERT)

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## **Setup Instructions**

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Download resources:
   - Vocabulary and embedder: [Google Drive Link](https://drive.google.com/drive/folders/1A0wEw391OoMt6hYpVIo-MuTaxlf7QiXO?usp=sharing)
   - Dataset for text summarizer: [Google Drive Link](https://drive.google.com/file/d/12gBLxW9TnjlWa6D5FzqEc2C8uVGXQAeg/view?usp=sharing)



   ```

---

## **Usage**

### **1. Scraping News**
Run scrapper notebook to scrap news 

### **2. Querying the Bot**
The system determines whether the query relates to:
- **Current news**: Fetches the latest news information.
- **Archived news**: Searches for relevant articles in the categorized archive.

### **3. Summarization**
The retrieved news articles are summarized to provide a concise overview. The summarizer uses sentence embeddings based on a fine-tuned BERT model for Bengali.

---

## **Limitations and Future Work**

### **Challenges Faced**
- Computational limitations restricted fine-tuning to a dataset of 500 rows, impacting the summarizer's accuracy.
- Summarizer performance is suboptimal for long and complex queries.

### **Future Improvements**
- Expand the dataset for fine-tuning the summarizer to improve accuracy.
- Use distributed training to overcome computational constraints.
- Explore advanced architectures like GPT for better summarization performance.
- Integrate a feedback mechanism to improve query handling dynamically.

---

## **Acknowledgments**

All ideas for this project were conceptualized and implemented by the repository author. While some computational limitations were encountered, the project demonstrates the feasibility of combining embedding-based search and summarization for a Bengali news bot.

---

## **Contact**
For questions or suggestions, feel free to reach out:
- **Email**: [csebrur.hasinmanjare34@gmail.com]
- **GitHub**: [https://github.com/Hasin-Al]

---


