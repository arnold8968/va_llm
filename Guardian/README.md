# VA Disability Conversational Knowledge Retrieval System

## Project Description

This project demonstrates the creation of a conversational knowledge retrieval system using the domain knowledge from the VA disability benefits website. The system leverages the following 21 articles from the VA website to provide accurate answers to user queries:
This project aims to create a conversational knowledge retrieval system using domain knowledge from the VA disability benefits website. The system uses advanced techniques in large language models (LLMs), LangChain, and prompt engineering to provide accurate answers to user queries based on the content from selected VA articles.

Example:
- https://www.va.gov/disability/after-you-file-claim/
- https://www.va.gov/change-address/
- https://www.va.gov/change-direct-deposit/
- https://www.va.gov/claim-or-appeal-status/



The goal is to demonstrate proficiency in using LLMs, LangChain, prompt engineering, and related technologies to build a successful LLM application.

## Folder Structure

### 1. `prepare_data.ipynb`
The project started with scraping the content from 21 selected VA articles. The content was saved in HTML format, focusing on the paragraphs/q_a_section structure, and stored in a dataframe to prepare for vector database creation.

- Analyzes the VA website and saves the content in HTML format.
- The web structure follows `paragraphs/q_a_section`.
- Data is saved in a dataframe with layout structure, preparing for vector database.
- Includes question generation for retrieval evaluation using the OpenAI API.

### 2. `vector_db.ipynb`
Using the all-MiniLM-L6-v2 embedding model, the scraped data was converted into embeddings and stored in a Chroma vector database. This setup allows efficient similarity searches.

- Embedding model: `all-MiniLM-L6-v2`
- Uses Chroma vector database.

### 3. `RAG_LLM.ipynb`
#### Prompt Engineering
Effective prompt engineering was applied to ensure the model receives clear instructions, allowing it to think through the problem and break down complex tasks. This approach improves the quality of the generated responses.
- The prompt structure is tested in the playground and follows clear instructions:
  - Give the model time to think.
  - Break down complex tasks.
  - Format the answer output.

#### Different Models
1. **Basic LLaMA**: Asks the question without any related content.
2. **RAG LLaMA**: Uses vector database to calculate similarity and adds the content into the prompt.
3. **RAG Rerank LLaMA**: (Brief explanation of how it works included in the notebook)

- For better comparison, all outputs are saved into a dataset.

### 4. `finetuning_qlora.ipynb`
Although the fine-tuning process was not completed due to data and computational limits, the initial setup for fine-tuning the LLaMA model using the QLoRA algorithm was included. This technique aims to further improve the model's performance by fine-tuning it on specific question-answer pairs.
- Includes code for fine-tuning the LLaMA model using QLoRA algorithm.
- Due to data and computational resource limits, this part was not completed but includes the working code.

### 5. `evaluation.ipynb`
The models were evaluated using the Average BLEU Score and Average ROUGE-L Score metrics. These metrics were chosen to assess the quality of the generated answers based on their relevance and fluency.

- Evaluates using Average BLEU Score and Average ROUGE-L Score.
- Explains the reason for these metrics.

#### Evaluation Results
- **Method: LLaMA**
  - Average BLEU Score: 0.05
  - Average ROUGE-L Score: 0.25

- **Method: RAG LLaMA**
  - Average BLEU Score: 0.21
  - Average ROUGE-L Score: 0.45

- **Method: RAG LLaMA Rerank**
  - Average BLEU Score: 0.21
  - Average ROUGE-L Score: 0.48

## Installation Instructions

pip install -r requirements.txt

## Analysis
The RAG LLaMA models significantly outperformed the basic LLaMA model, as indicated by the higher BLEU and ROUGE-L scores. The RAG LLaMA Rerank model achieved the best performance, demonstrating the effectiveness of combining similarity search with reranking to improve the relevance and accuracy of the generated answers.

Key Takeaways

Data Preparation: Proper structuring and saving of data in HTML format allowed for efficient vector database creation and retrieval.

Embedding and Vector Database: Using a high-quality embedding model and vector database significantly enhanced the retrieval process.

Prompt Engineering: Well-designed prompts that give clear instructions and format the output correctly are crucial for improving model performance.

RAG Models: Combining retrieved content with the question in the prompt, especially with reranking, greatly improves the accuracy of the answers.

Evaluation Metrics: BLEU and ROUGE-L scores provided valuable insights into the quality of the generated answers.