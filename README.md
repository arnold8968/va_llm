# VA Disability Conversational Knowledge Retrieval System

## Project Description

This project aims to build a conversational knowledge retrieval system focused on VA disability benefits. The system leverages Large Language Models to provide accurate and detailed answers to questions related to VA disability claims and benefits. 

The knowledge base consists of 21 articles sourced from the VA website. The models used include LLaMA 3, RAG with LLaMA3, and RAG with reranking. The project demonstrates expertise in LLMs, LangChain, prompt engineering, and fine-tuning using the QLoRA algorithm.


Example:
- https://www.va.gov/disability/after-you-file-claim/
- https://www.va.gov/change-address/
- https://www.va.gov/change-direct-deposit/
- https://www.va.gov/claim-or-appeal-status/


## Folder Structure

### 1. `prepare_data.ipynb`
 - Analyzes the VA website and saves the content in HTML format.
 - Structures the data in a dataframe for vector database preparation.
 - Generates questions for retrieval evaluation using the ChatGPT.

The project started with scraping the content from 21 selected VA articles. The content was saved in HTML format, focusing on the paragraphs/q_a_section structure, and stored in a dataframe to prepare for vector database creation.



### 2. `vector_db.ipynb`
- Embedding model: `all-MiniLM-L6-v2`
- Uses Chroma vector database.
  
Using the all-MiniLM-L6-v2 embedding model, the scraped data was converted into embeddings and stored in a Chroma vector database. This setup allows efficient similarity searches.


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


![image](https://github.com/arnold8968/va_llm/assets/9800659/2cfead5b-6eea-4764-9e5d-3c0a8bae8c78)

Key Takeaways

Data Preparation: Proper structuring and saving of data in HTML format allowed for efficient vector database creation and retrieval.

Embedding and Vector Database: Using a high-quality embedding model and vector database significantly enhanced the retrieval process.

Prompt Engineering: Well-designed prompts that give clear instructions and format the output correctly are crucial for improving model performance.

RAG Models: Combining retrieved content with the question in the prompt, especially with reranking, greatly improves the accuracy of the answers.

Evaluation Metrics: BLEU and ROUGE-L scores provided valuable insights into the quality of the generated answers.


### Analysis of Sample Outputs

Example 1:

Question: What is the average number of days it takes for VA to complete disability-related claims in May 2024?

Content: 147.2 days Average number of days to complete disability-related claims in May 2024 Average number of days to complete disability-related claims in May 2024

Basic LLM Answer:
Output: "The average number of days it takes for VA to complete disability-related claims in May 2024 is 3 days."
Analysis: This response is incorrect. It significantly deviates from the given content, indicating a potential issue with the model's understanding or generation process.

RAG LLM Answer:
Output: "147"
Analysis: This response captures the essence but lacks completeness. The numerical value is correct, but it lacks context and additional information to be considered a comprehensive answer.

RAG LLM with Reranking Answer:
Output: "The average number of days it takes for VA to complete disability-related claims in May 2024 is 147.2 days."
Analysis: This response is accurate and complete, providing both the numerical value and the context. It demonstrates the benefit of reranking in generating more detailed and precise answers.


Example 2:

Question: What factors affect the time it takes to review a claim for disability after it has been filed?

Content: The time it takes to review your claim depends on these factors: The time it takes to review your claim depends on these factors: The type of claim you filed How many injuries or disabilities you claimed and how complex they are How long it takes us to collect the evidence we need to decide your claim

Basic LLM Answer:
Output: "The time it takes to review a claim for disability after it has been filed can vary depending on several factors, including the complexity of the claim, the availability of medical records and other evidence, and the workload of the VA regional office handling the claim. In general, the VA strives to process claims within a certain timeframe, but there may be delays due to these and other factors."
Analysis: This response is detailed but partially generic. It does not fully reflect the specific factors mentioned in the content but provides a broader view.

RAG LLM Answer:
Output: "The type of claim you filed, how many injuries or disabilities you claimed and how complex they are, and how long it takes us to collect the evidence we need to decide your claim."
Analysis: This response directly addresses the specific factors mentioned in the content. It's concise and accurate.

RAG LLM with Reranking Answer:
Output: "The time it takes to review a claim for disability after it has been filed is affected by the type of claim filed, the number and complexity of injuries or disabilities claimed, and the time it takes to collect the necessary evidence to decide the claim."
Analysis: This response is similar to the RAG LLM answer but slightly more polished in phrasing. It effectively captures the key factors.

## Overall Analysis

The RAG LLM with reranking generally provides the most accurate and comprehensive answers, closely adhering to the provided content. The Basic LLM sometimes introduces inaccuracies or additional information not present in the context, while the standard RAG LLM improves accuracy but may lack completeness. Reranking enhances the quality of the answers by refining the relevance and detail of the generated responses.


# Future steps to improve the performance
Enhance Embedding and Retrieval

Improve Vector Database Performance

Apply knowledge graph LLM

Fine-Tune on Domain-Specific Data

Enhanced RAG and Reranking. Dynamic Context Window, Advanced Reranking Models
