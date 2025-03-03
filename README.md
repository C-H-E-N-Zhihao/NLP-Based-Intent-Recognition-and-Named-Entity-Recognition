# NLP-Based-Intent-Recognition-and-Named-Entity-Recognition

## Project Overview
This project focuses on Natural Language Processing (NLP) for **Intent Recognition** and **Named Entity Recognition (NER)**. The goal is to develop deep learning models that can classify user intents and extract named entities from text, essential tasks for virtual assistants and chatbot applications.

## Features
### Intent Recognition
- Dataset with 22 different user intents.
- Preprocessing: Tokenization, stopword removal, stemming/lemmatization.
- Model comparison: CNN, LSTM, GRU, and hybrid approaches.
- Performance evaluation: Accuracy, F1-score, class balancing.

### Named Entity Recognition (NER)
- BIO encoding for entity classification.
- Deep learning models (LSTM, Transformers).
- Custom loss functions to address class imbalance.
- Hyperparameter tuning and model optimization.

## Installation
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd nlp-intent-ner-project
   ```
2. Install dependencies:
   ```sh
   pip install numpy pandas tensorflow keras nltk
   ```

## Dataset
- User intent classification dataset with labeled textual queries.
- Named entity dataset with BIO-encoded labels.

## Model Development Process
1. **Data Preprocessing**
   - Tokenization, lemmatization, stopword removal.
   - Encoding categorical data.
   - Class balancing strategies.
2. **Model Training & Evaluation**
   - Training deep learning models (LSTM, CNN, Transformers).
   - Fine-tuning hyperparameters.
   - Evaluating performance with precision, recall, F1-score.

## Results
- Intent classification model achieves **82% accuracy** with optimized embeddings.
- NER model improves F1-score using class-weighted training.

## Future Improvements
- Integrate reinforcement learning for dialogue management.
- Enhance entity recognition with BERT-based models.

## Authors
- **Zhihao Chen**
- **Zhiqian Zhou**

## References
- NLP & Machine Learning textbooks.
- Research papers on Named Entity Recognition and Intent Classification.

