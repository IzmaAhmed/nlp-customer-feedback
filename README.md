# Customer Feedback NLP System

A natural language processing system for analyzing customer feedback with sentiment analysis and text summarization.

## Project Structure

- `app.py` - Main Flask application
- `sentiment_analysis.ipynb` - Sentiment analysis model development
- `text_summarization_*.ipynb` - Text summarization models
- `final_sentiment_model/` - Trained sentiment model
- `t5-summarizer_results/` - Summarization model results
- `requirements.txt` - Python dependencies

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`

## Model Files Setup

The large model files are not included in this repository due to size limits.

1. Download model files from 
[https://drive.google.com/drive/folders/11b8czG52Pqdk2_6K2SIILtxfRVIxnfxb?usp=sharing
https://drive.google.com/drive/folders/1WkOJnX55GS8sGLDLo5Os6Hbug6-0FJrX?usp=sharing]

2. Place them in the respective folders:
   - `final_sentiment_model/`
   - `t5-summarizer_results/`
  
## View Notebooks Online:
- [Sentiment Analysis Notebook](https://nbviewer.org/github/IzmaAhmed/nlp-customer-feedback/blob/master/sentiment_analysis.ipynb)
- [Text Summarization Notebook](https://nbviewer.org/github/IzmaAhmed/nlp-customer-feedback/blob/master/text_summarization_2.ipynb)
