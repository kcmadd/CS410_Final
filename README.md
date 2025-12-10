# Tweet Sentiment Pipeline + Semantic Search Chat

A full data-processing and question-answering pipeline for stock-related tweets.

## Dataset Source
Raw tweet dataset:
https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction/data

Place `stock_tweets.csv` into:
```
data/raw/stock_tweets.csv
```

## Dependencies

### Python Packages
```
pip install pandas numpy matplotlib textblob ollama
pip install python-dateutil tqdm
```

### TextBlob Corpora
```
python -m textblob.download_corpora
```

### Ollama
Install: https://ollama.com/download

Required models:
```
ollama pull llama3
ollama pull nomic-embed-text
```

## Embedding Cache (Optional)
Avoid long embedding computation by downloading a prebuilt cache:

{{URL}}

Place it here:
```
data/processed/tweet_embeddings.npz
```

## Project Structure
```
project/
│
├── main.py
├── data/
│   ├── raw/
│   │   └── stock_tweets.csv
│   └── processed/
│       ├── cleaned_tweets.csv
│       ├── weekly_sentiment.csv
│       ├── weekly_sentiment_heatmap.png
│       ├── tweet_embeddings.npz
└── README.md
```

## How to Run

1. Ensure dataset is placed at:
```
data/raw/stock_tweets.csv
```

2. Run:
```
python main.py
```

This will:
- Clean & process tweets  
- Compute sentiment  
- Aggregate weekly sentiment  
- Generate a heatmap  
- Build/load embeddings  
- Launch an interactive chat

## Chat Examples
```
What were people saying about TSLA in October 2021?
Why did ticker F have negative sentiment on May 23, 2022?
Summarize sentiment for AAPL last week.
```

Type `exit` or `quit` to leave.

