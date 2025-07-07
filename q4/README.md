# 🧠 Smart Article Categorizer

A comprehensive system that automatically classifies news articles into categories using four different embedding approaches. The system includes a FastAPI backend and a Streamlit frontend for easy interaction.

## 🎯 Features

- **Multiple Embedding Models**: GloVe, BERT, Sentence-BERT, and Gemini
- **Real-time Classification**: Predict article categories with confidence scores
- **Interactive UI**: Beautiful Streamlit interface for training and prediction
- **Visualization**: UMAP 2D visualization of embedding clusters
- **Comprehensive Logging**: Request and error logging for monitoring
- **Model Comparison**: Side-by-side performance metrics

## 📊 Categories

The system classifies articles into 5 categories:
- **Tech**: Technology-related articles
- **Finance**: Business and financial news (mapped from "business")
- **Sport**: Sports news and updates
- **Politics**: Political news and analysis
- **Entertainment**: Entertainment and celebrity news

## 🔧 Embedding Models

### 1. GloVe (Global Vectors)
- Uses pre-trained GloVe 6B 300d vectors
- Averages word vectors for document representation
- Automatically downloads vectors if not present

### 2. BERT (Bidirectional Encoder Representations)
- Uses `bert-base-uncased` model
- Extracts [CLS] token from the last hidden state
- Processes in batches for memory efficiency

### 3. Sentence-BERT
- Uses `sentence-transformers/all-MiniLM-L6-v2`
- Direct sentence-level embeddings
- Optimized for semantic similarity

### 4. Gemini Text Embeddings
- Uses Google's `text-embedding-004` model
- Requires API key configuration
- Rate-limited API calls

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd smart-article-categorizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (for Gemini model)
```bash
# Create .env file or set environment variable
export GEMINI_API_KEY="your-gemini-api-key"
```

### Running the Application

1. **Start the FastAPI backend**
```bash
cd backend
python main.py
```
The API will be available at `http://localhost:8000`

2. **Start the Streamlit frontend** (in a new terminal)
```bash
cd frontend
streamlit run app.py
```
The web interface will be available at `http://localhost:8501`

## 📱 Usage

### Web Interface

1. **Home Page**: View dataset statistics and project overview
2. **Model Training**: Train individual models and view performance metrics
3. **Article Prediction**: Enter article text and get predictions from all models
4. **Model Comparison**: Compare performance across different models
5. **UMAP Visualization**: View 2D embedding clusters

### API Endpoints

- `GET /`: API information
- `GET /health`: Health check
- `GET /models`: List available models
- `GET /dataset/stats`: Dataset statistics
- `POST /train`: Train a specific model
- `POST /predict`: Predict article category
- `GET /umap/{model_name}`: Get UMAP visualization data
- `GET /models/status`: Get training status of all models

### Example API Usage

```python
import requests

# Train a model
response = requests.post("http://localhost:8000/train", json={"model": "bert"})

# Make prediction
response = requests.post("http://localhost:8000/predict", json={
    "model": "bert",
    "text": "Apple announces new iPhone with revolutionary features..."
})
```

## 📁 Project Structure

```
smart-article-categorizer/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # FastAPI routes
│   ├── config/
│   │   ├── __init__.py
│   │   ├── logging_config.py  # Logging configuration
│   │   └── settings.py        # Application settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_embedder.py   # Base embedder class
│   │   ├── bert_embedder.py   # BERT implementation
│   │   ├── gemini_embedder.py # Gemini implementation
│   │   ├── glove_embedder.py  # GloVe implementation
│   │   ├── model_factory.py   # Model factory
│   │   └── sbert_embedder.py  # Sentence-BERT implementation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Data loading utilities
│   │   └── visualization.py   # UMAP visualization
│   ├── dataset.csv            # Training dataset
│   └── main.py               # FastAPI application entry point
├── frontend/
│   └── app.py                # Streamlit application
├── logs/                     # Application logs (created automatically)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔍 Logging

The application includes comprehensive logging:

- **Request Logs**: All API requests with timestamps and IP addresses
- **Error Logs**: Detailed error information with stack traces
- **Model Logs**: Training progress and prediction events
- **Rotating Logs**: Automatic log rotation to prevent disk space issues

Logs are stored in the `logs/` directory:
- `app.log`: General application logs
- `errors.log`: Error-specific logs

## 🎛️ Configuration

### Model Settings

Edit `backend/config/settings.py` to modify:
- Model configurations (embedding dimensions, model names)
- Training parameters (test size, random state)
- Text preprocessing settings
- UMAP visualization parameters

### API Settings

- Host and port configuration
- CORS settings
- Request timeout settings

## 🧪 Model Performance

Each model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision per class and weighted average
- **Recall**: Recall per class and weighted average
- **F1-Score**: F1-score per class and weighted average
- **Classification Report**: Detailed per-class metrics

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch sizes in model configurations
2. **Gemini API Errors**: Check API key and rate limits
3. **Model Download Failures**: Ensure stable internet connection
4. **Port Already in Use**: Change ports in configuration files

### Performance Tips

1. **Use GPU**: Install CUDA-compatible PyTorch for faster training
2. **Increase Batch Size**: For systems with more memory
3. **Parallel Processing**: The system uses ThreadPoolExecutor for concurrent operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **GloVe**: Stanford NLP Group
- **BERT**: Google AI Language
- **Sentence-BERT**: UKP Lab
- **Gemini**: Google AI
- **UMAP**: Leland McInnes et al.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in the `logs/` directory
3. Open an issue on the repository

---

**Happy Categorizing! 🎉** 