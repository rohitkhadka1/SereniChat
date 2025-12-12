# Mental Health Chatbot - End-to-End Medical Chatbot System

A comprehensive AI-powered mental health chatbot system built with LangChain, Pinecone vector database, and HuggingFace embeddings. This project provides intelligent mental health support using retrieval-augmented generation (RAG) with document embeddings.

## Features

- 🤖 AI-powered mental health conversations
- 📚 RAG (Retrieval-Augmented Generation) with medical knowledge base
- 🔍 Vector similarity search using Pinecone
- 🧠 HuggingFace embeddings (all-MiniLM-L6-v2)
- 💬 Flask-based web interface
- 🚀 Scalable architecture with Flask-Limiter
- 📊 PDF document processing and embedding
- 🔐 Production-ready with security features

## System Requirements

- **Python**: 3.9+
- **OS**: Windows/Linux/macOS
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 5GB+ for models and dependencies

## Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd End-to-End-Medical-Chatbot
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv chatbot
chatbot\Scripts\activate

# Linux/macOS
python3 -m venv chatbot
source chatbot/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the project root:
```env
# API Keys
PINECONE_API_KEY=your_pinecone_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key (optional)
ELEVEN_API_KEY=your_elevenlabs_api_key (optional)

# Pinecone Configuration
PINECONE_ENVIRONMENT=us-west-2
PINECONE_INDEX_NAME=mental-health-chatbot

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=False
SECRET_KEY=your-secret-key-here
```

## Quick Start

### 1. Initialize Embeddings
```bash
python
>>> from embeddings import create_embeddings
>>> vector_store = create_embeddings()
```

### 2. Run the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

### 3. Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
├── app.py                          # Flask application entry point
├── embeddings.py                   # Vector embedding and Pinecone integration
├── src/
│   ├── helper.py                   # Utility functions
│   ├── prompt.py                   # LLM prompts
│   └── a-manual-of-mental-health-care-in-general-practice.pdf
├── templates/
│   └── chat.html                   # Web interface
├── research/
│   └── trails.ipynb               # Research and experimentation
├── requirements.txt                # Python dependencies
├── setup.py                        # Package configuration
├── .env.example                    # Environment variables template
└── README.md                       # This file
```

## Dependencies

### Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| langchain | >=0.2.16 | LLM orchestration framework |
| langchain-community | >=0.2.16 | Community integrations |
| langchain-pinecone | >=0.2.3 | Pinecone vector store integration |
| langchain-huggingface | 0.0.3 | HuggingFace embeddings |
| pinecone-client | >=3.2.2 | Pinecone vector database client |
| sentence-transformers | 2.7.0 | Text embeddings |
| transformers | 4.44.2 | HuggingFace transformers |
| torch | >=2.0.0 | PyTorch deep learning framework |
| pypdf | 4.3.1 | PDF document processing |
| flask | 3.0.3 | Web framework |

### Optional Dependencies
- **flask-cors**: Cross-origin resource sharing
- **flask-limiter**: Rate limiting
- **python-dotenv**: Environment variable management

## Configuration

### Environment Variables
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PINECONE_API_KEY` | Yes | - | Pinecone API key |
| `PINECONE_ENVIRONMENT` | Yes | - | Pinecone environment (e.g., us-west-2) |
| `PINECONE_INDEX_NAME` | No | mental-health-chatbot | Vector index name |
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key for LLM |
| `HUGGINGFACEHUB_API_TOKEN` | No | - | HuggingFace hub token |
| `FLASK_DEBUG` | No | False | Flask debug mode |
| `SECRET_KEY` | Yes | - | Flask secret key |

### Embeddings Configuration

The application uses `all-MiniLM-L6-v2` model for embeddings:
- **Dimension**: 384
- **Model Size**: ~22MB
- **Speed**: Fast inference
- **Accuracy**: Good balance for semantic search

## Usage

### Basic Chat Interaction
```bash
# Start the Flask server
python app.py

# Open web browser to http://localhost:5000
# Type your mental health question and get AI-powered responses
```

### Programmatic Usage
```python
from embeddings import load_embeddings, get_similar_docs

# Load the vector store
vector_store = load_embeddings()

# Search for similar documents
query = "How to manage anxiety?"
similar_docs = get_similar_docs(query, vector_store, k=3)

# Use in your application
for doc in similar_docs:
    print(doc.page_content)
```

## API Endpoints

### GET `/`
Returns the chat interface

### POST `/chat`
Sends a message and gets a response
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How to manage stress?"}'
```

### GET `/health`
Health check endpoint
```bash
curl http://localhost:5000/health
```

## Troubleshooting

### Import Errors
If you encounter import errors:
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Pinecone Connection Issues
```python
# Check Pinecone credentials
from embeddings import init_pinecone
pc = init_pinecone()
print(pc.list_indexes())
```

### Sentence Transformers Issues
```bash
# Reinstall with specific version
pip install sentence-transformers==2.7.0 --force-reinstall
```

### PDF Loading Errors
- Ensure the PDF file exists at: `src/a-manual-of-mental-health-care-in-general-practice.pdf`
- Check file permissions
- Verify PDF is not corrupted

## Performance Optimization

### Chunk Size Configuration
Edit `embeddings.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Adjust based on your needs
    chunk_overlap=50,
    length_function=len
)
```

### Model Configuration
For better performance on CPU:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
```

For GPU acceleration:
```python
model_kwargs={'device': 'cuda'}  # If CUDA is available
```

## Security

### Production Deployment Checklist
- [ ] Set `FLASK_DEBUG=False`
- [ ] Generate strong `SECRET_KEY`
- [ ] Store API keys in environment variables (never in code)
- [ ] Use HTTPS in production
- [ ] Enable rate limiting with Flask-Limiter
- [ ] Configure proper CORS origins
- [ ] Implement input validation
- [ ] Use environment-specific `.env` files
- [ ] Keep dependencies updated
- [ ] Monitor logs for errors

### Rate Limiting
The application includes Flask-Limiter with default limits:
- 30 requests/minute per IP
- 200 requests/day per IP
- Configurable in `app.py`

## Deployment

### Docker Deployment (if docker files available)
```bash
docker-compose up -d
docker-compose logs -f
```

### Cloud Deployment
Tested on:
- Heroku
- AWS EC2
- Google Cloud Platform
- Azure App Service

## Development

### Running Tests
```bash
# Create test file and run
pytest tests/
```

### Code Style
- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions
- Include type hints

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependency | `pip install -r requirements.txt` |
| Pinecone connection fails | Wrong API key | Verify `PINECONE_API_KEY` in `.env` |
| Embeddings slow | CPU limitations | Use GPU or reduce chunk size |
| PDF not found | Wrong path | Check file path in `embeddings.py` |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs in console output
3. Check existing GitHub issues
4. Create a new issue with detailed description

## Acknowledgments

- LangChain for the orchestration framework
- Pinecone for vector database
- HuggingFace for pre-trained models
- OpenRouter for LLM access
