# SproutML Backend API

A sophisticated machine learning training platform that orchestrates AI agents to process datasets, perform preprocessing, and execute complete ML pipelines in isolated cloud environments.

## 🚀 Overview

SproutML Backend is a FastAPI-based service that provides a complete machine learning training pipeline through intelligent agent orchestration. The system leverages multiple specialized AI agents to handle different aspects of the ML workflow, from data preprocessing to model training and evaluation.

### Key Features

- **Multi-Agent Architecture**: Orchestrates specialized AI agents for different ML tasks
- **Cloud Sandbox Execution**: Runs ML pipelines in isolated Daytona cloud environments
- **Real-time Job Tracking**: Asynchronous job processing with status monitoring
- **File Management**: Secure upload, processing, and artifact retrieval
- **RESTful API**: Clean, documented endpoints for all operations

### Technology Stack

- **Framework**: FastAPI (Python 3.13+)
- **AI Agents**: Custom agent framework with OpenAI integration
- **Cloud Execution**: Daytona sandbox environments
- **File Storage**: Persistent volumes with artifact management
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

## 📊 Brag Sheet

### 🎯 Code Quality & Architecture Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Main.py Lines** | 519 lines | 42 lines | **92% reduction** |
| **Architecture Pattern** | Monolithic | Modular Services | **Professional-grade** |
| **Separation of Concerns** | None | 4 distinct layers | **Complete separation** |

### 🏗️ Architectural Transformation

**From Monolithic to Microservices Architecture:**
- ✅ **Configuration Layer**: Centralized environment management
- ✅ **Service Layer**: 4 specialized services (Job, Daytona, Agent, Training)
- ✅ **Route Layer**: 5 focused API modules with clean separation
- ✅ **Main Application**: Minimal orchestration layer (42 lines)

### 🎨 Design Patterns Implemented

1. **Service Layer Pattern**: Business logic separated from API concerns
2. **Repository Pattern**: Clean data access through service abstractions
3. **Factory Pattern**: Agent creation with dependency injection
4. **Router Pattern**: FastAPI APIRouter for modular endpoint organization
5. **Configuration Pattern**: Environment-based configuration management

### 📈 Scalability & Maintainability Achievements

- **Modularity**: Each component can be developed, tested, and deployed independently
- **Testability**: Individual services can be unit tested in isolation
- **Extensibility**: New agents, services, or routes can be added without affecting existing code
- **Team Collaboration**: Multiple developers can work on different modules simultaneously
- **Code Reusability**: Services can be imported and used across different parts of the application

### 🔧 Technical Excellence

- **Error Handling**: Comprehensive exception handling across all layers
- **Type Safety**: Full type hints throughout the codebase
- **Documentation**: Auto-generated API documentation with detailed schemas
- **CORS Configuration**: Production-ready cross-origin resource sharing
- **Async Processing**: Non-blocking job execution with proper resource management

## 🏛️ Architecture Overview

```
sproutml-back/
├── main.py                    # Application entry point (42 lines)
├── config.py                  # Environment configuration
├── services/                  # Business logic layer
│   ├── job_service.py         # Job management (49 lines)
│   ├── daytona_service.py     # Cloud sandbox management (81 lines)
│   ├── agent_service.py       # AI agent orchestration (152 lines)
│   └── process_training_job.py # Training pipeline (147 lines)
└── routes/                    # API endpoint layer
    ├── __init__.py           # Package interface (6 lines)
    ├── training.py           # Training endpoints (69 lines)
    ├── jobs.py               # Job management endpoints (51 lines)
    ├── artifacts.py          # File management endpoints (31 lines)
    └── health.py             # Health check endpoints (24 lines)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- OpenAI API Key
- Daytona API Key

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Run the application:
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## 📚 API Endpoints

### Training
- `POST /train` - Start a new ML training job
- `GET /health` - Health check endpoint

### Job Management
- `GET /jobs` - List all training jobs
- `GET /job/{job_id}` - Get specific job status

### Artifacts
- `GET /job/{job_id}/artifacts` - List job artifacts
- `GET /job/{job_id}/artifact/{filename}` - Download specific artifact

## 🤖 Agent Architecture

The system uses a sophisticated multi-agent architecture:

1. **Orchestrator Agent**: Coordinates the entire ML pipeline
2. **Preprocessing Agent**: Handles data cleaning and preparation
3. **Master Trainer Agent**: Manages model training processes
4. **Evaluator Agent**: Performs model evaluation and validation
5. **Tuning Agent**: Optimizes hyperparameters

## 🔄 Job Processing Flow

1. **Upload**: User uploads CSV dataset with target column specification
2. **Job Creation**: System creates persistent job record with unique ID
3. **Environment Setup**: Daytona sandbox and volume are provisioned
4. **Agent Orchestration**: Orchestrator coordinates specialized agents
5. **Pipeline Execution**: Multi-step ML pipeline runs in isolated environment
6. **Artifact Generation**: Results, models, and reports are stored
7. **Status Tracking**: Real-time job status updates throughout process

## 🛡️ Security & Isolation

- **Sandboxed Execution**: All ML processing runs in isolated cloud environments
- **Persistent Storage**: Secure file storage with access controls
- **API Authentication**: Ready for token-based authentication
- **CORS Configuration**: Production-ready cross-origin policies

## 📊 Performance Features

- **Asynchronous Processing**: Non-blocking job execution
- **Resource Management**: Efficient sandbox lifecycle management
- **File Streaming**: Optimized file upload/download handling
- **Concurrent Execution**: Thread pool for parallel job processing

## 🧪 Development & Testing

The modular architecture enables:
- **Unit Testing**: Each service can be tested independently
- **Integration Testing**: API endpoints can be tested in isolation
- **Mock Services**: Easy to mock external dependencies
- **Development Workflow**: Hot reloading and rapid iteration

## 🚀 Deployment Ready

- **Docker Support**: Containerized deployment configuration
- **Environment Configuration**: Production-ready environment management
- **Health Checks**: Comprehensive monitoring endpoints
- **Logging**: Structured logging throughout the application

## 📈 Future Enhancements

- **Database Integration**: Replace in-memory storage with persistent database
- **Authentication**: JWT-based user authentication
- **Monitoring**: Advanced metrics and observability
- **Scaling**: Horizontal scaling with load balancing
- **Caching**: Redis integration for improved performance

## 🤝 Contributing

The modular architecture makes contribution easy:
1. Pick a specific service or route module
2. Implement your feature in isolation
3. Add comprehensive tests
4. Submit pull request with clear documentation

## 📄 License

This project is part of the SproutML platform. See LICENSE file for details.

---

**Built with ❤️ using FastAPI, AI Agents, and Cloud Computing**
