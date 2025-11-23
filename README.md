# Conversational Recommendation System

An interactive recommendation system that learns user preferences through natural conversation and provides personalized recommendations using multiple AI approaches.

## Features

- **Conversational Interface**: Natural language interaction to understand user preferences
- **Multiple Recommendation Models**: 
  - Popularity-based recommendations
  - Content-based filtering using TF-IDF
  - Collaborative filtering with matrix factorization
  - Hybrid approach combining all methods
- **Interactive Demo**: Streamlit web interface for easy testing
- **Comprehensive Evaluation**: Multiple metrics including Precision@K, Recall@K, NDCG@K, MAP@K
- **Synthetic Data Generation**: Realistic datasets for testing and demonstration
- **Modern Architecture**: Clean, modular code with type hints and documentation

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Conversational-Recommendation-System.git
cd Conversational-Recommendation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the System

#### Streamlit Demo (Recommended)
```bash
streamlit run demo.py
```

#### Command Line Interface
```bash
python main.py --mode cli
```

#### Model Evaluation
```bash
python main.py --mode evaluate
```

## Project Structure

```
├── src/
│   ├── types.py                    # Core data structures and types
│   ├── data.py                     # Data generation and loading utilities
│   ├── conversational_system.py    # Main system implementation
│   ├── conversation/
│   │   └── engine.py              # Conversation flow management
│   ├── models/
│   │   └── recommenders.py        # Recommendation algorithms
│   └── utils/
│       └── evaluation.py          # Evaluation metrics and utilities
├── configs/                        # Configuration files
├── notebooks/                      # Jupyter notebooks for analysis
├── scripts/                        # Utility scripts
├── tests/                          # Unit tests
├── assets/                         # Static assets
├── demo.py                         # Streamlit demo application
├── main.py                         # Main CLI script
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration
└── README.md                       # This file
```

## Usage Examples

### Basic Conversation Flow

1. **Start Conversation**: The system greets the user and asks about preferences
2. **Preference Collection**: Users specify categories, price ranges, and other preferences
3. **Recommendation Generation**: System provides personalized recommendations
4. **Feedback Loop**: Users can refine preferences or ask for explanations

### Example Conversation

```
Assistant: Hello! I'm here to help you find the perfect products. Let's start by learning about your preferences.

You: I'm looking for electronics

Assistant: Perfect! I see you're interested in Electronics. What's your preferred price range? (low/medium/high)

You: medium

Assistant: Great! Here are my recommendations for you:

1. TechCorp Electronics Product 15 ($450.00) - Hybrid recommendation combining 3 models
2. TechCorp Electronics Product 8 ($320.00) - Hybrid recommendation combining 3 models
3. TechCorp Electronics Product 23 ($380.00) - Hybrid recommendation combining 3 models
```

## Configuration

### Data Generation Parameters

- `n_items`: Number of items to generate (default: 100)
- `n_users`: Number of users to generate (default: 1000)
- `n_interactions`: Number of interactions to generate (default: 10000)
- `seed`: Random seed for reproducibility (default: 42)

### Model Configuration

- **Hybrid Weights**: Popularity (20%), Content-Based (40%), Collaborative Filtering (40%)
- **Collaborative Filtering**: 50 factors, 0.01 regularization
- **Content-Based**: TF-IDF with 1000 features

## Evaluation Metrics

The system evaluates recommendations using:

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **Coverage**: Fraction of catalog items that can be recommended
- **Diversity**: Intra-list diversity of recommendations
- **Novelty**: Fraction of new items in recommendations
- **Popularity Bias**: Tendency to recommend popular items

## API Reference

### ConversationalRecommendationSystem

Main system class for managing conversations and recommendations.

#### Methods

- `generate_synthetic_data(n_items, n_users, n_interactions)`: Generate synthetic dataset
- `initialize_recommenders()`: Fit all recommendation models
- `start_conversation(user_id)`: Start new conversation session
- `process_user_input(user_input)`: Process user message and return response
- `evaluate_models(test_size)`: Evaluate all models and return metrics
- `get_conversation_summary()`: Get current conversation summary

### Recommendation Models

#### BaseRecommender
Abstract base class for all recommendation models.

#### PopularityRecommender
Recommends items based on overall popularity.

#### ContentBasedRecommender
Uses TF-IDF and cosine similarity for content-based recommendations.

#### CollaborativeFilteringRecommender
Matrix factorization approach using implicit feedback.

#### HybridRecommender
Combines multiple recommendation approaches with weighted scoring.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
ruff check src/
```

### Type Checking

```bash
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with modern Python libraries including scikit-learn, pandas, and Streamlit
- Uses implicit library for collaborative filtering
- Inspired by conversational AI and recommendation system research
# Conversational-Recommendation-System
