"""
Project 349: Conversational Recommendation System

This is a simple example of the original conversational recommendation system.
For the full modernized version with advanced features, see the main.py script
and the src/ directory with the complete implementation.

The modernized version includes:
- Multiple recommendation algorithms (popularity, content-based, collaborative filtering, hybrid)
- Comprehensive evaluation metrics
- Interactive Streamlit demo
- Synthetic data generation
- Type hints and modern Python practices
- Modular architecture
- Unit tests

To run the full system:
    python main.py --mode cli          # Command line interface
    streamlit run demo.py              # Web interface
    python main.py --mode evaluate     # Model evaluation
"""

import random


def simple_conversational_recommender():
    """Simple conversational recommendation system (original implementation)."""
    
    # 1. Simulate items and user preferences
    items = ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch']
    item_details = {
        'Laptop': {'category': 'Electronics', 'price': 1000, 'brand': 'BrandA'},
        'Smartphone': {'category': 'Electronics', 'price': 800, 'brand': 'BrandB'},
        'Headphones': {'category': 'Electronics', 'price': 150, 'brand': 'BrandC'},
        'Tablet': {'category': 'Electronics', 'price': 500, 'brand': 'BrandA'},
        'Smartwatch': {'category': 'Electronics', 'price': 200, 'brand': 'BrandB'}
    }
    
    # 2. Define the conversational agent
    class ConversationalRecommender:
        def __init__(self, items, item_details):
            self.items = items
            self.item_details = item_details
        
        def ask_question(self, question):
            print(question)
            response = input("Your answer: ").strip().lower()
            return response
        
        def recommend(self):
            print("Hello! I'm here to help you find the best products.")
            
            # Ask about category preference
            category_response = self.ask_question("What category of product are you interested in? (e.g., Electronics)")
            
            # Ask about price range preference
            price_response = self.ask_question("What's your preferred price range? (e.g., low, medium, high)")
            
            # Filter items based on category and price
            recommended_items = []
            for item in self.items:
                details = self.item_details[item]
                
                # Filtering based on category
                if category_response in details['category'].lower():
                    # Filtering based on price
                    if price_response == 'low' and details['price'] < 300:
                        recommended_items.append(item)
                    elif price_response == 'medium' and 300 <= details['price'] <= 800:
                        recommended_items.append(item)
                    elif price_response == 'high' and details['price'] > 800:
                        recommended_items.append(item)
            
            if not recommended_items:
                print("Sorry, I couldn't find any products matching your preferences.")
            else:
                print(f"Based on your preferences, I recommend the following products: {', '.join(recommended_items)}")
    
    # 3. Create and interact with the recommender
    recommender = ConversationalRecommender(items, item_details)
    recommender.recommend()


if __name__ == "__main__":
    print("Simple Conversational Recommendation System")
    print("=" * 50)
    print("This is the original simple implementation.")
    print("For the full modernized system, run: python main.py")
    print("=" * 50)
    
    simple_conversational_recommender()