"""Main script for the conversational recommendation system."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.conversational_system import ConversationalRecommendationSystem


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Conversational Recommendation System")
    parser.add_argument("--mode", choices=["demo", "cli", "evaluate"], default="cli",
                       help="Mode to run the system")
    parser.add_argument("--items", type=int, default=100,
                       help="Number of items to generate")
    parser.add_argument("--users", type=int, default=1000,
                       help="Number of users to generate")
    parser.add_argument("--interactions", type=int, default=10000,
                       help="Number of interactions to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Initialize system
    print("Initializing Conversational Recommendation System...")
    system = ConversationalRecommendationSystem(seed=args.seed)
    
    # Generate data
    print("Generating synthetic data...")
    system.generate_synthetic_data(
        n_items=args.items,
        n_users=args.users,
        n_interactions=args.interactions
    )
    
    # Initialize models
    print("Initializing recommendation models...")
    system.initialize_recommenders()
    
    if args.mode == "demo":
        print("Starting Streamlit demo...")
        import subprocess
        subprocess.run(["streamlit", "run", "demo.py"])
    
    elif args.mode == "evaluate":
        print("Evaluating models...")
        results = system.evaluate_models()
        print("\n" + "="*50)
        print(results['report'])
        print("="*50)
    
    elif args.mode == "cli":
        print("Starting CLI interface...")
        run_cli_interface(system)


def run_cli_interface(system: ConversationalRecommendationSystem):
    """Run command-line interface."""
    print("\nWelcome to the Conversational Recommendation System!")
    print("Type 'quit' to exit, 'help' for commands.\n")
    
    user_id = input("Enter your user ID (or press Enter for 'demo_user'): ").strip()
    if not user_id:
        user_id = "demo_user"
    
    # Start conversation
    greeting = system.start_conversation(user_id)
    print(f"\nAssistant: {greeting}")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nAssistant: Thank you for using our recommendation system. Goodbye!")
            break
        
        if user_input.lower() == 'help':
            print("\nAvailable commands:")
            print("- quit/exit/bye: Exit the system")
            print("- help: Show this help message")
            print("- summary: Show conversation summary")
            print("- evaluate: Run model evaluation")
            continue
        
        if user_input.lower() == 'summary':
            summary = system.get_conversation_summary()
            if summary:
                print(f"\nConversation Summary:")
                print(f"- Session ID: {summary['session_id']}")
                print(f"- Turns: {summary['turns']}")
                print(f"- Duration: {summary['duration']:.2f} seconds")
                print(f"- Preferences: {summary['preferences']}")
            continue
        
        if user_input.lower() == 'evaluate':
            print("\nEvaluating models...")
            results = system.evaluate_models()
            print("\n" + results['report'])
            continue
        
        # Process user input
        try:
            system_response, is_complete, recommendations = system.process_user_input(user_input)
            print(f"\nAssistant: {system_response}")
            
            if recommendations:
                print(f"\nRecommendations:")
                for i, rec in enumerate(recommendations[:5], 1):
                    item = system._get_item_by_id(rec.item_id)
                    if item:
                        print(f"{i}. {item.title} (${item.price:.2f}) - {rec.explanation}")
            
            if is_complete:
                print("\nConversation completed. Starting new conversation...")
                greeting = system.start_conversation(user_id)
                print(f"\nAssistant: {greeting}")
        
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'help' for available commands.")


if __name__ == "__main__":
    main()
