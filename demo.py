"""Streamlit demo for the conversational recommendation system."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import time

from src.conversational_system import ConversationalRecommendationSystem
from src.types import Recommendation


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'system' not in st.session_state:
        st.session_state.system = ConversationalRecommendationSystem()
        st.session_state.system.generate_synthetic_data(n_items=50, n_users=500, n_interactions=2000)
        st.session_state.system.initialize_recommenders()
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = "demo_user"
    
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []


def display_item_details(item_id: str, system: ConversationalRecommendationSystem):
    """Display detailed information about an item."""
    item = system._get_item_by_id(item_id)
    if not item:
        return
    
    with st.expander(f"Item Details: {item.title}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Category:** {item.category}")
            st.write(f"**Brand:** {item.brand}")
            st.write(f"**Price:** ${item.price:.2f}")
            st.write(f"**Description:** {item.description}")
        
        with col2:
            st.write("**Features:**")
            for feature, value in item.features.items():
                st.write(f"- {feature}: {value}")
            
            st.write("**Tags:**")
            for tag in item.tags:
                st.write(f"- {tag}")


def display_recommendations(recommendations: List[Recommendation], system: ConversationalRecommendationSystem):
    """Display recommendations in an interactive format."""
    if not recommendations:
        st.info("No recommendations available yet. Please continue the conversation.")
        return
    
    st.subheader("Recommendations")
    
    for i, rec in enumerate(recommendations[:5], 1):
        item = system._get_item_by_id(rec.item_id)
        if not item:
            continue
        
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{i}. {item.title}**")
                st.write(f"Category: {item.category} | Price: ${item.price:.2f}")
                st.write(f"*{rec.explanation}*")
            
            with col2:
                st.metric("Score", f"{rec.score:.3f}")
            
            with col3:
                if st.button(f"Details", key=f"details_{i}"):
                    display_item_details(rec.item_id, system)
            
            st.divider()


def display_conversation_history():
    """Display the conversation history."""
    if not st.session_state.conversation_history:
        return
    
    st.subheader("Conversation History")
    
    for i, (user_msg, system_msg) in enumerate(st.session_state.conversation_history):
        with st.container():
            # User message
            st.write(f"**You:** {user_msg}")
            
            # System message
            st.write(f"**Assistant:** {system_msg}")
            
            if i < len(st.session_state.conversation_history) - 1:
                st.divider()


def display_model_evaluation():
    """Display model evaluation results."""
    st.subheader("Model Performance")
    
    if st.button("Evaluate Models"):
        with st.spinner("Evaluating models..."):
            results = st.session_state.system.evaluate_models()
            
            st.text(results['report'])
            
            # Create visualization
            df = st.session_state.system.evaluator.compare_models(results['model_results'])
            
            # Plot metrics
            metrics_to_plot = ['Precision@5', 'Recall@5', 'NDCG@5', 'MAP@5', 'HitRate@5']
            
            fig = go.Figure()
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df['Model'],
                    y=df[metric]
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Models",
                yaxis_title="Score",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)


def display_data_overview():
    """Display overview of the dataset."""
    st.subheader("Dataset Overview")
    
    system = st.session_state.system
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Items", len(system.items))
    
    with col2:
        st.metric("Total Users", len(system.users))
    
    with col3:
        st.metric("Total Interactions", len(system.interactions))
    
    # Category distribution
    categories = [item.category for item in system.items]
    category_counts = pd.Series(categories).value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Item Distribution by Category"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Price distribution
    prices = [item.price for item in system.items]
    fig = px.histogram(
        x=prices,
        title="Price Distribution",
        labels={'x': 'Price ($)', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Conversational Recommendation System",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("Conversational Recommendation System")
    st.markdown("An interactive AI-powered recommendation system that learns your preferences through conversation.")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page",
            ["Chat Interface", "Data Overview", "Model Evaluation", "Settings"]
        )
    
    if page == "Chat Interface":
        # Main chat interface
        st.header("Chat with the Recommendation Assistant")
        
        # Start conversation button
        if not st.session_state.conversation_started:
            if st.button("Start Conversation"):
                greeting = st.session_state.system.start_conversation(st.session_state.current_user)
                st.session_state.conversation_history.append(("", greeting))
                st.session_state.conversation_started = True
                st.rerun()
        else:
            # Display conversation history
            display_conversation_history()
            
            # User input
            user_input = st.text_input(
                "Type your message:",
                placeholder="Tell me what you're looking for...",
                key="user_input"
            )
            
            if st.button("Send") and user_input:
                # Process user input
                system_response, is_complete, recommendations = st.session_state.system.process_user_input(user_input)
                
                # Update conversation history
                st.session_state.conversation_history.append((user_input, system_response))
                
                # Update recommendations
                if recommendations:
                    st.session_state.recommendations = recommendations
                
                # Clear input
                st.session_state.user_input = ""
                st.rerun()
            
            # Display recommendations
            if st.session_state.recommendations:
                display_recommendations(st.session_state.recommendations, st.session_state.system)
            
            # Conversation summary
            summary = st.session_state.system.get_conversation_summary()
            if summary:
                with st.expander("Conversation Summary"):
                    st.write(f"**Session ID:** {summary['session_id']}")
                    st.write(f"**Turns:** {summary['turns']}")
                    st.write(f"**Duration:** {summary['duration']:.2f} seconds")
                    st.write(f"**Extracted Preferences:** {summary['preferences']}")
    
    elif page == "Data Overview":
        display_data_overview()
    
    elif page == "Model Evaluation":
        display_model_evaluation()
    
    elif page == "Settings":
        st.header("Settings")
        
        st.subheader("System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Configuration:**")
            st.write(f"- Items: {len(st.session_state.system.items)}")
            st.write(f"- Users: {len(st.session_state.system.users)}")
            st.write(f"- Interactions: {len(st.session_state.system.interactions)}")
        
        with col2:
            st.write("**Actions:**")
            if st.button("Regenerate Data"):
                st.session_state.system.generate_synthetic_data()
                st.session_state.system.initialize_recommenders()
                st.success("Data regenerated!")
            
            if st.button("Reset Conversation"):
                st.session_state.conversation_history = []
                st.session_state.conversation_started = False
                st.session_state.recommendations = []
                st.success("Conversation reset!")
        
        st.subheader("Model Information")
        st.write("**Available Models:**")
        for name in st.session_state.system.recommenders.keys():
            st.write(f"- {name.title()}")
        
        st.write("**Model Weights (Hybrid):**")
        st.write("- Popularity: 20%")
        st.write("- Content-Based: 40%")
        st.write("- Collaborative Filtering: 40%")


if __name__ == "__main__":
    main()
