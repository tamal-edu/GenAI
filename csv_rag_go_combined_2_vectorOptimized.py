import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
import os
import pickle
import numpy as np

# Load environment variables
load_dotenv()

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Initialize sentiment analyzers
sia = SentimentIntensityAnalyzer()
sentiment_pipeline = pipeline("sentiment-analysis", model="./model/")
# distilbert/distilbert-base-uncased-finetuned-sst-2-english

# Load and preprocess data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if len(df) > 10000:  # Adjust this threshold as needed
        df = df.sample(10000, random_state=42)
    df['Request Open Date'] = pd.to_datetime(df['Request Open Date'])
    df['Request Closed Date'] = pd.to_datetime(df['Request Closed Date'])
    df['Resolution Time'] = (df['Request Closed Date'] - df['Request Open Date']).dt.total_seconds() / 3600  # in hours
    return df

# Create vector store (updated version)
@st.cache_resource
def create_vector_store(df):
    cache_file = "embeddings_cache.pkl"
    
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
            texts = cached_data['texts']
            embeddings_list = cached_data['embeddings']
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(df.to_string())
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        embeddings_list = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = embeddings.embed_documents(batch)
            embeddings_list.extend(batch_embeddings)
            # Update progress
            progress = min((i + batch_size) / len(texts), 1.0)
            progress_bar.progress(progress)
        
        # Ensure progress bar reaches 100%
        progress_bar.progress(1.0)
        
        # Cache the texts and embeddings
        with open(cache_file, "wb") as f:
            pickle.dump({'texts': texts, 'embeddings': embeddings_list}, f)
    
    # Create FAISS index
    embeddings_array = np.array(embeddings_list)
    vector_store = FAISS.from_embeddings(zip(texts, embeddings_array), GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    
    return vector_store

# Initialize Gemini model
@st.cache_resource
def initialize_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

# Data Overview page
def data_overview(df):
    st.header('Data Overview')
    st.dataframe(df)

    # Basic Statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())

    # Service Satisfaction Distribution
    st.subheader("Service Satisfaction Distribution")
    fig = px.histogram(df, x="Service Satisfaction", nbins=5, title="Distribution of Service Satisfaction Ratings")
    st.plotly_chart(fig)

    # Company distribution
    st.subheader('Requests by Company')
    company_counts = df['Company Name'].value_counts()
    fig = px.pie(values=company_counts.values, names=company_counts.index)
    st.plotly_chart(fig)

# Detailed Analysis page
def detailed_analysis(df):
    st.header('Detailed Analysis')

    # Perform sentiment analysis
    df['Sentiment'] = df['Overall Comment'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
    fig = px.histogram(df, x="Sentiment", nbins=20, title="Distribution of Sentiment Scores")
    st.plotly_chart(fig)

    # Correlation between Service Satisfaction and Sentiment
    st.subheader("Correlation: Service Satisfaction vs Sentiment")
    fig = px.scatter(df, x="Service Satisfaction", y="Sentiment", trendline="ols", 
                     title="Service Satisfaction vs Sentiment Score")
    st.plotly_chart(fig)

    # Company-wise Analysis
    st.subheader("Company-wise Analysis")
    company_stats = df.groupby("Company Name").agg({
        "Service Satisfaction": "mean",
        "Sentiment": "mean",
        "Resolution Time": "mean"
    }).reset_index()
    fig = px.bar(company_stats, x="Company Name", y=["Service Satisfaction", "Sentiment"], 
                 title="Average Service Satisfaction and Sentiment by Company")
    st.plotly_chart(fig)

    # Resolution Time Analysis
    st.subheader("Resolution Time Analysis")
    fig = px.box(df, x="Company Name", y="Resolution Time", title="Resolution Time by Company")
    st.plotly_chart(fig)

    # Word Cloud
    st.subheader("Word Cloud of Customer Comments")
    text = " ".join(df["Overall Comment"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# Query System page
def query_system(vector_store, llm):
    st.header('Query System')
    
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create retrieval chain with memory
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    
    # User input
    user_question = st.text_input("Ask a question about the customer feedback:")
    if user_question:
        # Generate response
        response = qa_chain({"question": user_question})
        st.write("Answer:", response['answer'])
        
        # Update sidebar with chat history
        st.sidebar.title("Chat History")
        for i, message in enumerate(memory.chat_memory.messages):
            if i % 2 == 0:
                st.sidebar.text_area(f"Human {i//2 + 1}", value=message.content, height=50, disabled=True)
            else:
                st.sidebar.text_area(f"AI {i//2 + 1}", value=message.content, height=100, disabled=True)

# Sentiment Analysis page
def sentiment_analysis(df):
    st.header('Sentiment Analysis')
    
    if st.button('Analyze Sentiment'):
        df['sentiment'] = df['Overall Comment'].apply(lambda x: sentiment_pipeline(x)[0])
        sentiment_summary = df['sentiment'].apply(lambda x: x['label']).value_counts()
        st.write("Sentiment Summary:", sentiment_summary)

        # Create a bar chart of sentiment distribution
        sentiment_data = df['sentiment'].apply(lambda x: x['label']).value_counts().reset_index()
        sentiment_data.columns = ['Sentiment', 'Count']
        fig = px.bar(sentiment_data, x='Sentiment', y='Count', color='Sentiment',
                     labels={'Count': 'Number of Reviews'})
        st.plotly_chart(fig)

        # Display average satisfaction score
        avg_satisfaction = df['Service Satisfaction'].mean()
        st.metric("Average Satisfaction Score", f"{avg_satisfaction:.2f}/5")

        # Show a few example comments with their sentiments
        st.subheader("Sample Comments and Sentiments")
        sample_df = df.sample(min(5, len(df)))
        for _, row in sample_df.iterrows():
            st.write(f"Comment: {row['Overall Comment']}")
            st.write(f"Sentiment: {row['sentiment']['label']} (Score: {row['sentiment']['score']:.2f})")
            st.write("---")

# Main Streamlit app
def main():
    st.title("Customer Feedback Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            
            with st.spinner('Creating vector store... This may take a few minutes.'):
                vector_store = create_vector_store(df)

            # Initialize LLM
            llm = initialize_llm()

            # Sidebar for navigation
            page = st.sidebar.selectbox('Choose a page', ['Data Overview', 'Detailed Analysis', 'Query System', 'Sentiment Analysis'])

            if page == 'Data Overview':
                data_overview(df)
            elif page == 'Detailed Analysis':
                detailed_analysis(df)
            elif page == 'Query System':
                query_system(vector_store, llm)
            elif page == 'Sentiment Analysis':
                sentiment_analysis(df)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your CSV file and try again.")

    else:
        st.write("Please upload a CSV file to begin the analysis.")

if __name__ == "__main__":
    main()