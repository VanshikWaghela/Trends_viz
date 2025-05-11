# Reddit Trend Analysis Dashboard

An interactive dashboard for analyzing trends and anomalies in Reddit discourse, with a focus on detecting and understanding unusual spikes in activity.

**Live Demo:** [https://surge-ll4j.onrender.com](https://surge-ll4j.onrender.com)

## Features

### 1. Story Overview
- Narrative presentation of the information surge with context
- Timeline visualization with key events
- Summary of main findings
<img width="1512" alt="Screenshot 2025-05-12 at 1 58 22 AM" src="https://github.com/user-attachments/assets/0cd0da50-234f-4b84-a113-8b3dfe4d6605" />
<img width="1512" alt="Screenshot 2025-05-12 at 1 58 29 AM" src="https://github.com/user-attachments/assets/2995e111-4a87-4ca0-95e4-7397adea171d" />

### 2. Overview Analysis
- Time series of post volume with keyword filtering
- Top subreddits, authors and domains
-Content word clouds
<img width="1286" alt="Screenshot 2025-05-12 at 2 13 22 AM" src="https://github.com/user-attachments/assets/1c7820f5-3dab-4483-899e-3bfcb4fcdd67" />


### 3. Spike Analysis
- Comparative analysis of abnormal vs baseline periods
- Author and subreddit activity changes during the spike
- Domain distribution changes
<img width="1512" alt="Screenshot 2025-05-12 at 1 58 49 AM" src="https://github.com/user-attachments/assets/00e786fe-3fe9-4019-a5ac-8bb1db6b4418" />

### 4. Advanced Analysis
- Topic modeling with LDA and NMF algorithms
- Engagement amplification factor calculation 
- Content similarity detection
<img width="1512" alt="Screenshot 2025-05-12 at 1 59 04 AM" src="https://github.com/user-attachments/assets/6a1edd65-bcd1-4528-bdc2-5fc7b07e9b1f" />
<img width="1512" alt="Screenshot 2025-05-12 at 1 58 56 AM" src="https://github.com/user-attachments/assets/e2653db2-a80c-4dd1-8081-73b40a46b98b" />

### 5. Network Analysis
- Author-topic network visualization
- Community detection via graph clustering
- Coordination detection between authors
<img width="1512" alt="Screenshot 2025-05-12 at 1 59 18 AM" src="https://github.com/user-attachments/assets/35968500-aa5e-4d69-b2cd-c0c6312e10ce" />
<img width="1512" alt="Screenshot 2025-05-12 at 1 59 13 AM" src="https://github.com/user-attachments/assets/9bd950d7-a8be-4490-8f14-14a65be20730" />

### 6. AI Insights
- AI-generated trend summaries
- Data-aware chatbot for answering questions
- Predictive engagement analysis
  <img width="1512" alt="Screenshot 2025-05-12 at 1 59 31 AM" src="https://github.com/user-attachments/assets/71827095-c63b-41d3-82db-49e8677cdc60" />

## Setup Instructions

1. Ensure you have Python 3.9+ installed
2. Clone this repository
3. Create and activate a virtual environment (recommended):

```bash
# Create environment
conda create -n myenv python=3.9
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

4. Set up your Gemini API key (required for AI-powered features):

```bash
# For Linux/Mac
export GOOGLE_API_KEY=your_api_key_here

# For Windows (Command Prompt)
set GOOGLE_API_KEY=your_api_key_here

# For Windows (PowerShell)
$env:GOOGLE_API_KEY="your_api_key_here"
```

You can obtain a Gemini API key from [Google AI Studio](https://ai.google.dev/).

5. Run the application:

```bash
python app.py
```

6. Navigate to http://127.0.0.1:8050/ in your web browser to view the dashboard


## LLM-Powered Features

The dashboard includes several AI-powered features in the "AI Insights" tab:

1. **AI-Generated Trend Summary**: Get a concise executive summary of key trends and insights from the dataset
2. **Insights Chatbot**: Ask questions about the data and analysis to receive context-aware answers
3. **Predictive Analysis**: Predict engagement levels for hypothetical Reddit posts with AI-powered analysis

These features require a valid Gemini API key. The dashboard uses Google's `gemini-2.5-flash-preview-04-17` model for optimal performance.

## Development

This dashboard is built with:
- Dash and Plotly for interactive visualizations
- Dash Bootstrap Components for styling
- scikit-learn for topic modeling
- NetworkX for network analysis
- WordCloud for text visualization
- Google's Generative AI API for LLM-powered features
