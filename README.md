# Reddit Trend Analysis Dashboard

An interactive dashboard for analyzing trends and anomalies in Reddit discourse, with a focus on detecting and understanding unusual spikes in activity.

**Live Demo:** [https://surge-ll4j.onrender.com](https://surge-ll4j.onrender.com)

## Features

- **Temporal Analysis**: Visualize post volume over time to identify activity spikes
- **Community Insights**: Analyze top subreddits, authors, and domains
- **Topic Modeling**: Use LDA and NMF to uncover latent topics in discussions
- **Engagement Amplification Analysis**: Detect unusual engagement patterns compared to baselines
- **Author-Topic Network Visualization**: Discover relationships between authors and topics
- **Coordinated Activity Detection**: Find instances of coordinated posting behavior
- **AI-Generated Insights**: Get AI-powered summaries and engage with a topic-aware chatbot

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
