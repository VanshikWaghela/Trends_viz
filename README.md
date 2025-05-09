# Reddit Trend Analysis Dashboard

An interactive dashboard for analyzing trends and anomalies in Reddit discourse, with a focus on detecting and understanding unusual spikes in activity.

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

## Deployment on Render.com

This dashboard is ready for deployment on Render.com. Follow these steps:

1. Push this codebase to a GitHub repository:
   - Make sure the `.gitignore` file is properly set up to exclude data files
   - The repository should include: `app.py`, `requirements.txt`, and `build.sh`

2. Create a new Web Service on Render:
   - Sign up or log in to [Render.com](https://render.com)
   - Click "New" and select "Web Service"
   - Connect your GitHub repository
   - Set the name for your app
   - Ensure Environment is set to "Python"
   - Set Build Command to: `./build.sh`
   - Set Start Command to: `gunicorn app:server`
   - Under "Advanced" settings, add environment variables:
     - Add `GOOGLE_API_KEY` with your API key value

3. Select a plan (Free tier works for testing) and click "Create Web Service"

4. Render will build and deploy your app. Once deployed, you can access it at the provided URL.

Note: The free tier on Render has some limitations, including spinning down after periods of inactivity. For production use, consider upgrading to a paid plan.

## Data Requirements

The dashboard expects a CSV file named `reddit_preprocessed_data.csv` with the following columns:
- `id`: Unique identifier for each post
- `timestamp`: Post creation time (will be converted to datetime)
- `subreddit`: Subreddit where the post was made
- `author`: Username of the post author
- `author_id_stable`: Stable identifier for the author
- `title`: Post title
- `selftext`: Post body text
- `domain`: Domain of any linked content
- `final_url`: Full URL of any linked content
- `score`: Post score/engagement metric

## LLM-Powered Features

The dashboard includes several AI-powered features in the "AI Insights" tab:

1. **AI-Generated Trend Summary**: Get a concise executive summary of key trends and insights from the dataset
2. **Insights Chatbot**: Ask questions about the data and analysis to receive context-aware answers
3. **Predictive Analysis**: Predict engagement levels for hypothetical Reddit posts with AI-powered analysis

These features require a valid Gemini API key. The dashboard uses Google's `gemini-1.5-flash-preview-0417` model for optimal performance.

## Development

This dashboard is built with:
- Dash and Plotly for interactive visualizations
- Dash Bootstrap Components for styling
- scikit-learn for topic modeling
- NetworkX for network analysis
- WordCloud for text visualization
- Google's Generative AI API for LLM-powered features
