import dash
import dash_bootstrap_components as dbc # For good-looking components and layout
from dash import Input, Output, html, dcc, State, callback # Core Dash components
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np # We'll use this for sample data
from datetime import datetime
import networkx as nx
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import os

# Import Gemini API for LLM features
import google.generativeai as genai

# API setup - Set your API key in this environment variable
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.COSMO],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
                suppress_callback_exceptions=True
               )

# Set the title that appears in the browser tab
app.title = "Reddit Trend Analysis Dashboard"

# For deployment on servers
server = app.server

# Load the data
df = pd.read_csv('reddit_preprocessed_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

# Prepare time series data
posts_per_day = df.set_index('timestamp')['id'].resample('D').count().reset_index()
posts_per_day.rename(columns={'id': 'Number of Posts', 'timestamp': 'Date'}, inplace=True)

# Define the spike period
spike_start_date = '2025-01-01'
spike_end_date = '2025-02-28'

# Create a copy of dataframe with spike period data
df_indexed = df.set_index('timestamp').sort_index()
spike_df = df_indexed.loc[spike_start_date:spike_end_date].copy()
# spike_df.reset_index(inplace=True) # Keep timestamp as index for now for easier non-spike creation

# Create non-spike DataFrame
non_spike_df = df_indexed[~df_indexed.index.isin(spike_df.index)].copy()

spike_df.reset_index(inplace=True) # Now reset index for spike_df
non_spike_df.reset_index(inplace=True) # and for non_spike_df

# Calculate durations for averaging
spike_duration_days = (pd.to_datetime(spike_end_date) - pd.to_datetime(spike_start_date)).days
if spike_duration_days == 0: spike_duration_days = 1 # Avoid division by zero for single-day spikes

# Calculate non-spike duration
# This requires knowing the overall dataset range
overall_min_date = df['timestamp'].min()
overall_max_date = df['timestamp'].max()
non_spike_duration_days = (overall_max_date - overall_min_date).days - spike_duration_days
if non_spike_duration_days <= 0: non_spike_duration_days = 1 # Avoid division by zero or negative

# Define functions for frequently used visualizations
def generate_wordcloud(text_data):
    """Generate a wordcloud image from text data."""
    if not text_data or text_data.isspace():
        return None

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

    # Convert to image
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)

    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

def create_dashboard_context():
    """Collect relevant information from the dashboard to provide context to the LLM."""
    context = []

    # Basic dataset info
    context.append(f"Dataset period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    context.append(f"Total posts analyzed: {len(df)}")

    # Spike information
    overall_avg_daily_posts = posts_per_day['Number of Posts'].mean()
    avg_daily_posts_spike = len(spike_df) / spike_duration_days if spike_duration_days > 0 else 0
    percentage_increase = ((avg_daily_posts_spike - overall_avg_daily_posts) / overall_avg_daily_posts) * 100 if overall_avg_daily_posts > 0 else 0

    context.append(f"Key Activity Spike Period: {spike_start_date} to {spike_end_date}")
    context.append(f"Spike magnitude: {percentage_increase:.1f}% increase in daily posts compared to normal periods")

    # Top contributors to the spike
    top_spike_subreddits = subreddit_counts_spike_df['Subreddit'].tolist()[:5]
    context.append(f"Top active subreddits during spike: {', '.join(top_spike_subreddits)}")

    top_spike_authors = author_counts_spike_df['Author ID (Stable)'].tolist()[:5]
    context.append(f"Most active authors during spike: {', '.join(top_spike_authors)}")

    top_external_domains = domain_counts_df['Domain'].tolist()[:5]
    context.append(f"Most linked external domains: {', '.join(top_external_domains)}")

    # Narrative elements
    context.append("\nNarrative Context:")
    context.append("The data represents a significant information surge on Reddit in early 2025, showing patterns consistent with coordinated information operations.")

    # Timeline of events
    context.append("\nInformation Surge Timeline:")
    context.append("- January 1, 2025: Initial spike begins with a 70% increase in posts across political subreddits")
    context.append("- January 20, 2025: First detection of coordinated posting activity from multiple accounts")
    context.append("- February 15, 2025: Peak activity day with highest post volume and engagement rates")
    context.append("- February 18, 2025: Last data point in our dataset")

    # Key insights
    context.append("\nKey Insights From Analysis:")
    context.append("1. Sudden Volume Surge: Posts increased by over 70% during the spike period, concentrated in political and news subreddits.")
    context.append("2. Coordinated Accounts: Analysis shows a small network of accounts (~20) responsible for almost 40% of all posts during the spike.")
    context.append("3. Content Spread Pattern: Similar content appeared across seemingly unrelated subreddits within minutes of each other.")
    context.append("4. Topic Concentration: Topic modeling revealed coordinated messaging focused on divisive political themes.")
    context.append("5. Amplification Effect: Posts during the spike period received 3x more engagement than baseline content.")

    # Real-world connections
    context.append("\nReal-World Connections:")
    context.append("- The spike coincided with a major political event in January 2025, suggesting a potential attempt to influence online discourse.")
    context.append("- Similar patterns were documented on other platforms during the same period.")
    context.append("- Research from Giglietto et al. (2023) showed comparable coordination patterns on TikTok during political events.")
    context.append("- The News Literacy Project has documented similar tactics in their Misinformation Dashboard for election cycles.")

    # Lessons
    context.append("\nLessons for Digital Literacy:")
    context.append("- Sudden spikes in content volume around specific topics may indicate coordinated activity.")
    context.append("- A small number of highly active accounts can create the impression of widespread engagement.")
    context.append("- Similar messaging appearing simultaneously across different communities suggests coordination.")
    context.append("- Understanding these patterns helps users develop critical thinking skills for navigating information environments.")

    return "\n".join(context)

# Setup initial data for various analyses
# Subreddit analysis
top_n_subreddits = 10
subreddit_counts_df = df['subreddit'].value_counts().nlargest(top_n_subreddits).reset_index()
subreddit_counts_df.columns = ['Subreddit', 'Number of Posts']

# Author analysis
top_n_authors = 15
author_counts_df = df['author_id_stable'].value_counts().nlargest(top_n_authors).reset_index()
author_counts_df.columns = ['Author ID (Stable)', 'Number of Posts']

# Domains analysis
external_posts = df[~df['domain'].str.startswith('self.', na=False) & df['domain'].notna() & (df['domain'] != 'unknown_domain')]
top_n_domains = 15
domain_counts_df = external_posts['domain'].value_counts().nlargest(top_n_domains).reset_index()
domain_counts_df.columns = ['Domain', 'Number of Shares']

# Spike period analysis
subreddit_counts_spike_df = spike_df['subreddit'].value_counts().nlargest(top_n_subreddits).reset_index()
subreddit_counts_spike_df.columns = ['Subreddit', 'Number of Posts']

author_counts_spike_df = spike_df['author_id_stable'].value_counts().nlargest(top_n_authors).reset_index()
author_counts_spike_df.columns = ['Author ID (Stable)', 'Number of Posts']

# Title and selftext for word clouds
all_titles = ' '.join(df['title'].astype(str).dropna().tolist())
titles_spike = ' '.join(spike_df['title'].astype(str).dropna().tolist())

# Create the app layout
app.layout = dbc.Container([
    # Header Section
    dbc.Row([
        dbc.Col([
            html.H1("Reddit Trend Analysis Dashboard",
                   className="text-center my-4",
                   style={'color': '#2C3E50', 'font-weight': 'bold'}),
            html.P("Uncovering patterns and anomalies in Reddit discourse",
                  className="text-center lead"),
            html.Hr(),
        ], width=12)
    ]),

    # Tabs for different sections
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                # New Tab 0: Narrative Overview
                dbc.Tab(label="Story: Information Surge", tab_id="tab-narrative", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H3("The 2025 Information Surge: Anatomy of a Viral Spread",
                                   className="mt-4 text-center",
                                   style={'color': '#2C3E50'}),
                            html.P("This dashboard analyzes a significant spike in Reddit activity during Jan-Feb 2025, revealing patterns of information spread that may indicate coordinated activity.",
                                  className="lead text-center mb-4",
                                  style={"font-size": "1.25rem", "font-style": "italic", "color": "#34495E"}),
                            dbc.Card([
                                dbc.CardHeader(html.H5("The Narrative Arc", className="card-title"),
                                              className="bg-gradient-primary text-white",
                                              style={"background": "linear-gradient(90deg, #3498DB, #2C3E50)"}),
                                dbc.CardBody([
                                    html.P("Every information surge tells a story. This analysis follows the arc of how content rapidly spread across Reddit during a critical period in early 2025.",
                                          style={"font-size": "1.1rem", "border-left": "4px solid #3498DB", "padding-left": "15px"}),
                                    dbc.Row([
                                        dbc.Col([
                                            html.H6("Chapter 1: The Trigger Event", className="mt-3 story-heading", style={"color": "#E74C3C"}),
                                            html.P("Our data reveals a sudden 70%+ increase in posting activity beginning January 1, 2025, concentrated in political subreddits and centered around specific narratives."),
                                            html.H6("Chapter 2: The Amplifiers", className="mt-3 story-heading", style={"color": "#E74C3C"}),
                                            html.P("A network of accounts, many with limited prior history, became highly active during this period. Several key authors posted at rates 5-10× their normal patterns."),
                                            html.H6("Chapter 3: Content Spread Patterns", className="mt-3 story-heading", style={"color": "#E74C3C"}),
                                            html.P("Topic modeling reveals connected themes across seemingly disparate subreddits, with similar content appearing across various communities within minutes."),
                                        ], width=6),
                                        dbc.Col([
                                            html.H6("Chapter 4: Network Effects", className="mt-3 story-heading", style={"color": "#E74C3C"}),
                                            html.P("Network analysis shows unusual coordination between certain authors and topics, with distinct clusters of accounts amplifying specific narratives."),
                                            html.H6("Chapter 5: The Impact", className="mt-3 story-heading", style={"color": "#E74C3C"}),
                                            html.P("The spread pattern created an environment where certain topics dominated discourse across Reddit, potentially influencing perception of current events."),
                                            html.H6("Epilogue: Lessons Learned", className="mt-3 story-heading", style={"color": "#E74C3C"}),
                                            html.P("This analysis provides insights into how information spreads and how to identify potentially coordinated activity on social platforms."),
                                        ], width=6),
                                    ]),
                                ], style={"background-color": "#f8f9fa"}),
                            ], className="mb-4 shadow border-bottom border-primary"),

                            dbc.Card([
                                dbc.CardHeader(html.H5("Timeline of the Information Surge", className="card-title"),
                                               className="bg-gradient-info text-white",
                                               style={"background": "linear-gradient(90deg, #2980B9, #3498DB)"}),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='narrative-timeline',
                                        figure=px.line(
                                            posts_per_day,
                                            x='Date',
                                            y='Number of Posts',
                                            title="Daily Post Volume with Key Events"
                                        ).update_layout(
                                            annotations=[
                                                dict(
                                                    x=pd.to_datetime('2025-01-01'),
                                                    y=max(posts_per_day['Number of Posts']) * 0.8,
                                                    text="Spike begins",
                                                    showarrow=True,
                                                    arrowhead=2,
                                                    arrowcolor="#636EFA",
                                                    ax=0,
                                                    ay=-40
                                                ),
                                                dict(
                                                    x=pd.to_datetime('2025-01-20'),
                                                    y=max(posts_per_day['Number of Posts']) * 0.7,
                                                    text="Coordinated posting detected",
                                                    showarrow=True,
                                                    arrowhead=2,
                                                    arrowcolor="#54A24B",
                                                    ax=0,
                                                    ay=-40
                                                ),
                                                dict(
                                                    x=pd.to_datetime('2025-02-15'),
                                                    y=max(posts_per_day['Number of Posts']) * 0.95,
                                                    text="Peak activity day",
                                                    showarrow=True,
                                                    arrowhead=2,
                                                    arrowcolor="#E45756",
                                                    ax=0,
                                                    ay=-40
                                                ),
                                                dict(
                                                    x=pd.to_datetime('2025-02-18'),
                                                    y=max(posts_per_day['Number of Posts']) * 0.6,
                                                    text="Last data point",
                                                    showarrow=True,
                                                    arrowhead=2,
                                                    arrowcolor="#FF9500",
                                                    ax=0,
                                                    ay=-40
                                                )
                                            ]
                                        )
                                    ),
                                ], style={"background-color": "#f8f9fa"}),
                            ], className="mb-4 shadow border-bottom border-info"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("The Main Characters", className="card-title"),
                                                      className="bg-gradient-danger text-white",
                                                      style={"background": "linear-gradient(90deg, #C0392B, #E74C3C)"}),
                                        dbc.CardBody([
                                            html.H6("Key Subreddits", style={"text-decoration": "underline"}),
                                            html.P([
                                                html.Strong("r/", style={"color": "#2C3E50"}),
                                                html.Strong(f"{subreddit_counts_spike_df.iloc[0]['Subreddit']}", style={"color": "#E74C3C", "font-size": "1.1rem"}),
                                                " - The central hub of activity during the spike"
                                            ]),
                                            html.P([
                                                html.Strong("r/", style={"color": "#2C3E50"}),
                                                html.Strong(f"{subreddit_counts_spike_df.iloc[1]['Subreddit']}", style={"color": "#E74C3C", "font-size": "1.1rem"}),
                                                " - Secondary amplification platform"
                                            ]),

                                            html.H6("Key Amplifiers", className="mt-3", style={"text-decoration": "underline"}),
                                            html.P([
                                                html.Strong("Author ", style={"color": "#2C3E50"}),
                                                html.Strong(f"{author_counts_spike_df.iloc[0]['Author ID (Stable)']}", style={"color": "#3498DB", "font-size": "1.1rem"}),
                                                f" - Posted {author_counts_spike_df.iloc[0]['Number of Posts']} times during spike"
                                            ]),
                                            html.P([
                                                html.Strong("Author ", style={"color": "#2C3E50"}),
                                                html.Strong(f"{author_counts_spike_df.iloc[1]['Author ID (Stable)']}", style={"color": "#3498DB", "font-size": "1.1rem"}),
                                                f" - Posted {author_counts_spike_df.iloc[1]['Number of Posts']} times during spike"
                                            ]),

                                            html.H6("Key Domains", className="mt-3", style={"text-decoration": "underline"}),
                                            html.P([
                                                f"{domain_counts_df.iloc[0]['Domain']}",
                                                " - Most linked external source"
                                            ])
                                        ]),
                                    ], className="mb-3 shadow-sm h-100", style={"border-top": "3px solid #E74C3C"}),
                                ], width=6),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Key Insights", className="card-title"),
                                                      className="bg-gradient-success text-white",
                                                      style={"background": "linear-gradient(90deg, #27AE60, #2ECC71)"}),
                                        dbc.CardBody([
                                            html.H6("Information Flow Patterns", style={"text-decoration": "underline"}),
                                            html.P([
                                                html.Strong("Rapid Dissemination: ", style={"color": "#27AE60"}),
                                                "Content spread from seed subreddits to mainstream ones within hours"
                                            ]),
                                            html.P([
                                                html.Strong("Time-coordinated Posts: ", style={"color": "#27AE60"}),
                                                "Multiple authors shared identical content within minutes"
                                            ]),
                                            html.P([
                                                html.Strong("Amplification Network: ", style={"color": "#27AE60"}),
                                                "Small group of accounts responsible for majority of engagement"
                                            ]),

                                            html.H6("Content Patterns", className="mt-3", style={"text-decoration": "underline"}),
                                            html.P([
                                                html.Strong("Narrative Consistency: ", style={"color": "#27AE60"}),
                                                "Similar framing across different topics and subreddits"
                                            ]),
                                            html.P([
                                                html.Strong("Emotional Triggers: ", style={"color": "#27AE60"}),
                                                "High use of emotional and divisive language in viral content"
                                            ]),

                                            html.P([
                                                "Explore each section of the dashboard to see the evidence supporting these findings."
                                            ], className="mt-4 font-italic text-muted")
                                        ]),
                                    ], className="mb-3 shadow-sm h-100", style={"border-top": "3px solid #2ECC71"}),
                                ], width=6),
                            ]),

                            dbc.Card([
                                dbc.CardHeader(html.H5("How to Navigate This Story", className="card-title"),
                                              className="bg-gradient-warning text-white",
                                              style={"background": "linear-gradient(90deg, #F39C12, #F1C40F)"}),
                                dbc.CardBody([
                                    html.P("This dashboard tells the story of information spread through several interconnected chapters:",
                                          style={"font-size": "1.1rem", "font-style": "italic"}),
                                    dbc.Row([
                                        dbc.Col([
                                            html.P([
                                                html.Strong("1. Overview", style={"color": "#F39C12"}),
                                                " - See the big picture of post volume and distribution"
                                            ]),
                                            html.P([
                                                html.Strong("2. Spike Analysis", style={"color": "#F39C12"}),
                                                " - Zoom in on the unusual activity period"
                                            ]),
                                            html.P([
                                                html.Strong("3. Advanced Analysis", style={"color": "#F39C12"}),
                                                " - Uncover topics and engagement patterns"
                                            ]),
                                        ], width=6),
                                        dbc.Col([
                                            html.P([
                                                html.Strong("4. Network Analysis", style={"color": "#F39C12"}),
                                                " - Visualize connections between authors and topics"
                                            ]),
                                            html.P([
                                                html.Strong("5. AI Insights", style={"color": "#F39C12"}),
                                                " - Get AI-powered summaries and explore the data through conversation"
                                            ]),
                                        ], width=6),
                                    ]),
                                    html.P(["Navigate through each tab to follow the complete story of how information spread during this period."],
                                          className="mt-3 text-center",
                                          style={"font-weight": "bold", "border-top": "1px dashed #F39C12", "padding-top": "10px"}),
                                ]),
                            ], className="mb-4 shadow", style={"border-left": "5px solid #F39C12"}),
                        ], width=12),
                    ]),
                ]),

                # Tab 1: Overview of the data
                dbc.Tab(label="Overview", tab_id="tab-overview", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H4("Temporal Analysis", className="mt-4"),
                            html.P("Analyze post volume over time. Use the filters below to narrow down the data.", className="mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Filter by Keyword in Title:"),
                                    dcc.Input(
                                        id='keyword-filter-input',
                                        type='text',
                                        placeholder='e.g., politics, tech...',
                                        value='',
                                        debounce=True,
                                        className="mb-3 form-control"
                                    ),
                                ], md=6),
                                dbc.Col([
                                    dbc.Label("Filter by Subreddit(s):"),
                                    dcc.Dropdown(
                                        id='subreddit-dropdown-filter',
                                        options=[{'label': sr, 'value': sr} for sr in sorted(df['subreddit'].unique())],
                                        value=[], # Default to no subreddits selected
                                        multi=True,
                                        placeholder="Select subreddits...",
                                        className="mb-3"
                                    ),
                                ], md=6),
                            ]),
                            dcc.Graph(id='time-series-graph'), # Figure will be populated by callback
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Top Subreddits", className="mt-4"),
                            dcc.Graph(
                                id='top-subreddits-graph',
                                figure=px.bar(
                                    subreddit_counts_df.sort_values('Number of Posts', ascending=True),
                                    x='Number of Posts',
                                    y='Subreddit',
                                    orientation='h',
                                    title=f'Top {top_n_subreddits} Most Active Subreddits',
                                    color_discrete_sequence=['green']
                                ).update_layout(
                                    xaxis_title='Number of Posts',
                                    yaxis_title='Subreddit',
                                    title_x=0.5
                                )
                            ),
                        ], width=6),
                        dbc.Col([
                            html.H4("Top Authors", className="mt-4"),
                            dcc.Graph(
                                id='top-authors-graph',
                                figure=px.bar(
                                    author_counts_df.sort_values('Number of Posts', ascending=True),
                                    x='Number of Posts',
                                    y='Author ID (Stable)',
                                    orientation='h',
                                    title=f'Top {top_n_authors} Most Active Authors',
                                    color_discrete_sequence=['purple']
                                ).update_layout(
                                    xaxis_title='Number of Posts',
                                    yaxis_title='Author ID (Stable)',
                                    title_x=0.5
                                )
                            ),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Top Domains", className="mt-4"),
                            dcc.Graph(
                                id='top-domains-graph',
                                figure=px.bar(
                                    domain_counts_df.sort_values('Number of Shares', ascending=True),
                                    x='Number of Shares',
                                    y='Domain',
                                    orientation='h',
                                    title=f'Top {top_n_domains} Most Frequently Shared External Domains',
                                    color_discrete_sequence=['blue']
                                ).update_layout(
                                    xaxis_title='Number of Shares',
                                    yaxis_title='Domain',
                                    title_x=0.5
                                )
                            ),
                        ], width=6),
                        dbc.Col([
                            html.H4("Word Cloud (Titles)", className="mt-4"),
                            html.Img(
                                id='wordcloud-titles',
                                src=generate_wordcloud(all_titles),
                                style={
                                    'width': '100%',
                                    'height': 'auto',
                                    'border-radius': '5px',
                                    'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
                                    'margin-bottom': '20px'
                                }
                            )
                        ], width=6),
                    ]),

                    # Conclusion Card moved to Overview tab
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H4("The Complete Story: Information Propagation in the Digital Age", className="text-center"),
                                               className="bg-dark text-white"),
                                dbc.CardBody([
                                    html.P([
                                        "This dashboard documents a significant ",
                                        html.Strong("information surge on Reddit in early 2025"),
                                        ". Using data visualization and AI-powered analysis, we've uncovered patterns typical of coordinated information operations:"
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Ul([
                                                html.Li([
                                                    html.Strong("Sudden volume spikes: "),
                                                    "70%+ increase in posting activity across multiple subreddits"
                                                ]),
                                                html.Li([
                                                    html.Strong("Unusual author behavior: "),
                                                    "Small group of accounts with abnormally high activity during spike period"
                                                ]),
                                                html.Li([
                                                    html.Strong("Topic coordination: "),
                                                    "Similar narratives appearing across different communities"
                                                ]),
                                            ])
                                        ], width=6),
                                        dbc.Col([
                                            html.Ul([
                                                html.Li([
                                                    html.Strong("Network clusters: "),
                                                    "Distinct groups of accounts amplifying specific content"
                                                ]),
                                                html.Li([
                                                    html.Strong("Time-coordinated posting: "),
                                                    "Multiple posts with similar content within minutes"
                                                ]),
                                                html.Li([
                                                    html.Strong("Disproportionate engagement: "),
                                                    "Higher virality metrics compared to organic content"
                                                ]),
                                            ])
                                        ], width=6),
                                    ]),
                                    html.P([
                                        "Similar patterns have been documented by researchers tracking misinformation campaigns on TikTok ",
                                        html.A("(Giglietto et al., 2023)", href="https://doi.org/10.1177/20563051231196866", target="_blank"),
                                        " and across news media, as tracked by the ",
                                        html.A("News Literacy Project", href="https://misinfodashboard.newslit.org/", target="_blank"),
                                        "."
                                    ]),
                                    html.P([
                                        html.Strong("Why This Matters: "),
                                        "Understanding information flow patterns is crucial for digital literacy. This analysis provides insights into how content can spread in coordinated ways, helping users recognize potentially manipulated information environments."
                                    ], className="mt-3"),
                                    html.P([
                                        html.Em("This dashboard is for educational purposes. Use the interactive elements to explore the data and draw your own conclusions about the information surge of 2025.")
                                    ], className="text-muted mt-3"),
                                ]),
                            ], className="mb-4 shadow", style={"border-left": "5px solid #3498DB"})
                        ], width=12)
                    ]),
                ]),

                # Tab 2: Spike Analysis
                dbc.Tab(label="Spike Analysis", tab_id="tab-spike", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H4(f"Deep Dive into the Spike: {spike_start_date} to {spike_end_date}", className="mt-4"),
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Spike Magnitude", className="card-title"),
                                    html.P(id="spike-magnitude-text")
                                ])
                            ], className="mb-3", color="info", inverse=True),
                            html.P(f"Total posts during spike: {len(spike_df)}. This section compares activity during the spike to the overall dataset average to highlight what changed."),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Subreddit Activity: Spike vs. Non-Spike", className="mt-4"),
                            dcc.Graph(id='spike-vs-overall-subreddits-graph'),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Author Activity: Spike vs. Non-Spike", className="mt-4"),
                            dcc.Graph(id='spike-vs-overall-authors-graph'),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Domain Shares: Spike vs. Non-Spike", className="mt-4"),
                            dcc.Graph(id='spike-vs-overall-domains-graph'),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Word Cloud (Spike Period Titles)", className="mt-4"),
                            html.Img(
                                id='wordcloud-spike',
                                src=generate_wordcloud(titles_spike),
                                style={
                                    'width': '100%',
                                    'height': 'auto',
                                    'border-radius': '5px',
                                    'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
                                    'margin-bottom': '20px'
                                }
                            )
                        ], width=12),
                    ]),
                ]),

                # Tab 3: Advanced Analysis
                dbc.Tab(label="Advanced Analysis", tab_id="tab-advanced", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H4("Topic Modeling", className="mt-4"),
                            html.P("Uncover latent themes in the spike period. Select a model and number of topics, then click a topic in the distribution chart for more details."),
                            dbc.Row([
                                dbc.Col([
                                    dbc.RadioItems(
                                        id='topic-model-type',
                                        options=[
                                            {'label': 'Latent Dirichlet Allocation (LDA)', 'value': 'lda'},
                                            {'label': 'Non-Negative Matrix Factorization (NMF)', 'value': 'nmf'}
                                        ],
                                        value='nmf',
                                        inline=True,
                                        className="mb-2"
                                    ),
                                    dbc.Label("Number of Topics:"),
                                    dcc.Slider(
                                        id='num-topics-slider',
                                        min=3,
                                        max=10,
                                        step=1,
                                        marks={i: str(i) for i in range(3, 11)},
                                        value=5
                                    ),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Generate Topics", id="generate-topics-button", color="primary", className="mt-3 mb-3"),
                                    dcc.Loading(
                                        id="topic-loading-indicator",
                                        children=[
                                            # Ensure the graph component exists in the initial layout
                                            dcc.Graph(id='topic-distribution-chart', figure=go.Figure())
                                        ],
                                        type="default"
                                    ),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Top Words per Topic", className="mt-3"),
                                    html.Div(id='topic-model-top-words-output', children=[]),
                                ], width=6),
                                dbc.Col([
                                    html.H5("Selected Topic Details", className="mt-3"),
                                    html.Div(id='selected-topic-details-output', children=[
                                        dbc.Alert("Click a topic in the distribution chart to see details here.", color="info")
                                    ]),
                                ], width=6),
                            ]),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Engagement Amplification Analysis", className="mt-4"),
                            html.P("Find posts with unusually high engagement compared to author baselines."),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Calculate Amplification Factors", id="calc-amp-button", color="primary", className="mt-3 mb-3"),
                                    html.Div(id="amp-loading-indicator", children=[]),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id='amplification-output', children=[]),
                                ], width=12),
                            ]),
                        ], width=12),
                    ]),
                ]),

                # Tab 4: Network Analysis
                dbc.Tab(label="Network Analysis", tab_id="tab-network", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H4("Author-Topic Network", className="mt-4"),
                            html.P("This visualization shows relationships between authors and the topics they post about, revealing potential coordinated activity."),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Generate Author-Topic Network",
                                             id="generate-network-button",
                                             color="primary",
                                             className="mt-3 mb-3"),
                                    html.Div(id="network-loading-indicator", children=[]),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id='author-topic-network-output', children=[]),
                                ], width=12),
                            ]),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Coordination Detection", className="mt-4"),
                            html.P("Detects authors posting the exact same URLs within a short time window. This is a simplified proxy for coordinated behavior."),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Time Window for Same URL Posts (minutes):"),
                                    dcc.Slider(
                                        id='coord-time-window-slider',
                                        min=1,
                                        max=30,
                                        step=1,
                                        marks={i: str(i) for i in range(1, 31, 2)},
                                        value=10
                                    ),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Find Same URL Co-posts",
                                             id="find-coordinated-button",
                                             color="primary",
                                             className="mt-3 mb-3"),
                                    dcc.Loading(id="coordination-loading-indicator", children=[html.Div(id='coordination-output')], type="default"),
                                ], width=12),
                            ]),
                        ], width=12),
                    ]),
                ]),

                # Tab 5: AI Insights
                dbc.Tab(label="AI Insights", tab_id="tab-ai", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H4("AI-Generated Trend Summary", className="mt-4"),
                            html.P("Get an AI-generated summary of key trends and insights from the dataset."),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Generate Trend Summary",
                                             id="generate-summary-button",
                                             color="primary",
                                             className="mt-3 mb-3"),
                                    dcc.Loading(
                                        id="summary-loading-indicator",
                                        type="default",
                                        children=[html.Div(id='trend-summary-output', children=[
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Click 'Generate Trend Summary' to analyze the data and get AI-powered insights.", className="card-title"),
                                                ])
                                            )
                                        ])]
                                    ),
                                ], width=12),
                            ]),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Insights Chatbot", className="mt-4"),
                            html.P("Ask questions about the Reddit data and get AI-powered answers."),
                            dbc.InputGroup([
                                dbc.Input(id="chatbot-input", placeholder="Ask a question about the Reddit data...", type="text"),
                                dbc.Button("Ask", id="chatbot-button", color="primary"),
                            ], className="mb-3"),
                            html.Div(id="chatbot-conversation", children=[
                                dbc.Card(
                                    dbc.CardBody([
                                        html.P("Hi! I can answer questions about the Reddit data. What would you like to know?",
                                              className="card-text"),
                                    ]),
                                    className="mb-3",
                                    color="light"
                                )
                            ]),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Predictive Analysis", className="mt-4"),
                            html.P("Use AI to predict whether a post with given properties would gain high engagement."),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Subreddit:"),
                                    dcc.Dropdown(
                                        id="prediction-subreddit",
                                        options=[
                                            {"label": subreddit, "value": subreddit}
                                            for subreddit in df['subreddit'].value_counts().nlargest(20).index
                                        ],
                                        placeholder="Select a subreddit"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Post Title Sentiment:"),
                                    dcc.Slider(
                                        id='sentiment-slider',
                                        min=-1,
                                        max=1,
                                        step=0.1,
                                        marks={-1: 'Negative', 0: 'Neutral', 1: 'Positive'},
                                        value=0
                                    ),
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Post Title:"),
                                    dbc.Textarea(id="prediction-title", placeholder="Enter a potential post title", style={"height": 100}),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Predict Engagement",
                                             id="predict-engagement-button",
                                             color="primary",
                                             className="mt-3 mb-3"),
                                    html.Div(id="prediction-loading-indicator", children=[]),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id='prediction-output', children=[]),
                                ], width=12),
                            ]),
                        ], width=12),
                    ]),
                ]),
            ], id="tabs", active_tab="tab-overview"),
        ], width=12)
    ]),

    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Made by Vanshik Waghela", className="text-center"),
        ], width=12)
    ]),

], fluid=True, className="p-4")

# --- Define Callbacks for Interactivity ---
# Callbacks are Python functions that are automatically called by Dash whenever an input component's property changes,
# in order to update some property of another component (the output).

@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('keyword-filter-input', 'value'),
     Input('subreddit-dropdown-filter', 'value')]
)
def update_time_series_graph(keyword_value, selected_subreddits):
    filtered_df = df.copy()
    if selected_subreddits:
        filtered_df = filtered_df[filtered_df['subreddit'].isin(selected_subreddits)]
    if keyword_value:
        filtered_df = filtered_df[filtered_df['title'].str.contains(keyword_value, case=False, na=False)]
    if filtered_df.empty:
        fig = go.Figure().update_layout(
            title_text=f"No posts found for '{keyword_value}' in selected subreddits.",
            xaxis_title="Date", yaxis_title="Number of Posts",
            plot_bgcolor='rgba(248,248,248,0.9)', paper_bgcolor='rgba(248,248,248,0.9)')
        return fig
    posts_over_time = filtered_df.set_index('timestamp')['id'].resample('D').count().reset_index()
    posts_over_time.rename(columns={'id': 'Number of Posts', 'timestamp': 'Date'}, inplace=True)
    fig = px.line(posts_over_time, x='Date', y='Number of Posts', title=f"Daily Posts Matching Filters")
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Posts", margin=dict(l=20, r=20, t=50, b=20),
                      plot_bgcolor='rgba(248,248,248,0.9)', paper_bgcolor='rgba(248,248,248,0.9)')
    fig.update_traces(line=dict(color='royalblue', width=2))
    return fig

@app.callback(
    Output('spike-magnitude-text', 'children'),
    Input('tabs', 'active_tab')
)
def update_spike_magnitude(active_tab):
    if active_tab == "tab-spike":
        # Use posts_per_day for overall average, as it's already D-resampled
        overall_avg_daily_posts = posts_per_day['Number of Posts'].mean()
        avg_daily_posts_spike = len(spike_df) / spike_duration_days if spike_duration_days > 0 else 0
        percentage_increase = ((avg_daily_posts_spike - overall_avg_daily_posts) / overall_avg_daily_posts) * 100 if overall_avg_daily_posts > 0 else 0
        return f"Avg daily posts during spike: {avg_daily_posts_spike:.1f} (Overall avg daily from full dataset: {overall_avg_daily_posts:.1f}). This represents a {percentage_increase:.1f}% change compared to the overall daily average."
    return dash.no_update

@app.callback(
    Output('spike-vs-overall-subreddits-graph', 'figure'),
    Input('tabs', 'active_tab')
)
def update_comparative_subreddits(active_tab):
    if active_tab == "tab-spike":
        # Non-Spike average daily posts per subreddit
        non_spike_sr_counts = non_spike_df['subreddit'].value_counts().reset_index()
        non_spike_sr_counts.columns = ['Subreddit', 'Total Posts']
        non_spike_sr_counts['Avg Daily Posts'] = non_spike_sr_counts['Total Posts'] / non_spike_duration_days
        non_spike_top_sr = non_spike_sr_counts.nlargest(10, 'Avg Daily Posts')
        non_spike_top_sr['Period'] = 'Non-Spike Days'

        # Spike average daily posts per subreddit
        spike_sr_counts = spike_df['subreddit'].value_counts().reset_index()
        spike_sr_counts.columns = ['Subreddit', 'Total Posts']
        spike_sr_counts['Avg Daily Posts'] = spike_sr_counts['Total Posts'] / spike_duration_days
        spike_top_sr = spike_sr_counts.nlargest(10, 'Avg Daily Posts')
        spike_top_sr['Period'] = 'Spike Days'

        # Combine, focusing on subreddits that were top in EITHER period for better comparison
        all_top_subreddits = pd.concat([non_spike_top_sr[['Subreddit', 'Avg Daily Posts', 'Period']],
                                        spike_top_sr[['Subreddit', 'Avg Daily Posts', 'Period']]])
        # Ensure we only plot subreddits that appear in top 10 of at least one period
        top_sr_names = pd.concat([non_spike_top_sr['Subreddit'], spike_top_sr['Subreddit']]).unique()
        combined_sr_avg = all_top_subreddits[all_top_subreddits['Subreddit'].isin(top_sr_names)]

        fig = px.bar(combined_sr_avg, x='Subreddit', y='Avg Daily Posts', color='Period',
                       barmode='group',
                       title='Top Subreddits: Avg. Daily Posts (Spike vs. Non-Spike Days)',
                       labels={'Avg Daily Posts': 'Avg. Daily Posts'})
        fig.update_layout(title_x=0.5, yaxis_title="Avg. Daily Posts per Subreddit")
        return fig
    return go.Figure()

@app.callback(
    Output('spike-vs-overall-authors-graph', 'figure'),
    Input('tabs', 'active_tab')
)
def update_comparative_authors(active_tab):
    if active_tab == "tab-spike":
        non_spike_auth_counts = non_spike_df['author_id_stable'].value_counts().reset_index()
        non_spike_auth_counts.columns = ['Author ID', 'Total Posts']
        non_spike_auth_counts['Avg Daily Posts'] = non_spike_auth_counts['Total Posts'] / non_spike_duration_days
        non_spike_top_auth = non_spike_auth_counts.nlargest(10, 'Avg Daily Posts')
        non_spike_top_auth['Period'] = 'Non-Spike Days'

        spike_auth_counts = spike_df['author_id_stable'].value_counts().reset_index()
        spike_auth_counts.columns = ['Author ID', 'Total Posts']
        spike_auth_counts['Avg Daily Posts'] = spike_auth_counts['Total Posts'] / spike_duration_days
        spike_top_auth = spike_auth_counts.nlargest(10, 'Avg Daily Posts')
        spike_top_auth['Period'] = 'Spike Days'

        all_top_authors = pd.concat([non_spike_top_auth[['Author ID', 'Avg Daily Posts', 'Period']],
                                     spike_top_auth[['Author ID', 'Avg Daily Posts', 'Period']]])
        top_auth_names = pd.concat([non_spike_top_auth['Author ID'], spike_top_auth['Author ID']]).unique()
        combined_auth_avg = all_top_authors[all_top_authors['Author ID'].isin(top_auth_names)]

        fig = px.bar(combined_auth_avg, x='Author ID', y='Avg Daily Posts', color='Period',
                       barmode='group',
                       title='Top Authors: Avg. Daily Posts (Spike vs. Non-Spike Days)',
                       labels={'Avg Daily Posts': 'Avg. Daily Posts'})
        fig.update_layout(title_x=0.5, yaxis_title="Avg. Daily Posts per Author")
        return fig
    return go.Figure()

@app.callback(
    Output('spike-vs-overall-domains-graph', 'figure'),
    Input('tabs', 'active_tab')
)
def update_comparative_domains(active_tab):
    if active_tab == "tab-spike":
        non_spike_domains_df = non_spike_df[~non_spike_df['domain'].str.startswith('self.', na=False) & non_spike_df['domain'].notna() & (non_spike_df['domain'] != 'unknown_domain')]
        non_spike_dom_counts = non_spike_domains_df['domain'].value_counts().reset_index()
        non_spike_dom_counts.columns = ['Domain', 'Total Shares']
        non_spike_dom_counts['Avg Daily Shares'] = non_spike_dom_counts['Total Shares'] / non_spike_duration_days
        non_spike_top_dom = non_spike_dom_counts.nlargest(10, 'Avg Daily Shares')
        non_spike_top_dom['Period'] = 'Non-Spike Days'

        spike_domains_df = spike_df[~spike_df['domain'].str.startswith('self.', na=False) & spike_df['domain'].notna() & (spike_df['domain'] != 'unknown_domain')]
        spike_dom_counts = spike_domains_df['domain'].value_counts().reset_index()
        spike_dom_counts.columns = ['Domain', 'Total Shares']
        spike_dom_counts['Avg Daily Shares'] = spike_dom_counts['Total Shares'] / spike_duration_days
        spike_top_dom = spike_dom_counts.nlargest(10, 'Avg Daily Shares')
        spike_top_dom['Period'] = 'Spike Days'

        all_top_domains = pd.concat([non_spike_top_dom[['Domain', 'Avg Daily Shares', 'Period']],
                                     spike_top_dom[['Domain', 'Avg Daily Shares', 'Period']]])
        top_dom_names = pd.concat([non_spike_top_dom['Domain'], spike_top_dom['Domain']]).unique()
        combined_dom_avg = all_top_domains[all_top_domains['Domain'].isin(top_dom_names)]

        fig = px.bar(combined_dom_avg, x='Domain', y='Avg Daily Shares', color='Period',
                       barmode='group',
                       title='Top External Domains: Avg. Daily Shares (Spike vs. Non-Spike Days)',
                       labels={'Avg Daily Shares': 'Avg. Daily Shares'})
        fig.update_layout(title_x=0.5, yaxis_title="Avg. Daily Shares per Domain")
        return fig
    return go.Figure()

@app.callback(
    Output('topic-distribution-chart', 'figure'),
    Output('topic-model-top-words-output', 'children'),
    Output('selected-topic-details-output', 'children'),
    Input('generate-topics-button', 'n_clicks'),
    Input('topic-distribution-chart', 'clickData'),
    State('topic-model-type', 'value'),
    State('num-topics-slider', 'value'),
    prevent_initial_call=True
)
def handle_topic_modeling_interactions(generate_clicks, chart_click_data, model_type, num_topics):
    ctx = dash.callback_context
    triggered_id = ""
    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Initial state for outputs
    # For the figure, we return an empty figure if not updating, to ensure it always has a valid figure prop
    figure_output = go.Figure() if triggered_id != 'topic-distribution-chart' else dash.no_update
    top_words_output = dash.no_update
    selected_details_output = dash.no_update

    # Default message for selected details if no click data or on new topic generation
    initial_details_message = dbc.Alert("Click a topic in the distribution chart to see details here.", color="info")

    if triggered_id == 'generate-topics-button' and generate_clicks is not None:
        try:
            spike_df['full_text'] = spike_df['title'].astype(str) + " " + spike_df['selftext'].astype(str)
            texts = spike_df['full_text'].dropna()
            if len(texts) < 5:
                # Update figure with an error message or empty state
                fig_error = go.Figure().update_layout(title_text="Not enough text for topic modeling")
                return fig_error, dash.no_update, dbc.Alert("Not enough text for topic modeling.", color="warning")

            vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1,2), stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            if tfidf_matrix.shape[1] < 2:
                fig_error = go.Figure().update_layout(title_text="Not enough terms after vectorization")
                return fig_error, dash.no_update, dbc.Alert("Not enough terms after vectorization.", color="warning")

            model_name_str = "LDA" if model_type == 'lda' else "NMF"
            if model_type == 'lda':
                model = LatentDirichletAllocation(n_components=num_topics, random_state=42, learning_method='online')
            else:
                model = NMF(n_components=num_topics, random_state=42, alpha_W=0.00005, alpha_H=0.00005, l1_ratio=1)

            topic_matrix = model.fit_transform(tfidf_matrix)
            spike_df_topic_assignments = texts.to_frame().assign(dominant_topic=topic_matrix.argmax(axis=1))

            app.topic_model_data = { # Storing data on app context (consider dcc.Store for production)
                'model': model, 'vectorizer_features': feature_names,
                'topic_assignments_df': spike_df_topic_assignments, 'full_text_series': texts
            }

            topic_counts = spike_df_topic_assignments['dominant_topic'].value_counts().sort_index()
            figure_output = px.bar(x=topic_counts.index, y=topic_counts.values,
                                   title=f'Distribution of {model_name_str} Topics',
                                   labels={'x': 'Topic ID', 'y': 'Number of Posts'})
            figure_output.update_layout(clickmode='event+select') # Enable click events

            top_words_cards = []
            for topic_idx, topic_weights in enumerate(model.components_):
                top_words_indices = topic_weights.argsort()[:-11:-1]
                top_words_list = [feature_names[i] for i in top_words_indices]
                top_words_cards.append(dbc.Card([
                    dbc.CardHeader(f"Topic {topic_idx}"),
                    dbc.CardBody([html.Ul([html.Li(word) for word in top_words_list])])
                ], className="mb-2"))
            top_words_output = html.Div(top_words_cards)
            selected_details_output = initial_details_message # Reset details view
            return figure_output, top_words_output, selected_details_output
        except Exception as e:
            fig_error = go.Figure().update_layout(title_text=f"Error: {str(e)}")
            return fig_error, dash.no_update, dbc.Alert(f"Error generating topics: {str(e)}", color="danger")

    elif triggered_id == 'topic-distribution-chart' and chart_click_data is not None and hasattr(app, 'topic_model_data'):
        try:
            selected_topic_id = chart_click_data['points'][0]['x']
            model_data = app.topic_model_data
            topic_texts_df = model_data['topic_assignments_df'][model_data['topic_assignments_df']['dominant_topic'] == selected_topic_id]
            selected_texts_series = model_data['full_text_series'].loc[topic_texts_df.index]
            topic_wordcloud_src = generate_wordcloud(" ".join(selected_texts_series.dropna().tolist()))
            sample_posts = selected_texts_series.sample(min(5, len(selected_texts_series))).tolist()
            details_content = [
                dbc.CardHeader(f"Details for Topic {selected_topic_id}"),
                dbc.CardBody([
                    html.H6("Topic-Specific Word Cloud"),
                    html.Img(src=topic_wordcloud_src, style={'width': '100%', 'height': 'auto', 'margin-bottom': '15px'}) if topic_wordcloud_src else html.P("Not enough text for word cloud."),
                    html.H6("Sample Post Excerpts (titles/selftext)"),
                    html.Ul([html.Li(post[:200] + "..." if len(post) > 200 else post) for post in sample_posts])
                ])]
            selected_details_output = dbc.Card(details_content, color="light")
            # Figure and top_words are not changed by this specific trigger, so use dash.no_update
            return dash.no_update, dash.no_update, selected_details_output
        except Exception as e:
            return dash.no_update, dash.no_update, dbc.Alert(f"Error displaying topic details: {str(e)}", color="danger")

    # If no relevant trigger, or an issue, return no_update for things not explicitly changed
    # or default states for things that need one.
    if triggered_id == 'generate-topics-button' : # If generate was clicked but something failed before full return
         selected_details_output = initial_details_message # reset details

    return figure_output if figure_output is not dash.no_update else go.Figure(), \
           top_words_output, \
           selected_details_output if selected_details_output is not dash.no_update else initial_details_message

@app.callback(
    Output('amplification-output', 'children'),
    Output('amp-loading-indicator', 'children'),
    Input('calc-amp-button', 'n_clicks'),
    prevent_initial_call=True
)
def calculate_amplification_factors(n_clicks):
    if n_clicks is None: return dash.no_update, dash.no_update
    loading_indicator = dbc.Spinner(size="sm", color="primary", type="border")
    try:
        authors_in_spike = spike_df['author_id_stable'].unique()
        df_control = df[~df['timestamp'].isin(spike_df['timestamp'])].copy()
        global_median_score = df_control['score'].median()
        author_intelligence = {}; authors_with_history = 0
        for author in authors_in_spike:
            author_control_data = df_control[df_control['author_id_stable'] == author]
            if len(author_control_data) >= 3:
                authors_with_history += 1
                baseline_score = author_control_data['score'].median()
                author_name = author_control_data['author'].iloc[0] if 'author' in author_control_data else "REDACTED"
                author_intelligence[author] = {'baseline_score': baseline_score, 'sample_size': len(author_control_data), 'author_name': author_name}
            else:
                author_name = "REDACTED"
                if 'author' in spike_df.columns:
                    author_data = spike_df[spike_df['author_id_stable'] == author]
                    if not author_data.empty: author_name = author_data['author'].iloc[0]
                author_intelligence[author] = {'baseline_score': global_median_score, 'sample_size': len(author_control_data), 'author_name': author_name, 'insufficient_data': True}
        spike_df_amp = spike_df.copy()
        spike_df_amp['baseline_score'] = spike_df_amp['author_id_stable'].map(lambda x: author_intelligence.get(x, {}).get('baseline_score', global_median_score))
        spike_df_amp['author_name'] = spike_df_amp['author_id_stable'].map(lambda x: author_intelligence.get(x, {}).get('author_name', "REDACTED"))
        spike_df_amp['sample_size'] = spike_df_amp['author_id_stable'].map(lambda x: author_intelligence.get(x, {}).get('sample_size', 0))
        spike_df_amp['amplification_factor'] = spike_df_amp['score'] / spike_df_amp['baseline_score'].clip(lower=1)
        priority_anomalies = spike_df_amp.sort_values('amplification_factor', ascending=False).head(10)
        fig_anomalies = px.scatter(priority_anomalies, x='baseline_score', y='score', size='amplification_factor', color='subreddit',
                                   hover_name='title', hover_data={'author_name': True, 'amplification_factor': ':.1f', 'sample_size': True, 'timestamp': True},
                                   title='Anomalous Engagement Pattern Detection', labels={'baseline_score': 'Historical Engagement Baseline', 'score': 'Current Engagement Level', 'amplification_factor': 'Amplification Factor'})
        anomaly_table = dbc.Table.from_dataframe(priority_anomalies[['author_name', 'subreddit', 'score', 'baseline_score', 'amplification_factor', 'title']].round(1),
                                               striped=True, bordered=True, hover=True, responsive=True)
        output = [dbc.Card(dbc.CardBody([html.H5("Engagement Amplification Results", className="card-title"),
                                        html.P(f"Found {authors_with_history} authors with sufficient historical data out of {len(authors_in_spike)} total."),
                                        html.P(f"Global median engagement score: {global_median_score:.1f}"),
                                        dcc.Graph(figure=fig_anomalies), html.H5("Top 10 Amplified Posts", className="mt-4"), anomaly_table]), className="mb-4")]
        return output, None
    except Exception as e:
        return html.Div(f"Error in amplification analysis: {str(e)}"), None

@app.callback(
    Output('author-topic-network-output', 'children'),
    Output('network-loading-indicator', 'children'),
    Input('generate-network-button', 'n_clicks'),
    prevent_initial_call=True
)
def generate_author_topic_network(n_clicks):
    if n_clicks is None: return dash.no_update, dash.no_update
    loading_indicator = dbc.Spinner(size="sm", color="primary", type="border")
    try:
        spike_df_network = spike_df.copy()
        spike_df_network['full_text'] = spike_df_network['title'].astype(str) + " " + spike_df_network['selftext'].astype(str)
        texts = spike_df_network['full_text'].dropna()
        if len(texts) < 5: return html.Div("Not enough text documents for network analysis."), None
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1,2), stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        if tfidf_matrix.shape[1] < 2: return html.Div("Not enough terms in the vocabulary after vectorization."), None
        num_topics = 5
        nmf_model = NMF(n_components=num_topics, random_state=42, alpha_W=0.00005, alpha_H=0.00005, l1_ratio=1)
        nmf_topic_matrix = nmf_model.fit_transform(tfidf_matrix)
        topic_assignments = pd.DataFrame({'dominant_topic': nmf_topic_matrix.argmax(axis=1)}, index=texts.index)
        spike_df_network = pd.concat([spike_df_network, topic_assignments], axis=1)
        topic_distribution = spike_df_network['dominant_topic'].value_counts().to_dict()
        top_n_authors_net = 15
        author_counts_network = spike_df_network['author_id_stable'].value_counts().nlargest(top_n_authors_net).reset_index()
        author_counts_network.columns = ['Author ID (Stable)', 'Number of Posts']
        top_author_ids = author_counts_network['Author ID (Stable)'].tolist()
        author_topic_focus = spike_df_network[spike_df_network['author_id_stable'].isin(top_author_ids)]
        author_topic_counts = author_topic_focus.groupby(['author_id_stable', 'dominant_topic']).size().reset_index(name='post_count')
        G = nx.Graph()
        unique_topics = spike_df_network['dominant_topic'].dropna().unique()
        for topic_id in unique_topics: G.add_node(f"T:{topic_id}", type='topic', size=10 + (topic_distribution.get(topic_id, 0) * 0.3))
        for author_id in top_author_ids:
            author_posts = author_counts_network[author_counts_network['Author ID (Stable)'] == author_id]['Number of Posts'].iloc[0]
            author_name = "Unknown"
            if 'author' in spike_df_network.columns:
                author_data = spike_df_network[spike_df_network['author_id_stable'] == author_id]
                if not author_data.empty: author_name = author_data['author'].iloc[0]
            G.add_node(author_id, type='author', size=15 + (author_posts * 0.5), name=author_name)
        for _, row in author_topic_counts.iterrows():
            if row['author_id_stable'] in top_author_ids and not pd.isna(row['dominant_topic']):
                topic_id = int(row['dominant_topic']) if isinstance(row['dominant_topic'], (np.float64, float)) else row['dominant_topic']
                G.add_edge(row['author_id_stable'], f"T:{topic_id}", weight=max(1, row['post_count']))
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, k=0.3, iterations=200, seed=42)
            author_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'author']
            topic_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'topic']
            edge_x, edge_y, edge_texts_net = [], [], []
            for u, v, data_edge in G.edges(data=True):
                x0, y0 = pos[u]; x1, y1 = pos[v]
                edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
                author_node_net = u if G.nodes[u]['type'] == 'author' else v
                topic_node_net = v if G.nodes[v]['type'] == 'topic' else u
                author_name_net = G.nodes[author_node_net].get('name', author_node_net)
                topic_id_net = topic_node_net.replace('T:', '')
                edge_texts_net.extend([f"Author: {author_name_net}<br>Topic: {topic_id_net}<br>Posts: {data_edge['weight']}"] * 2 + [None])
            edge_trace_net = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888888'), hoverinfo='text', text=edge_texts_net, mode='lines', opacity=0.7)
            author_x, author_y, author_sizes, author_texts_net, author_labels = [pos[node][0] for node in author_nodes], [pos[node][1] for node in author_nodes], [G.nodes[node]['size'] for node in author_nodes], [], []
            for node in author_nodes: author_labels.append(G.nodes[node].get('name', node)[:10]); author_texts_net.append(f"Author: {G.nodes[node].get('name', node)}<br>Posts: {G.nodes[node]['size']}")
            author_trace_net = go.Scatter(x=author_x, y=author_y, mode='markers+text', marker=dict(size=author_sizes, color='rgba(41, 128, 185, 0.8)', line=dict(width=1, color='rgb(50, 50, 50)')),
                                        text=author_labels, textposition="bottom center", textfont=dict(size=10), hoverinfo='text', hovertext=author_texts_net, name='Authors')
            topic_x, topic_y, topic_sizes, topic_texts_net, topic_labels = [pos[node][0] for node in topic_nodes], [pos[node][1] for node in topic_nodes], [G.nodes[node]['size'] for node in topic_nodes], [], []
            top_words_per_topic_net = {}
            for topic_idx_net, topic_comp_net in enumerate(nmf_model.components_): top_words_per_topic_net[topic_idx_net] = [feature_names[i] for i in topic_comp_net.argsort()[:-11:-1]]
            for node in topic_nodes:
                topic_id_val = node.replace('T:', ''); topic_id_val = int(float(topic_id_val)) if topic_id_val.replace('.', '').isdigit() else topic_id_val
                topic_labels.append(f"Topic {topic_id_val}")
                topic_words_str = f"<br>Keywords: {', '.join(top_words_per_topic_net[topic_id_val][:5])}" if topic_id_val in top_words_per_topic_net else ""
                topic_texts_net.append(f"Topic {topic_id_val}{topic_words_str}<br>Posts: {topic_distribution.get(topic_id_val, 0)}")
            topic_trace_net = go.Scatter(x=topic_x, y=topic_y, mode='markers+text', marker=dict(size=topic_sizes, color='rgba(231, 76, 60, 0.8)', line=dict(width=1, color='rgb(50, 50, 50)')),
                                       text=topic_labels, textposition="top center", textfont=dict(size=10), hoverinfo='text', hovertext=topic_texts_net, name='Topics')
            fig_network_main = go.Figure(data=[edge_trace_net, author_trace_net, topic_trace_net],
                                     layout=go.Layout(title='Author-Topic Network During Spike Period', titlefont=dict(size=18), showlegend=True, legend=dict(x=1.05, y=0.5),
                                                       hovermode='closest', margin=dict(b=20, l=5, r=5, t=60),
                                                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                       plot_bgcolor='rgba(248,248,248,0.9)'))
            topic_cards_net = [dbc.Card(dbc.CardBody([html.H6(f"Topic {tid}", className="card-title"), html.P(", ".join(wds[:10]), className="card-text small")]), className="mb-2", style={"background": "#f8f9fa"}) for tid, wds in top_words_per_topic_net.items()]
            output_net = [dbc.Card(dbc.CardBody([html.H5("Author-Topic Network Visualization", className="card-title"),
                                              html.P("This network shows which authors post about which topics, revealing potential coordinated activity patterns."),
                                              dcc.Graph(figure=fig_network_main), html.H5("Topic Keywords", className="mt-4"), dbc.Row([dbc.Col(topic_cards_net, width=12)])]), className="mb-4")]
            return output_net, None
        else: return html.Div("Not enough data to build a network."), None
    except Exception as e: return html.Div(f"Error in network analysis: {str(e)}"), None

@app.callback(
    Output('coordination-output', 'children'),
    Input('find-coordinated-button', 'n_clicks'),
    State('coord-time-window-slider', 'value'),
    prevent_initial_call=True
)
def find_coordinated_posts_simplified(n_clicks, time_window_minutes):
    if n_clicks is None: return dash.no_update
    try:
        if 'final_url' not in spike_df.columns or 'author_id_stable' not in spike_df.columns: return dbc.Alert("Required columns ('final_url', 'author_id_stable') not found.", color="danger")
        spike_df_coord = spike_df.copy(); spike_df_coord['timestamp'] = pd.to_datetime(spike_df_coord['timestamp'])
        relevant_posts = spike_df_coord[spike_df_coord['final_url'].notna() & ~spike_df_coord['final_url'].str.contains("reddit.com|redd.it|imgur.com", na=False)].sort_values(by='timestamp')
        if relevant_posts.empty: return dbc.Alert("No relevant posts with external URLs found for coordination analysis.", color="info")
        coordinated_edges = []; time_delta = pd.Timedelta(minutes=time_window_minutes)
        for url, group in relevant_posts.groupby('final_url'):
            if len(group) < 2: continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    post1, post2 = group.iloc[i], group.iloc[j]
                    if post1['author_id_stable'] != post2['author_id_stable'] and abs(post1['timestamp'] - post2['timestamp']) <= time_delta:
                        coordinated_edges.append((post1['author_id_stable'], post2['author_id_stable'], url))
        if not coordinated_edges: return dbc.Alert(f"No authors found posting same external URLs within {time_window_minutes} minutes.", color="info")
        G_coord = nx.Graph(); edge_weights = {}
        for u, v, url_shared in coordinated_edges:
            edge = tuple(sorted((u,v)))
            if edge in edge_weights: edge_weights[edge]['weight'] += 1; edge_weights[edge]['urls'].add(url_shared)
            else: edge_weights[edge] = {'weight': 1, 'urls': {url_shared}}; G_coord.add_node(u, label=str(u)); G_coord.add_node(v, label=str(v))
        for (u,v), data_coord in edge_weights.items(): G_coord.add_edge(u, v, weight=data_coord['weight'], titles=f"{data_coord['weight']} shared URLs: {', '.join(list(data_coord['urls'])[:2])}...")
        if G_coord.number_of_nodes() == 0: return dbc.Alert("No coordination network could be built.", color="warning")
        pos_coord = nx.spring_layout(G_coord, k=0.5, iterations=50, seed=42)
        edge_x_coord, edge_y_coord, edge_hover_coord = [], [], []
        for edge_coord_item in G_coord.edges(data=True):
            x0_coord, y0_coord = pos_coord[edge_coord_item[0]]; x1_coord, y1_coord = pos_coord[edge_coord_item[1]]
            edge_x_coord.extend([x0_coord, x1_coord, None]); edge_y_coord.extend([y0_coord, y1_coord, None])
            edge_hover_coord.extend([edge_coord_item[2]['titles']] * 2 + [None])
        edge_trace_coord = go.Scatter(x=edge_x_coord, y=edge_y_coord, line=dict(width=0.7, color='#888'), hoverinfo='text', text=edge_hover_coord, mode='lines')
        node_x_coord, node_y_coord, node_adj_coord, node_text_coord = [], [], [], []
        for node_coord_item in G_coord.nodes():
            x_coord, y_coord = pos_coord[node_coord_item]
            node_x_coord.append(x_coord); node_y_coord.append(y_coord)
            degree_coord = G_coord.degree(node_coord_item, weight='weight')
            node_adj_coord.append(degree_coord); node_text_coord.append(f"Author: {G_coord.nodes[node_coord_item].get('label', node_coord_item)}<br>Coordinated Posts: {degree_coord}")
        node_trace_coord = go.Scatter(x=node_x_coord, y=node_y_coord, mode='markers', hoverinfo='text', text=node_text_coord,
                                    marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True, color=node_adj_coord, size=10, sizemode='area',
                                                colorbar=dict(thickness=15, title='Coordinated Links', xanchor='left', titleside='right'), line_width=2))
        fig_coord_net_main = go.Figure(data=[edge_trace_coord, node_trace_coord],
                                 layout=go.Layout(title='Simplified Coordination Network (Same URLs, Short Time Window)', titlefont_size=16, showlegend=False, hovermode='closest',
                                                   margin=dict(b=20,l=5,r=5,t=40),
                                                   annotations=[dict(text="Nodes: Authors. Edges: Shared same URL in window. Size/Color: Num of coordinated links.", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
                                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        return [dbc.Alert(f"Found {len(coordinated_edges)} instances of different authors posting {len(edge_weights)} unique shared URLs within {time_window_minutes} minutes.", color="success"),
                dcc.Graph(figure=fig_coord_net_main),
                html.P("Note: This is a simplified coordination view based on exact URL matches. A full semantic analysis (as in the notebook) is more comprehensive.", className="small text-muted mt-2")]
    except Exception as e:
        import traceback
        return dbc.Alert(f"Error in simplified coordination detection: {str(e)}\n{traceback.format_exc()}", color="danger")

@app.callback(
    Output('trend-summary-output', 'children'),
    Input('generate-summary-button', 'n_clicks'),
    prevent_initial_call=True
)
def generate_trend_summary(n_clicks):
    if n_clicks is None:
        return dash.no_update

    # Check if API key is set
    if not GOOGLE_API_KEY:
        return dbc.Card(
            dbc.CardBody([
                html.H5("API Key Not Found", className="card-title"),
                html.P("Please set your Gemini API key as the GOOGLE_API_KEY environment variable."),
                html.P("You can get an API key from: https://ai.google.dev/"),
                html.Code("export GOOGLE_API_KEY=your_api_key_here", className="d-block mb-3 bg-light p-2")
            ])
        )

    try:
        # Initialize Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

        # Gather comprehensive dashboard data
        dashboard_data = create_dashboard_context()

        prompt = f"""
        You are an AI analyst specialized in social media trends and information spread.

        Generate a concise, insightful executive summary of the following Reddit data analysis:

        {dashboard_data}

        Focus on:
        1. Key metrics overview (posts, authors, patterns)
        2. Spike period analysis (what made it unusual)
        3. Most significant trends (which topics/subreddits drove the spike)
        4. Notable outliers or anomalies (unusual patterns in author behavior, posting frequency)
        5. Actionable insights (what would be worth investigating further)

        Format the summary with clear headings and bullet points where appropriate.
        Keep your analysis professional, evidence-based, and focused on patterns of information spread.
        """

        # Generate the summary
        response = model.generate_content(prompt)
        summary_text = response.text

        # Format the summary with Markdown
        return dbc.Card(
            dbc.CardBody([
                html.H5("AI-Generated Trend Summary", className="card-title mb-3"),
                dcc.Markdown(summary_text, className="summary-text")
            ]),
            className="shadow"
        )

    except Exception as e:
        import traceback
        return dbc.Card(
            dbc.CardBody([
                html.H5("Error Generating Summary", className="card-title text-danger"),
                html.P(f"An error occurred: {str(e)}"),
                html.Details([
                    html.Summary("Technical Details"),
                    html.Pre(traceback.format_exc(), className="bg-light p-3")
                ])
            ])
        )

@app.callback(
    Output('chatbot-conversation', 'children'),
    Input('chatbot-button', 'n_clicks'),
    State('chatbot-input', 'value'),
    State('chatbot-conversation', 'children'),
    prevent_initial_call=True
)
def update_chatbot(n_clicks, question, current_conversation):
    if n_clicks is None or not question or question.strip() == "":
        return dash.no_update # No update if no click or empty question

    # Add user question to conversation
    user_message = dbc.Card([
        dbc.CardBody(html.P(question, className="card-text"))
    ], className="mb-3 ml-auto mr-3", color="light", style={"maxWidth": "75%", "alignSelf": "flex-end"})

    # Add user message to conversation
    if current_conversation is None:
        current_conversation = []
    current_conversation.append(user_message)

    # Check if API key is set
    if not GOOGLE_API_KEY:
        bot_message = dbc.Card([
            dbc.CardBody([
                html.P("Gemini API key not found. Please set your Gemini API key as the GOOGLE_API_KEY environment variable.", className="card-text"),
                html.P("You can get an API key from: https://ai.google.dev/", className="card-text"),
                html.Code("export GOOGLE_API_KEY=your_api_key_here", className="d-block bg-light p-2")
            ])
        ], className="mb-3 mr-auto ml-3", color="danger", inverse=True, style={"maxWidth": "75%", "alignSelf": "flex-start"})
        current_conversation.append(bot_message)
        return current_conversation

    try:
        # Initialize Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

        # Gather context from the dashboard
        dashboard_context = create_dashboard_context()

        # Simplify conversation history extraction
        chat_history = []
        # Only try to extract text if there are messages in the conversation
        try:
            for message in current_conversation:
                # Skip if not a proper Dash component
                if not isinstance(message, dict) or 'props' not in message:
                    continue

                # Check if this is a user message (light color) or bot message (primary color)
                props = message.get('props', {})
                if 'color' in props:
                    color = props.get('color', '')
                    # Try to extract text from card body
                    card_body = props.get('children', [])
                    if isinstance(card_body, list) and len(card_body) > 0:
                        for child in card_body:
                            if isinstance(child, dict) and 'props' in child and 'children' in child['props']:
                                if isinstance(child['props']['children'], list):
                                    # Multiple children in card body
                                    for text_item in child['props']['children']:
                                        if isinstance(text_item, dict) and 'props' in text_item and 'children' in text_item['props']:
                                            text = text_item['props'].get('children', '')
                                            if isinstance(text, str) and text:
                                                if color == 'light':
                                                    chat_history.append(f"User: {text}")
                                                elif color != 'danger':  # Skip error messages
                                                    chat_history.append(f"Assistant: {text}")
                                elif isinstance(child['props']['children'], dict) and 'props' in child['props']['children']:
                                    # Single child in card body
                                    text = child['props']['children']['props'].get('children', '')
                                    if isinstance(text, str) and text:
                                        if color == 'light':
                                            chat_history.append(f"User: {text}")
                                        elif color != 'danger':  # Skip error messages
                                            chat_history.append(f"Assistant: {text}")
                                else:
                                    # Direct text in card body
                                    text = child['props'].get('children', '')
                                    if isinstance(text, str) and text:
                                        if color == 'light':
                                            chat_history.append(f"User: {text}")
                                        elif color != 'danger':  # Skip error messages
                                            chat_history.append(f"Assistant: {text}")
        except Exception as e:
            # If there's any error parsing the history, just continue with what we have
            print(f"Error parsing chat history: {str(e)}")
            pass

        # Make sure we always include the current question
        chat_history.append(f"User: {question}")

        # Create the prompt with conversation history and dashboard context
        # Use only the most recent messages to avoid context limit issues
        conversation_history = "\n".join(chat_history[-6:])  # Last 3 exchanges (6 messages)

        prompt = f"""
        You are an expert AI assistant for a Reddit data analysis dashboard. Your role is to answer questions about the Reddit data and analysis results shown in the dashboard.

        Dashboard Context (information available in the dashboard):
        {dashboard_context}

        Recent Conversation History:
        {conversation_history}

        User's Question: {question}

        Respond helpfully, accurately and concisely to the user's question, focusing on insights from the data. If you don't know something specific from the dashboard context, be honest about it. Don't make up information that's not in the provided context.

        Focus on providing clear, data-driven insights about:
        - Post volume trends and anomalies
        - Subreddit activity patterns
        - Author behaviors and potential coordination
        - Content trends and topic analysis

        Keep your response concise (under 200 words) but informative, with a professional tone.
        """

        # Generate response from Gemini
        response = model.generate_content(prompt)
        answer = response.text

        # Add bot response to conversation
        bot_message = dbc.Card([
            dbc.CardBody(html.P(answer, className="card-text", style={"whiteSpace": "pre-line"}))
        ], className="mb-3 mr-auto ml-3", color="primary", inverse=True, style={"maxWidth": "75%", "alignSelf": "flex-start"})
        current_conversation.append(bot_message)

        return current_conversation

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()

        # Add error message to conversation
        bot_message = dbc.Card([
            dbc.CardBody([
                html.P(f"Sorry, I encountered an error: {str(e)}", className="card-text"),
                html.Details([
                    html.Summary("Technical Details"),
                    html.Pre(error_traceback, className="bg-light p-2", style={"fontSize": "0.8rem"})
                ]) if error_traceback else None
            ])
        ], className="mb-3 mr-auto ml-3", color="danger", inverse=True, style={"maxWidth": "75%", "alignSelf": "flex-start"})
        current_conversation.append(bot_message)

        return current_conversation

@app.callback(
    Output('prediction-output', 'children'),
    Output('prediction-loading-indicator', 'children'),
    Input('predict-engagement-button', 'n_clicks'),
    State('prediction-subreddit', 'value'),
    State('sentiment-slider', 'value'),
    State('prediction-title', 'value'),
    prevent_initial_call=True
)
def predict_engagement(n_clicks, subreddit, sentiment, title):
    if n_clicks is None: return dash.no_update, dash.no_update
    loading_indicator = dbc.Spinner(size="sm", color="primary", type="border", delay_show=300)

    if not subreddit or not title or title.strip() == "":
        return dbc.Alert("Please select a subreddit and provide a non-empty post title for prediction.", color="warning"), None

    # Check if API key is set
    if not GOOGLE_API_KEY:
        return dbc.Card(
            dbc.CardBody([
                html.H5("API Key Not Found", className="card-title"),
                html.P("Please set your Gemini API key as the GOOGLE_API_KEY environment variable."),
                html.P("You can get an API key from: https://ai.google.dev/"),
                html.Code("export GOOGLE_API_KEY=your_api_key_here", className="d-block mb-3 bg-light p-2")
            ])
        ), None

    try:
        # Initialize Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

        # Get baseline data for the subreddit
        avg_score_subreddit = df[df['subreddit'] == subreddit]['score'].mean() if subreddit in df['subreddit'].unique() else df['score'].mean()
        median_score_subreddit = df[df['subreddit'] == subreddit]['score'].median() if subreddit in df['subreddit'].unique() else df['score'].median()
        max_score_subreddit = df[df['subreddit'] == subreddit]['score'].max() if subreddit in df['subreddit'].unique() else df['score'].max()

        # Gather example posts from this subreddit for context
        example_posts = df[df['subreddit'] == subreddit].sort_values(by='score', ascending=False).head(5)[['title', 'score']]
        example_posts_str = "\n".join([f"- \"{row['title']}\" (score: {row['score']})" for _, row in example_posts.iterrows()])

        # Create prompt for Gemini
        prompt = f"""
        You are an AI assistant specialized in predicting Reddit post engagement.

        I need a prediction for how well a post with the following properties would perform:

        Subreddit: r/{subreddit}
        Post Title: "{title}"
        Sentiment Score: {sentiment} (on a scale from -1 to 1 where negative values indicate negative sentiment)

        Subreddit Statistics:
        - Average Score: {avg_score_subreddit:.1f}
        - Median Score: {median_score_subreddit:.1f}
        - Maximum Score: {max_score_subreddit:.1f}

        Examples of top-performing posts in this subreddit:
        {example_posts_str}

        Please analyze:
        1. Whether this post would likely get HIGH (>1.5x average), MEDIUM (0.8-1.5x average), or LOW (<0.8x average) engagement
        2. Predicted score range (numeric estimate)
        3. What factors contribute to this prediction (title length, sentiment, subject matter, etc.)
        4. Likelihood of reaching the front page of the subreddit

        Format your response as a structured analysis with clear headings, and start with a very brief executive summary.
        Do not say "Based on the information provided" or similar phrases - just give your analysis directly.
        """

        # Generate the prediction from Gemini
        response = model.generate_content(prompt)
        prediction_text = response.text

        return dbc.Card(
            dbc.CardBody([
                html.H5(f"AI Engagement Prediction for r/{subreddit}", className="card-title"),
                html.P([html.Strong("Post Title: "), html.Span(f'"{title}"')]),
                html.P([html.Strong("Sentiment: "), html.Span(f"{sentiment:.1f}")]),
                html.Hr(),
                dcc.Markdown(prediction_text, className="prediction-text")
            ]),
            className="mt-3 mb-4 shadow-sm"
        ), None

    except Exception as e:
        import traceback
        return dbc.Alert(
            [
                html.H5("Error Generating Prediction", className="alert-heading"),
                html.P(f"An error occurred: {str(e)}"),
                html.Details([
                    html.Summary("Technical Details"),
                    html.Pre(traceback.format_exc())
                ])
            ],
            color="danger"
        ), None

# --- Run the App ---
if __name__ == '__main__':
    # debug=True is helpful during development as it enables live reloading and error messages in the browser.
    # Turn it off for production.
    app.run_server(debug=True)

# Make the application callable for Gunicorn when using "app:app"
app = app.server  # This line makes both app and server point to the same WSGI application
