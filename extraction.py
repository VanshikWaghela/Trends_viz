import json
import pandas as pd
from datetime import datetime
import os

def load_and_process_reddit_data(file_path):
    # Read the JSONL file
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError:
                    print(f"Error parsing line: {line[:100]}...")

    # If no data was loaded, return an empty DataFrame
    if not data:
        return pd.DataFrame()

    # Extract all possible keys from the first record to establish our columns
    processed_posts = []

    for post in data:
        # Check if this is a post (t3) and has data field
        if "data" in post:
            # Flatten all the nested data for each post
            flat_post = {}

            # Include the 'kind' field from the parent object
            flat_post['kind'] = post.get('kind')

            # Iterate through all keys in the data dictionary
            for key, value in post["data"].items():
                # Handle simple types directly
                if isinstance(value, (str, int, float, bool)) or value is None:
                    flat_post[key] = value
                # For complex types (lists, dicts), convert to JSON strings
                else:
                    flat_post[f"{key}_json"] = json.dumps(value)

            # Add timestamp for convenience
            if 'created_utc' in post["data"]:
                try:
                    flat_post['timestamp'] = datetime.fromtimestamp(
                        post["data"]['created_utc']
                    ).strftime('%Y-%m-%d %H:%M:%S')
                except (TypeError, ValueError):
                    flat_post['timestamp'] = None

            processed_posts.append(flat_post)

    # Create DataFrame
    df = pd.DataFrame(processed_posts)

    return df

def main():
    input_path = 'data.jsonl'
    output_dir = 'output'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process the data
    df = load_and_process_reddit_data(input_path)

    if df.empty:
        print("No data was loaded or processed.")
        return

    # Print column names to verify we got everything
    print(f"Total columns extracted: {len(df.columns)}")
    print("Columns:")
    for col in sorted(df.columns):
        print(f"  - {col}")

    # Save to CSV
    output_path = os.path.join(output_dir, 'reddit_complete_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Processed {len(df)} posts and saved to {output_path}")

    # Basic summary
    print("\nBasic Summary:")
    print(f"Total posts: {len(df)}")
    if 'timestamp' in df.columns:
        print(f"Date range: {df['timestamp'].min() if not df['timestamp'].isna().all() else 'NA'} to {df['timestamp'].max() if not df['timestamp'].isna().all() else 'NA'}")
    if 'author' in df.columns:
        print("Top authors:")
        print(df['author'].value_counts().head(5))

if __name__ == "__main__":
    main()
