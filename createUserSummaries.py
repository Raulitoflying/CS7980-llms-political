import json
import random
import re
from collections import defaultdict
import time
from tqdm import tqdm
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class UserSummarizer:
    def __init__(self, model_name="google/gemini-2.0-flash-001"):
        # Get API key from environment variables
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Make sure OPENROUTER_API_KEY is set in your .env file.")
            
        self.model_name = model_name
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:5000",
        }
        
        # Store user summaries
        self.user_summaries = {}
    
    @staticmethod
    def clean_text(text):
        """Clean HTML tags and other artifacts from text"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www.\S+', '', text)
        return ' '.join(text.split())
    
    def random_sample_posts(self, data_file, sample_percentage=0.1, seed=42):
        """
        Randomly sample a percentage of posts from the data file
        """
        print(f"Loading data from {data_file}...")
        with open(data_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        posts = data['posts']
        total_posts = len(posts)
        print(f"Loaded {total_posts} posts")
        
        # Calculate sample size
        sample_size = int(total_posts * sample_percentage)
        
        # Randomly select posts
        random.seed(seed)
        sampled_indices = random.sample(range(total_posts), sample_size)
        sample_posts = [posts[i] for i in sampled_indices]
        remaining_posts = [posts[i] for i in range(total_posts) if i not in sampled_indices]
        
        print(f"Randomly sampled {len(sample_posts)} posts ({sample_percentage:.1%} of total)")
        
        # Save sampled posts to file
        sample_file = f"sampled_{sample_percentage:.1f}_posts.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump({"posts": sample_posts}, f, indent=2)
        
        # Save remaining posts to file
        remaining_file = f"remaining_{1-sample_percentage:.1f}_posts.json"
        with open(remaining_file, 'w', encoding='utf-8') as f:
            json.dump({"posts": remaining_posts}, f, indent=2)
        
        print(f"Saved sampled posts to {sample_file}")
        print(f"Saved remaining posts to {remaining_file}")
        
        return sample_posts, remaining_posts, sample_file, remaining_file
    
    def group_posts_by_user(self, posts):
        """
        Group posts by user and return a dictionary
        """
        user_posts = defaultdict(list)
        for post in posts:
            # Clean post text
            post['clean_text'] = self.clean_text(post['text'])
            user_posts[post['name']].append(post)
        
        print(f"Grouped posts for {len(user_posts)} unique users")
        
        # Print some statistics about posts per user
        post_counts = [len(posts) for posts in user_posts.values()]
        avg_posts = sum(post_counts) / len(post_counts) if post_counts else 0
        max_posts = max(post_counts) if post_counts else 0
        min_posts = min(post_counts) if post_counts else 0
        
        print(f"Average posts per user: {avg_posts:.1f}")
        print(f"Maximum posts per user: {max_posts}")
        print(f"Minimum posts per user: {min_posts}")
        
        return user_posts
    
    def generate_user_summary(self, username, posts):
        """
        Generate a summary of a user's political behavior based on their posts
        With improved error handling for API responses
        """
        # Format posts for the prompt
        posts_text = ""
        for i, post in enumerate(posts[:30]):  # Limit to 30 posts to avoid context overflows
            # Use clean_text if available, otherwise use original text
            post_text = post.get('clean_text', self.clean_text(post['text']))
            # Include topic for context
            posts_text += f"POST {i+1}:\nTopic: {post['topic']}\nContent: {post_text}\n\n"
        
        # Call LLM API to generate summary
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": """Analyze the following set of forum posts by the user and create a concise political profile summary. For this task:
    1. Identify any consistent political indicators in their posts (criticism of specific politicians/parties, stance on issues, etc.)
    2. Note recurring topics this user discusses
    3. Observe distinctive language patterns (formal/informal, emotional/detached, specific phrases)
    4. Identify who/what they consistently criticize or support
    5. Determine if there's sufficient evidence to classify them as LEFT, RIGHT, or UNKNOWN

    Format your response as a JSON object with these fields:
    {
        "username": "the username",
        "political_leaning": "left/right/unknown",
        "confidence": "high/medium/low",
        "key_indicators": ["3-5 specific examples from posts that indicate political leaning"],
        "recurring_topics": ["list frequent topics"],
        "language_style": "brief description of their communication style",
        "sentiment_patterns": "who/what they criticize or support", 
        "context_notes": "any additional relevant information"
    }

    IMPORTANT:
    - Focus on clear patterns rather than isolated statements
    - Maintain objectivity and avoid overinterpreting ambiguous content
    - If there isn't sufficient evidence to determine orientation, mark as "unknown"
    - Ensure your response is a valid JSON object"""
                },
                {
                    "role": "user",
                    "content": f"Username: {username}\n\nPOSTS FOR ANALYSIS:\n{posts_text}"
                }
            ],
            "temperature": 0.2,
            "max_tokens": 800
        }
        
        try:
            print(f"Generating summary for user: {username} with {len(posts)} posts")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=60
            )
            
            # First, check if the response status is ok
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                return {
                    "username": username,
                    "political_leaning": "unknown",
                    "confidence": "low",
                    "error": f"API Error: {response.status_code}",
                    "key_indicators": [],
                    "recurring_topics": [],
                    "language_style": "",
                    "sentiment_patterns": "",
                    "context_notes": f"Error generating summary: {response.text}"
                }
            
            # Next, try to parse the response JSON
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                print(f"Failed to parse API response as JSON: {str(e)}")
                print(f"Response text: {response.text[:200]}...")  # Print first 200 chars of response
                return {
                    "username": username,
                    "political_leaning": "unknown",
                    "confidence": "low",
                    "error": f"JSON parse error: {str(e)}",
                    "key_indicators": [],
                    "recurring_topics": [],
                    "language_style": "",
                    "sentiment_patterns": "",
                    "context_notes": f"Invalid JSON in response"
                }
            
            # Check if 'choices' key exists in the response
            if 'choices' not in response_data:
                print(f"No 'choices' in response. Keys: {list(response_data.keys())}")
                print(f"Response: {json.dumps(response_data)[:200]}...")  # Print first 200 chars
                return {
                    "username": username,
                    "political_leaning": "unknown",
                    "confidence": "low",
                    "error": "Missing 'choices' in API response",
                    "key_indicators": [],
                    "recurring_topics": [],
                    "language_style": "",
                    "sentiment_patterns": "",
                    "context_notes": f"Unexpected API response structure: {list(response_data.keys())}"
                }
            
            if not response_data['choices'] or len(response_data['choices']) == 0:
                print(f"Empty 'choices' array in response")
                return {
                    "username": username,
                    "political_leaning": "unknown",
                    "confidence": "low",
                    "error": "Empty choices array",
                    "key_indicators": [],
                    "recurring_topics": [],
                    "language_style": "",
                    "sentiment_patterns": "",
                    "context_notes": "API returned empty choices array"
                }
            
            # Check if the expected structure is in the first choice
            if 'message' not in response_data['choices'][0]:
                print(f"No 'message' in first choice. Keys: {list(response_data['choices'][0].keys())}")
                return {
                    "username": username,
                    "political_leaning": "unknown",
                    "confidence": "low",
                    "error": "Missing 'message' in API response choice",
                    "key_indicators": [],
                    "recurring_topics": [],
                    "language_style": "",
                    "sentiment_patterns": "",
                    "context_notes": f"Unexpected choice structure: {list(response_data['choices'][0].keys())}"
                }
            
            if 'content' not in response_data['choices'][0]['message']:
                print(f"No 'content' in message. Keys: {list(response_data['choices'][0]['message'].keys())}")
                return {
                    "username": username,
                    "political_leaning": "unknown",
                    "confidence": "low",
                    "error": "Missing 'content' in API response message",
                    "key_indicators": [],
                    "recurring_topics": [],
                    "language_style": "",
                    "sentiment_patterns": "",
                    "context_notes": f"Unexpected message structure: {list(response_data['choices'][0]['message'].keys())}"
                }
            
            # Now we can safely access the content
            summary_text = response_data['choices'][0]['message']['content'].strip()
            
            # Extract JSON from response
            try:
                # Find JSON-like structure in the text
                summary_json = None
                # First try to parse the entire text as JSON
                try:
                    summary_json = json.loads(summary_text)
                except:
                    # If that fails, try to extract JSON part from text
                    json_start = summary_text.find('{')
                    json_end = summary_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = summary_text[json_start:json_end]
                        summary_json = json.loads(json_str)
                
                if summary_json:
                    # Ensure all required fields are present
                    summary_json["username"] = username
                    if "political_leaning" not in summary_json:
                        summary_json["political_leaning"] = "unknown"
                    if "confidence" not in summary_json:
                        summary_json["confidence"] = "low"
                    
                    return summary_json
            except Exception as json_err:
                print(f"JSON parsing error for {username}: {str(json_err)}")
            
            # Fallback if JSON parsing fails
            return {
                "username": username,
                "political_leaning": "unknown",
                "confidence": "low",
                "raw_response": summary_text,
                "key_indicators": [],
                "recurring_topics": [],
                "language_style": "",
                "sentiment_patterns": "",
                "context_notes": "Error parsing summary as JSON"
            }
            
        except Exception as e:
            print(f"Exception for {username}: {str(e)}")
            # Print stack trace for debugging
            import traceback
            traceback.print_exc()
            return {
                "username": username,
                "political_leaning": "unknown",
                "confidence": "low",
                "error": str(e),
                "key_indicators": [],
                "recurring_topics": [],
                "language_style": "",
                "sentiment_patterns": "",
                "context_notes": f"Error during processing: {str(e)}"
            }
    
    def generate_all_user_summaries(self, user_posts_dict, min_posts=3, save_incremental=True):
        """
        Generate summaries with rate limit handling and incremental saving
        """
        print(f"Generating summaries for users with at least {min_posts} posts...")
        
        # Filter users who have enough posts
        qualified_users = {user: posts for user, posts in user_posts_dict.items() if len(posts) >= min_posts}
        print(f"Found {len(qualified_users)} users with at least {min_posts} posts")
        
        # Initialize user_summaries if not already done
        if not hasattr(self, 'user_summaries') or self.user_summaries is None:
            self.user_summaries = {}
        
        # Load any existing summaries if available
        incremental_file = "user_summaries_incremental.json"
        if os.path.exists(incremental_file):
            try:
                with open(incremental_file, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    if 'user_summaries' in saved_data:
                        self.user_summaries = saved_data['user_summaries']
                        print(f"Loaded {len(self.user_summaries)} existing summaries")
            except Exception as e:
                print(f"Error loading incremental file: {e}")
        
        # Identify which users still need processing
        users_to_process = [u for u in qualified_users.keys() if u not in self.user_summaries]
        print(f"Need to process {len(users_to_process)} remaining users")
        
        # Generate summaries
        try:
            for username in tqdm(users_to_process):
                posts = qualified_users[username]
                max_retries = 3
                rate_limit_delay = 60  # Start with 1 minute delay for rate limits
                
                for attempt in range(max_retries):
                    try:
                        result = self.generate_user_summary(username, posts)
                        self.user_summaries[username] = result
                        
                        # Save progress incrementally
                        if save_incremental and len(self.user_summaries) % 5 == 0:
                            with open(incremental_file, 'w', encoding='utf-8') as f:
                                json.dump({"user_summaries": self.user_summaries}, f, indent=2)
                                print(f"\nSaved progress: {len(self.user_summaries)}/{len(qualified_users)} users")
                        
                        # Standard delay between successful requests
                        time.sleep(5)
                        break
                        
                    except Exception as e:
                        error_str = str(e)
                        print(f"Attempt {attempt+1}/{max_retries} failed for {username}: {error_str}")
                        
                        # Check if it's a rate limit error
                        if "429" in error_str or "rate limit" in error_str.lower():
                            print(f"Rate limit detected. Waiting for {rate_limit_delay} seconds...")
                            time.sleep(rate_limit_delay)
                            # Increase backoff for next rate limit encounter
                            rate_limit_delay = min(rate_limit_delay * 2, 600)  # Cap at 10 minutes
                        elif attempt == max_retries - 1:
                            # If all retries failed, store a failure result
                            self.user_summaries[username] = {
                                "username": username,
                                "political_leaning": "unknown",
                                "confidence": "low",
                                "error": f"Failed after {max_retries} attempts: {error_str}",
                                "key_indicators": [],
                                "recurring_topics": [],
                                "language_style": "",
                                "sentiment_patterns": "",
                                "context_notes": "Maximum retries exceeded"
                            }
                        else:
                            # Standard backoff for non-rate-limit errors
                            time.sleep(10 * (attempt + 1))
        
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving current progress...")
            with open(incremental_file, 'w', encoding='utf-8') as f:
                json.dump({"user_summaries": self.user_summaries}, f, indent=2)
            print(f"Saved {len(self.user_summaries)} summaries to {incremental_file}")
        
        return self.user_summaries
    
    def save_user_summaries(self, output_file="user_summaries.json"):
        """
        Save all user summaries to a JSON file
        """
        print(f"Saving {len(self.user_summaries)} user summaries to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"user_summaries": self.user_summaries}, f, indent=2)
        
        # Print some statistics
        political_counts = {"left": 0, "right": 0, "unknown": 0}
        confidence_counts = {"high": 0, "medium": 0, "low": 0}
        
        for summary in self.user_summaries.values():
            leaning = summary.get("political_leaning", "unknown").lower()
            confidence = summary.get("confidence", "low").lower()
            
            if leaning in political_counts:
                political_counts[leaning] += 1
            else:
                political_counts["unknown"] += 1
                
            if confidence in confidence_counts:
                confidence_counts[confidence] += 1
            else:
                confidence_counts["low"] += 1
        
        print("\nSummary Statistics:")
        print(f"Political leanings: {political_counts}")
        print(f"Confidence levels: {confidence_counts}")
        
        return output_file

def main():
    """
    Main function to run the user summarization process
    """
    # Initialize summarizer
    summarizer = UserSummarizer(model_name="google/gemini-2.0-flash-001")
    
    # Random sample 10% of posts
    data_file = "filtered_posts_no_unknown.json"  # Update with your actual filename
    sample_posts, remaining_posts, sample_file, remaining_file = summarizer.random_sample_posts(
        data_file, sample_percentage=0.2
    )
    
    # Group sampled posts by user
    user_posts = summarizer.group_posts_by_user(sample_posts)
    
    # Generate summaries for all users with enough posts
    user_summaries = summarizer.generate_all_user_summaries(user_posts, min_posts=3)
    
    # Save summaries
    output_file = summarizer.save_user_summaries("political_user_summaries.json")
    
    print(f"Process complete. User summaries saved to {output_file}")
    print(f"You can now use these summaries as context when classifying posts in {remaining_file}")

if __name__ == "__main__":
    main()