import requests
import json
import time
from tqdm import tqdm
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class BaselineClassifier:
    def __init__(self, model_name, output_dir):
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
        self.output_dir = output_dir

    @staticmethod
    def clean_text(text):
        """Clean HTML tags and other artifacts from text"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www.\S+', '', text)
        return ' '.join(text.split())
    
    @staticmethod
    def extract_json_from_response(response_text):
        """Extract JSON from a potentially verbose response"""
        try:
            # First try parsing the entire text as JSON
            return json.loads(response_text)
        except:
            try:
                # Find JSON-like structure in the text
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
            except:
                pass
        return None

    def classify_post(self, post, max_retries=3):
        """Classify a post without user profile context"""
        for attempt in range(max_retries):
            try:
                text = self.clean_text(post['text'])
                
                # Standard prompt without user profile - exactly the same as in the context version
                system_prompt = """You are a person with Left political stance. Analyze the following discussion group post and classify the author's political orientation. 
Provide your response in this exact JSON format:
{
    "orientation": "LEFT|RIGHT|UNKNOWN",
    "explanation": "A detailed explanation of why you chose this classification based on the content"
}"""
                
                data = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": f"Author: {post['name']}\nPost content: {text}"
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 250
                }

                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=data,
                    timeout=30
                )

                if response.status_code != 200:
                    # Handle rate limits with exponential backoff
                    if "429" in str(response.status_code) or "rate limit" in str(response.text).lower():
                        wait_time = min(60 * (2 ** attempt), 600)  # Cap at 10 minutes
                        print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    
                    if attempt < max_retries - 1:
                        time.sleep((attempt + 1) * 5)
                        continue
                        
                    return {
                        "author": post['name'],
                        "orientation": "unknown",
                        "true_orientation": self.map_polafil_to_orientation(post['polafil']),
                        "raw_response": f"API Error: {response.status_code}",
                        "explanation": f"API Error: {response.status_code} - {response.text}"
                    }

                response_data = response.json()
                model_response = response_data['choices'][0]['message']['content'].strip()

                # Try to extract JSON from response
                parsed_response = self.extract_json_from_response(model_response)
                if parsed_response:
                    true_orientation = self.map_polafil_to_orientation(post['polafil'])
                    return {
                        "author": post['name'],
                        "orientation": parsed_response.get("orientation", "unknown").lower(),
                        "true_orientation": true_orientation,
                        "raw_response": model_response,
                        "explanation": parsed_response.get("explanation", "No explanation provided")
                    }
                
                # Fallback to text parsing if JSON extraction fails
                orientation = "unknown"
                if 'left' in model_response.lower() and 'right' not in model_response.lower():
                    orientation = "left"
                elif 'right' in model_response.lower() and 'left' not in model_response.lower():
                    orientation = "right"
                
                true_orientation = self.map_polafil_to_orientation(post['polafil'])
                return {
                    "author": post['name'],
                    "orientation": orientation,
                    "true_orientation": true_orientation,
                    "raw_response": model_response,
                    "explanation": model_response
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 5)
                    continue
                return {
                    "author": post['name'],
                    "orientation": "unknown",
                    "true_orientation": self.map_polafil_to_orientation(post['polafil']),
                    "raw_response": str(e),
                    "explanation": f"Error during processing: {str(e)}"
                }

    def process_posts(self, posts, subset_size=None):
        """Process all posts and generate metrics"""
        # Use subset if specified
        posts_to_process = posts[:subset_size] if subset_size else posts
        print(f"\nProcessing {len(posts_to_process)} posts using {self.model_name}...")
        
        classifications = []
        try:
            for post in tqdm(posts_to_process):
                classification = self.classify_post(post)
                classifications.append(classification)
                time.sleep(2)  # Rate limiting
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user. Saving results so far...")

        # Calculate metrics
        metrics = self.calculate_metrics(classifications)
        
        # Prepare output
        output = {
            "model_name": self.model_name,
            "total_posts_processed": len(classifications),
            "metrics": metrics,
            "classifications": classifications
        }

        # Save results
        filename = f"baseline_{self.model_name.replace('/', '_')}.json"
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        # Print summary
        print(f"\nResults saved to {output_path}")
        print("\nMetrics Summary:")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
        
        for orientation in ['right', 'left', 'unknown']:
            print(f"\n{orientation.upper()}:")
            print(f"Precision: {metrics[orientation]['precision']:.2f}%")
            print(f"Recall: {metrics[orientation]['recall']:.2f}%")
            print(f"F1 Score: {metrics[orientation]['f1_score']:.2f}%")

        return output

    @staticmethod
    def map_polafil_to_orientation(polafil):
        """Map political affiliations to orientations"""
        polafil_score = {
            'democrat': -1,
            'republican': 1,
            'liberal': -1,
            'conservative': 1,
            'green': -1,
            'l-fringe': -1,
            'r-fringe': 1,
            'libertarian': 1,
            'independent': 0,
            'centrist': 0,
            'unknown': 0
        }
        
        score = polafil_score.get(polafil.lower(), 0)
        if score < 0:
            return 'left'
        elif score > 0:
            return 'right'
        return 'unknown'

    def calculate_metrics(self, classifications):
        """Calculate comprehensive metrics for baseline classifier"""
        if not classifications:
            print("No data to calculate metrics")
            return {
                'total_predictions': 0,
                'overall_accuracy': 0,
                'right': {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0, 'total': 0, 'correct': 0},
                'left': {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0, 'total': 0, 'correct': 0},
                'unknown': {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0, 'total': 0, 'correct': 0}
            }

        # Initialize confusion matrix
        # Format: [predicted][actual]
        confusion_matrix = {
            'right': {'right': 0, 'left': 0, 'unknown': 0},
            'left': {'right': 0, 'left': 0, 'unknown': 0},
            'unknown': {'right': 0, 'left': 0, 'unknown': 0}
        }

        # Fill the confusion matrix
        for clf in classifications:
            pred_orientation = clf['orientation'].lower()
            true_orientation = clf['true_orientation']
            
            # Make sure orientations are valid
            if pred_orientation not in ['right', 'left', 'unknown']:
                pred_orientation = 'unknown'
            if true_orientation not in ['right', 'left', 'unknown']:
                true_orientation = 'unknown'
            
            # Update confusion matrix
            confusion_matrix[pred_orientation][true_orientation] += 1

        def safe_divide(n, d):
            return (n / d * 100) if d > 0 else 0

        # Calculate metrics for each class
        metrics = {}
        total_correct = 0
        total_samples = len(classifications)
        
        for orientation in ['right', 'left', 'unknown']:
            # True positives: correctly predicted this class
            tp = confusion_matrix[orientation][orientation]
            
            # False positives: predicted this class but was actually another class
            fp = sum(confusion_matrix[orientation].values()) - tp
            
            # False negatives: predicted another class but was actually this class
            fn = sum(row[orientation] for label, row in confusion_matrix.items() if label != orientation)
            
            # True negatives: correctly predicted not this class
            tn = total_samples - tp - fp - fn
            
            # Class totals
            class_total = sum(row[orientation] for row in confusion_matrix.values())
            class_correct = tp
            
            # Update total correct predictions
            total_correct += tp
            
            # Calculate precision, recall, F1, accuracy
            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, tp + fn)
            f1 = safe_divide(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0
            accuracy = safe_divide(class_correct, class_total) if class_total > 0 else 0
            
            metrics[orientation] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'total': class_total,
                'correct': class_correct,
                'true_positive': tp,
                'false_positive': fp,
                'true_negative': tn,
                'false_negative': fn
            }

        # Calculate overall accuracy
        metrics['total_predictions'] = total_samples
        metrics['overall_accuracy'] = safe_divide(total_correct, total_samples)
        
        # Add confusion matrix to metrics for additional analysis
        metrics['confusion_matrix'] = confusion_matrix

        return metrics

# Main execution for testing the baseline approach
def main():
    # File paths
    remaining_data_file = "remaining_0.8_posts.json"
    
    # List of models to test - should match what you use in the context-enhanced version
    models = [
        "meta-llama/llama-3.1-70b-instruct",
        "mistralai/mistral-small-24b-instruct-2501",
        "anthropic/claude-3.7-sonnet",
        "qwen/qwen-2.5-72b-instruct",
        "x-ai/grok-2-1212",
        "openai/gpt-4o-mini"
    ]
    
    # Prompt for output directory
    output_dir = input("Enter the output directory: ")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the data
    try:
        with open(remaining_data_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        with open(remaining_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # Set how many posts to process (useful for testing)
    subset_size = 200  # Should match what you use in the context-enhanced version
    
    for model_name in models:
        print(f"\n\nTesting model (baseline): {model_name}")
        # Initialize the classifier
        classifier = BaselineClassifier(model_name, output_dir)
        # Process the posts
        results = classifier.process_posts(data['posts'], subset_size)
    
    print("\nBaseline classification complete!")

if __name__ == "__main__":
    main()