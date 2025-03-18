import requests
import json
import time
from tqdm import tqdm
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PoliticalClassifier:
    def __init__(self, model_name):
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
            # Find JSON-like structure in the text
            json_start = response_text.rfind('{')  # Find the last occurrence of '{'
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        return None

    def classify_post(self, post, max_retries=3):
        """Classify a single post"""
        for attempt in range(max_retries):
            try:
                text = self.clean_text(post['text'])
    #You are a person with Right political stance. 
    #Analyze the following discussion group post and classify the author's political orientation
                token_limit = 250
                if "deepseek" in self.model_name:
                    token_limit = 1000
                
                data = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": """Analyze the following discussion group post and classify the author's political orientation. Provide your response in this exact JSON format:
{
    "orientation": "LEFT|RIGHT|UNKNOWN",
    "explanation": "A detailed explanation of why you chose this classification based on the content"
}"""
                        },
                        {
                            "role": "user",
                            "content": f"Author: {post['name']}\nPost content: {text}"
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000
                }

                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=data,
                    timeout=30
                )

                if response.status_code != 200:
                    if attempt < max_retries - 1:
                        time.sleep((attempt + 1) * 5)
                        continue
                    return {
                        "author": post['name'],
                        "orientation": "unknown",
                        "raw_response": f"API Error: {response.status_code}",
                        "explanation": f"API Error: {response.status_code} - {response.text}"
                    }

                response_data = response.json()
                model_response = response_data['choices'][0]['message']['content'].strip()

                # Try to extract JSON from response
                parsed_response = self.extract_json_from_response(model_response)
                if parsed_response:
                    return {
                        "author": post['name'],
                        "orientation": parsed_response.get("orientation", "unknown").lower(),
                        "raw_response": model_response,
                        "explanation": parsed_response.get("explanation", "No explanation provided")
                    }
                
                # Fallback to text parsing if JSON extraction fails
                orientation = "unknown"
                if 'left' in model_response.lower() and 'right' not in model_response.lower():
                    orientation = "left"
                elif 'right' in model_response.lower() and 'left' not in model_response.lower():
                    orientation = "right"
                
                return {
                    "author": post['name'],
                    "orientation": orientation,
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
                    "raw_response": str(e),
                    "explanation": f"Error during processing: {str(e)}"
                }

    def process_posts(self, posts):
        """Process all posts and generate metrics"""
        print(f"\nProcessing {len(posts)} posts using {self.model_name}...")
        
        classifications = []
        for post in tqdm(posts):
            classification = self.classify_post(post)
            classifications.append(classification)
            time.sleep(2)  # Rate limiting

        # Calculate metrics
        metrics = self.calculate_metrics(classifications, posts)
        
        # Prepare output
        output = {
            "model_name": self.model_name,
            "total_posts": len(posts),
            "metrics": metrics,
            "classifications": classifications
        }

        # Save results
        filename = f"{self.model_name.replace('/', '_')}_cs7980.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        # Print summary
        print(f"\nResults saved to {filename}")
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

    @staticmethod
    def calculate_metrics(classifications, original_posts):
        """Calculate comprehensive metrics including F1 scores, recall, and accuracy for each class"""
        if not classifications or not original_posts:
            print("No data to calculate metrics")
            return {
                'total_predictions': 0,
                'overall_accuracy': 0,
                'right': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0},
                'left': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0},
                'unknown': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
            }

        # Initialize counters for each class
        right_tp, right_fp, right_tn, right_fn = 0, 0, 0, 0
        left_tp, left_fp, left_tn, left_fn = 0, 0, 0, 0
        unk_tp, unk_fp, unk_tn, unk_fn = 0, 0, 0, 0

        # Calculate metrics for each post
        for clf, post in zip(classifications, original_posts):
            pred_orientation = clf['orientation'].lower()
            true_orientation = PoliticalClassifier.map_polafil_to_orientation(post['polafil'])

            # Right classification metrics
            if true_orientation == 'right' and pred_orientation == 'right':
                right_tp += 1
            elif true_orientation != 'right' and pred_orientation == 'right':
                right_fp += 1
            elif true_orientation != 'right' and pred_orientation != 'right':
                right_tn += 1
            elif true_orientation == 'right' and pred_orientation != 'right':
                right_fn += 1

            # Left classification metrics
            if true_orientation == 'left' and pred_orientation == 'left':
                left_tp += 1
            elif true_orientation != 'left' and pred_orientation == 'left':
                left_fp += 1
            elif true_orientation != 'left' and pred_orientation != 'left':
                left_tn += 1
            elif true_orientation == 'left' and pred_orientation != 'left':
                left_fn += 1

            # Unknown classification metrics
            if true_orientation == 'unknown' and pred_orientation == 'unknown':
                unk_tp += 1
            elif true_orientation != 'unknown' and pred_orientation == 'unknown':
                unk_fp += 1
            elif true_orientation != 'unknown' and pred_orientation != 'unknown':
                unk_tn += 1
            elif true_orientation == 'unknown' and pred_orientation != 'unknown':
                unk_fn += 1

        def safe_divide(n, d):
            return (n / d * 100) if d > 0 else 0

        def calculate_class_metrics(tp, fp, tn, fn):
            accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, tp + fn)
            f1 = safe_divide(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positive': tp,
                'false_positive': fp,
                'true_negative': tn,
                'false_negative': fn
            }

        # Calculate metrics for each class
        metrics = {
            'right': calculate_class_metrics(right_tp, right_fp, right_tn, right_fn),
            'left': calculate_class_metrics(left_tp, left_fp, left_tn, left_fn),
            'unknown': calculate_class_metrics(unk_tp, unk_fp, unk_tn, unk_fn),
            'total_predictions': len(classifications)
        }

        # Calculate overall accuracy
        total_correct = sum(1 for clf, post in zip(classifications, original_posts)
                          if clf['orientation'].lower() == PoliticalClassifier.map_polafil_to_orientation(post['polafil']))
        metrics['overall_accuracy'] = safe_divide(total_correct, len(classifications))

        return metrics

# Example usage:
def evaluate_model(model_name, posts):
    classifier = PoliticalClassifier(model_name)
    results = classifier.process_posts(posts)
    return results

# Load and process data
with open('synthetic20Posts_high_confidence_gemini-2.0-flash-001.json', 'r') as f:
    data = json.load(f)

# Models to evaluate
models = [
    "deepseek/deepseek-r1:free",
]

# Run evaluation for each model
for model_name in models:
    results = evaluate_model(model_name, data['posts'])
    print(f"\nFinished evaluating {model_name}\n")