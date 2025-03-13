import requests
import json
import time
import os
import random
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PoliticalPostGenerator:
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
        
        # Define political affiliations and their orientations
        self.political_affiliations = [
            'democrat', 'republican', 'liberal', 'conservative', 'green', 
            'l-fringe', 'r-fringe', 'libertarian', 'independent', 'centrist'
        ]
        
        # Define topics for post generation
        self.topics = [
                        "Healthcare",
                        "Immigration",
                        "Climate Change",
                        "Gun Control",
                        "Economic Policy",
                        "Education",
                        "Foreign Policy",
                        "COVID-19 Response",
                        "LGBTQ+ Rights",
                        "Abortion",
                        "Criminal Justice Reform",
                        "Military Spending",
                        "Social Security",
                        "Tax Policy",
                        "Environmental Regulation",
                        "Big Tech Regulation",
                        "Free Speech",
                        "Voting Rights",
                        "Energy Policy",
                        "Drug Policy",
                        "AI & Automation",
                        "Minimum Wage",
                        "Universal Basic Income",
                        "Workers' Rights & Unions",
                        "Cancel Culture",
                        "Welfare Reform",
                        "Space Exploration & Colonization",
                        "Nuclear Energy",
                        "Police Reform",
                        "Racial Justice",
                        "Housing Crisis",
                        "Affirmative Action",
                        "Religious Freedom",
                        "Net Neutrality",
                        "U.S.-China Relations",
                        "Russia-Ukraine Conflict",
                        "Israel-Palestine Conflict",
                        "Domestic Terrorism",
                        "Mental Health Policy",
                        "Reproductive Rights",
                        "Cryptocurrency Regulation",
                        "Corporate Taxes & Loopholes",
                        "Sex Work Legalization",
                        "Surveillance & Privacy",
                        "Gun Rights vs. Gun Control",
                        "School Curriculum & Censorship",
                        "Food Security & Agriculture Policy",
                        "Transportation Infrastructure",
                        "Indigenous Rights",
                        "Corporate Influence in Politics",
                        "Healthcare Price Transparency",
                        "Water Rights & Drought Policy",
                        "NATO & Global Alliances",
                        "Transgender Rights & Sports",
                        "Fossil Fuel Industry Regulation",
                        "Public Transportation & High-Speed Rail",
                        "Gerrymandering & Redistricting",
                        "Gun Violence Prevention Programs",
                        "Military Draft & Conscription",
                        "Student Loan Forgiveness",
                        "Automation & Job Displacement",
                        "Cybersecurity & Digital Warfare",
                        "Antitrust Laws & Corporate Monopolies",
                        "Human Trafficking & Exploitation",
                        "Sex Education in Schools",
                        "Prison Reform & Mass Incarceration",
                        "Climate Refugees & Migration",
                        "Children's Online Safety & Regulation",
                        "Universal Healthcare vs. Private Insurance",
                        "Media Bias & Fake News",
                        "Opioid Crisis & Drug Decriminalization",
                        "Labor Shortages & Immigration Reform",
        ]

    def generate_post(self, topic, political_affiliation, confidence_level="high", max_retries=3):
        """Generate a political post on a specific topic with a given political affiliation and confidence level"""
        for attempt in range(max_retries):
            try:
                # Adjust the prompt based on confidence level
                if confidence_level == "high":
                    confidence_instruction = """The post should:
1. Be 1-2 sentences long
2. Express a STRONG and CONFIDENT opinion aligned with {political_affiliation} values
3. Use definitive language (e.g., "definitely", "certainly", "always", "never")
4. Sound assertive and convinced
5. Make clear, unequivocal statements
6. Include strong emotion or conviction
7. Be specific to the topic of {topic}"""
                    temperature = 0.5  # Lower temperature for more deterministic output
                    
                else:  # low confidence
                    confidence_instruction = """The post should:
1. Be 1-2 sentences long
2. Express a TENTATIVE or UNCERTAIN opinion aligned with {political_affiliation} values
3. Use hedging language (e.g., "maybe", "possibly", "sometimes", "I think", "I wonder if")
4. Include questions or express doubt
5. Acknowledge complexity or multiple sides
6. Sound unsure or contemplative
7. Be specific to the topic of {topic}"""
                    temperature = 0.85  # Higher temperature for more variable output
                
                # Format the instruction
                system_content = f"""You will generate a realistic social media post from someone with a {political_affiliation} political viewpoint on the topic of {topic}. 
                
{confidence_instruction.format(political_affiliation=political_affiliation, topic=topic)}

IMPORTANT: Your response must be ONLY valid JSON with no additional text or explanation.
Provide your response in this exact JSON format:
{{
    "text": "The social media post content",
    "name": "A username that relates to the topic",
}}"""

                data = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_content
                        },
                        {
                            "role": "user",
                            "content": f"Create a {confidence_level}-confidence {political_affiliation} social media post about {topic}."
                        }
                    ],
                    "temperature": temperature,
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
                        "error": f"API Error: {response.status_code}",
                        "topic": topic,
                        "polafil": political_affiliation,
                        "confidence": confidence_level
                    }

                response_data = response.json()
                model_response = response_data['choices'][0]['message']['content'].strip()

                # Extract JSON from response
                try:
                    # Find JSON-like structure in the text
                    json_start = model_response.find('{')
                    json_end = model_response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = model_response[json_start:json_end]
                        post_data = json.loads(json_str)
                        
                        return {
                            "thread": 10000 + (self.topics.index(topic) * 100) + 1,
                            "topic": topic,
                            "index": self.topics.index(topic) + 1,
                            "date": "2025/03/15 @",
                            "uid": 8000 + random.randint(1, 1000),
                            "name": post_data.get("name", f"User{random.randint(1000, 9999)}"),
                            "polafil": political_affiliation,
                            "text": post_data.get("text", "No text generated"),
                            "textparts": f"t{10000 + (self.topics.index(topic) * 100) + 1}p{self.topics.index(topic) + 1:04d}"
                        }
                except Exception as e:
                    print(f"Error parsing JSON: {e}")
                    if attempt < max_retries - 1:
                        time.sleep((attempt + 1) * 5)
                        continue

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 5)
                    continue
                return {
                    "error": str(e),
                    "topic": topic,
                    "polafil": political_affiliation,
                    "confidence": confidence_level
                }
                
        # If all retries failed
        return {
            "thread": 10000 + (self.topics.index(topic) * 100) + 1,
            "topic": topic,
            "index": self.topics.index(topic) + 1,
            "date": "2025/03/15 @",
            "uid": 8000 + random.randint(1, 1000),
            "name": f"User{random.randint(1000, 9999)}",
            "polafil": political_affiliation,
            "text": f"This is a placeholder {confidence_level}-confidence post about {topic} from a {political_affiliation} perspective.",
            "textparts": f"t{10000 + (self.topics.index(topic) * 100) + 1}p{self.topics.index(topic) + 1:04d}"
        }

    def generate_mixed_posts_with_confidence(self, count=20, confidence_level="high"):
        """Generate a specified number of posts across different topics and political affiliations with specified confidence level"""
        print(f"\nGenerating {count} {confidence_level}-confidence posts using {self.model_name}...")
        
        # Distribute posts across topics and affiliations 
        # ensuring a balance of political orientations
        posts = []
        
        # Ensure we have at least one of each affiliation type
        all_affiliations = []
        for i in range(count):
            # For balanced left/right representation
            if i < count // 2:
                # Left-leaning
                affiliation = random.choice(['democrat', 'liberal', 'green', 'l-fringe'])
            else:
                # Right-leaning or neutral
                affiliation = random.choice(['republican', 'conservative', 'r-fringe', 'libertarian', 'independent', 'centrist'])
            
            all_affiliations.append(affiliation)
            
        # Randomly select topics (can have repeats)
        selected_topics = random.choices(self.topics, k=count)
        
        # Generate posts
        for i, (topic, affiliation) in tqdm(enumerate(zip(selected_topics, all_affiliations))):
            post = self.generate_post(topic, affiliation, confidence_level)
            posts.append(post)
            time.sleep(2)  # Rate limiting
            
        # Prepare output
        output = {
            "posts": posts
        }

        # Save results
        filename = f"synthetic{count}Posts_{confidence_level}_confidence_{self.model_name.split('/')[-1]}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {filename}")
        return output

    def generate_topic_posts_with_confidence(self, topic, count=20, confidence_level="high"):
        """Generate posts for a specific topic with varied political affiliations and specified confidence level"""
        print(f"\nGenerating {count} {confidence_level}-confidence posts about {topic} using {self.model_name}...")
        
        # Distribute political affiliations for balance
        # Left-leaning affiliations
        left_affiliations = ['democrat', 'liberal', 'green', 'l-fringe']
        # Right-leaning affiliations
        right_affiliations = ['republican', 'conservative', 'r-fringe', 'libertarian'] 
        # Neutral affiliations
        neutral_affiliations = ['independent', 'centrist']
        
        # Calculate distribution: 40% left, 40% right, 20% neutral
        left_count = int(count * 0.4)
        right_count = int(count * 0.4)
        neutral_count = count - left_count - right_count
        
        # Select affiliations
        selected_affiliations = (
            random.choices(left_affiliations, k=left_count) +
            random.choices(right_affiliations, k=right_count) +
            random.choices(neutral_affiliations, k=neutral_count)
        )
        random.shuffle(selected_affiliations)  # Randomize order
        
        # Generate posts
        posts = []
        for i, affiliation in tqdm(enumerate(selected_affiliations)):
            post = self.generate_post(topic, affiliation, confidence_level)
            posts.append(post)
            time.sleep(2)  # Rate limiting
            
        # Prepare output
        output = {
            "model_name": self.model_name,
            "topic": topic,
            "confidence_level": confidence_level,
            "total_posts": len(posts),
            "posts": posts
        }

        # Save results
        filename = f"synthetic{count}Posts_{confidence_level}_confidence_{topic.replace(' ', '_')}_{self.model_name.split('/')[-1]}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {filename}")
        return output

# Example usage:
def generate_posts(model_name, mode="mixed", topic=None, count=20, confidence_level="high"):
    generator = PoliticalPostGenerator(model_name)
    
    if mode == "mixed":
        return generator.generate_mixed_posts_with_confidence(count, confidence_level)
    elif mode == "topic" and topic:
        return generator.generate_topic_posts_with_confidence(topic, count, confidence_level)
    else:
        raise ValueError("Invalid mode or missing topic")

# Models to use for generation
models = [
    "deepseek/deepseek-r1:free",
]

# Generate posts
if __name__ == "__main__":
    # Generate both high and low confidence posts across mixed topics
    for model_name in models:
        # Generate high confidence posts
        generate_posts(model_name, mode="mixed", count=20, confidence_level="high")
        
        # Generate low confidence posts
        generate_posts(model_name, mode="mixed", count=20, confidence_level="low")
    
    # Optional: Generate topic-specific posts with different confidence levels
    # topics_to_generate = ["Healthcare", "Gun Control"]
    # for topic in topics_to_generate:
    #     generate_posts(models[0], mode="topic", topic=topic, count=20, confidence_level="high")
    #     generate_posts(models[0], mode="topic", topic=topic, count=20, confidence_level="low")