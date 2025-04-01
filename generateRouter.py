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
        
        # Define political affiliations by orientation
        self.left_affiliations = ['democrat', 'liberal', 'green', 'l-fringe']
        self.right_affiliations = ['republican', 'conservative', 'r-fringe', 'libertarian']
        self.neutral_affiliations = ['independent', 'centrist']
        
        # Combined list of all affiliations
        self.political_affiliations = self.left_affiliations + self.right_affiliations + self.neutral_affiliations
        
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

    def generate_post(self, topic, political_affiliation, confidence_level="high", max_retries=15):
        """Generate a political post on a specific topic with a given political affiliation and confidence level"""
        for attempt in range(max_retries):
            try:
                # Adjust the prompt based on confidence level
                if confidence_level == "high":
                    confidence_instruction = """The post should:
1. Be 1-2 sentences long
2. Express a STRONG and CONFIDENT opinion aligned with {political_affiliation} values
3. Use definitive language
4. Sound assertive and convinced
5. Make clear, unequivocal statements
6. Include strong emotion or conviction
7. Be specific to the topic of {topic}"""
                    temperature = 0.5  # Lower temperature for more deterministic output
                    
                else:  # low confidence
                    confidence_instruction = """The post should:
1. Be 1-2 sentences long
2. Express a TENTATIVE or UNCERTAIN opinion aligned with {political_affiliation} values
3. Use hedging language or qualifiers
4. Include questions or express doubt
5. Acknowledge complexity or multiple sides
6. Sound unsure or contemplative
7. Be specific to the topic of {topic}"""
                    temperature = 0.85  # Higher temperature for more variable output
                
                # Format the instruction
                system_content = f"""You will generate a realistic social media post from someone with a {political_affiliation} political viewpoint on the topic of {topic}. 
                
{confidence_instruction.format(political_affiliation=political_affiliation, topic=topic)}

IMPORTANT: 
1. Your response must be ONLY valid JSON with no additional text or explanation.
2. DO NOT use contractions with apostrophes (like "It is" not "It's", "do not" not "don't").
3. Avoid ALL apostrophes in your text to prevent JSON parsing errors.

Provide your response in this exact JSON format:
{{
    "text": "The social media post content",
    "name": "A username that relates to the topic"
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
                    "max_tokens": 2000
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
                        
                        # Generate thread ID safely even for non-standard topics
                        try:
                            thread_id = 10000 + (self.topics.index(topic) * 100) + 1
                            index = self.topics.index(topic) + 1
                        except ValueError:
                            # For topics not in self.topics
                            thread_id = 10000 + (hash(topic) % 1000)
                            index = hash(topic) % 100
                        
                        return {
                            "thread": thread_id,
                            "topic": topic,
                            "index": index,
                            "date": "2025/03/15 @",
                            "uid": 8000 + random.randint(1, 1000),
                            "name": post_data.get("name", f"User{random.randint(1000, 9999)}"),
                            "polafil": political_affiliation,
                            "text": post_data.get("text", "No text generated"),
                            "textparts": f"t{thread_id}p{random.randint(1, 9999):04d}"
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
        try:
            thread_id = 10000 + (self.topics.index(topic) * 100) + 1
            index = self.topics.index(topic) + 1
        except ValueError:
            # For topics not in self.topics
            thread_id = 10000 + (hash(topic) % 1000)
            index = hash(topic) % 100
            
        return {
            "thread": thread_id,
            "topic": topic,
            "index": index,
            "date": "2025/03/15 @",
            "uid": 8000 + random.randint(1, 1000),
            "name": f"User{random.randint(1000, 9999)}",
            "polafil": political_affiliation,
            "text": f"This is a placeholder {confidence_level}-confidence post about {topic} from a {political_affiliation} perspective.",
            "textparts": f"t{thread_id}p{random.randint(1, 9999):04d}"
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
                affiliation = random.choice(self.left_affiliations)
            else:
                # Right-leaning or neutral
                affiliation = random.choice(self.right_affiliations + self.neutral_affiliations)
            
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
        
        # Calculate distribution: 40% left, 40% right, 20% neutral
        left_count = int(count * 0.4)
        right_count = int(count * 0.4)
        neutral_count = count - left_count - right_count
        
        # Select affiliations
        selected_affiliations = (
            random.choices(self.left_affiliations, k=left_count) +
            random.choices(self.right_affiliations, k=right_count) +
            random.choices(self.neutral_affiliations, k=neutral_count)
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
            "posts": posts
        }

        # Save results
        filename = f"synthetic{count}Posts_{confidence_level}_confidence_{topic.replace(' ', '_')}_{self.model_name.split('/')[-1]}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {filename}")
        return output
    
    def generate_balanced_topic_posts(self, topic, count_per_side=10, confidence_level="high"):
        """Generate posts for a specific topic with balanced left/right representations"""
        print(f"\nGenerating balanced {confidence_level}-confidence posts about '{topic}' using {self.model_name}...")
        
        posts = []
        
        # Generate left-leaning posts
        print(f"  Generating {count_per_side} LEFT-leaning posts...")
        for i in tqdm(range(count_per_side)):
            affiliation = random.choice(self.left_affiliations)
            post = self.generate_post(topic, affiliation, confidence_level)
            posts.append(post)
            time.sleep(2)  # Rate limiting
        
        # Generate right-leaning posts
        print(f"  Generating {count_per_side} RIGHT-leaning posts...")
        for i in tqdm(range(count_per_side)):
            affiliation = random.choice(self.right_affiliations)
            post = self.generate_post(topic, affiliation, confidence_level)
            posts.append(post)
            time.sleep(2)  # Rate limiting
        
        # Prepare output
        output = {
            "posts": posts
        }

        # Save results
        filename = f"balanced{count_per_side*2}Posts_{confidence_level}_confidence_{topic.replace(' ', '_')}_{self.model_name.split('/')[-1]}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {filename}")
        return output

# Example usage:
def generate_posts(model_name, mode="mixed", topic=None, count=20, confidence_level="high", count_per_side=10):
    generator = PoliticalPostGenerator(model_name)
    
    if mode == "mixed":
        return generator.generate_mixed_posts_with_confidence(count, confidence_level)
    elif mode == "topic" and topic:
        return generator.generate_topic_posts_with_confidence(topic, count, confidence_level)
    elif mode == "balanced" and topic:
        return generator.generate_balanced_topic_posts(topic, count_per_side, confidence_level)
    else:
        raise ValueError("Invalid mode or missing topic")

# Models to use for generation
models = [
    "mistralai/mistral-small-24b-instruct-2501",
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-r1",

]

# List of specific topics to generate balanced posts about
specific_topics = [
    "Immigration Reform: Border security, undocumented immigrants, and pathway to citizenship",
    "Climate Change Policies: Environmental regulations vs. economic growth",
    "LGBTQ+ and Transgender Rights: Rights in education, healthcare, and public spaces",
    "Healthcare and Insurance Reform: Universal healthcare, Medicare-for-all, and insurance privatization",
    "Voting Rights and Election Integrity: Voter ID laws, mail-in voting, gerrymandering",
    "Economic Inequality: Raising the minimum wage, wealth taxation, and addressing poverty",
    "Death Penalty: Debates around capital punishment and wrongful convictions",
    "Drug Legalization and Decriminalization: Marijuana legalization, drug sentencing reform",
    "Russia Ukraine war",
    "NBA"
]

# Generate posts
if __name__ == "__main__":
    # Comment out what you don't need
    
    # Option 1: Generate mixed posts (original functionality)
    # for model_name in models:
    #    generate_posts(model_name, mode="mixed", count=20, confidence_level="high")
    #    generate_posts(model_name, mode="mixed", count=20, confidence_level="low")
    
    # Option 2: Generate posts for specific topics with natural distribution
    # for topic in specific_topics:
    #    generate_posts(models[0], mode="topic", topic=topic, count=20, confidence_level="high")
    
    #    generate_posts(models[0], mode="topic", topic=topic, count=20, confidence_level="low")
    
    # Option 3: Generate balanced posts (10 left + 10 right) for specific topics
    # for topic in specific_topics:
        generate_posts(models[0], mode="balanced", topic="Immigration Reform: Border security, undocumented immigrants, and pathway to citizenship", confidence_level="high")
        
        generate_posts(models[0], mode="balanced", topic="Immigration Reform: Border security, undocumented immigrants, and pathway to citizenship", confidence_level="low")