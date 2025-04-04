import json

def filter_unknown_posts(input_file, output_file):
    # Read the original JSON file
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    # Filter out posts with polafil: "unknown"
    filtered_posts = [post for post in data['posts'] if post['polafil'].lower() != 'unknown']
    
    # Create a new dictionary with filtered posts
    filtered_data = {
        'posts': filtered_posts
    }
    
    # Write the filtered data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    # Print some stats
    print(f"Original number of posts: {len(data['posts'])}")
    print(f"Number of posts after filtering: {len(filtered_posts)}")
    print(f"Number of posts removed: {len(data['posts']) - len(filtered_posts)}")

# Usage
input_file = "posts_201908161514.json"  # Replace with your input file name
output_file = "filtered_posts_no_unknown.json"

filter_unknown_posts(input_file, output_file)