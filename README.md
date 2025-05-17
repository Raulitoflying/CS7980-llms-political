# CS7980-llms-political

# LLM Political Stance Analysis Project

## Overview
This project analyzes the political stance detection capabilities of different Large Language Models (LLMs) across various political topics. It includes tools for generating political stance classifications, analyzing results, and visualizing performance metrics.

## Project Structure

### Data Organization
- **[Model]Data_Analysis/** - Folders containing analysis files for each model (e.g., DeepseekData_Analysis, GeminiData_Analysis, MistralData_Analysis)
- **[Model]_balancedData/** - Folders containing balanced datasets for each model

### Core Scripts

#### Data Generation and Analysis
- **`generateRouter.py`** - Generates synthetic political posts with specified political leanings using various LLM models
- **`openRouter.py`** - Analyzes political text data using various LLM models through the OpenRouter API and processes classification results and calculates metrics


#### Visualization Tools
- **`visualize_political_data.py`** - Generates visualizations for a single model's performance
- **`visualize_Combined.py`** - Creates comparative visualizations across multiple models
- **`visualize_matrix.py`** - Produces cross-analysis matrices comparing how different models analyze each other's generated data

## API Configuration

### OpenRouter API Setup
1. Create an account at [OpenRouter](https://openrouter.ai/)
2. Generate an API key from your dashboard
3. Create a .env file in the project root with the following:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```
4. Configure the models list in openRouter.py:
   ```python
   models = [
       "anthropic/claude-3-sonnet",
       "google/gemini-1.5-pro",
       "meta-llama/llama-3-70b-instruct",
       "mistral/mistral-large",
       "deepseek/deepseek-coder"
   ]
   ```

### API Request Format
```python
response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
)
```

## JSON Data Structure

### Input Data Format
```json
{
  "posts": [
    {
      "textparts": "t10060p9790"
    },
    {
      "textparts": "t10060p5878"
    }
    // Additional posts...
  ]
}
```
This contains post identifiers that are analyzed by the models.

### Analysis Output Format
```json
{
  "model_name": "deepseek/deepseek-r1",
  "total_posts": 20,
  "metrics": {
    "right": {
      "accuracy": 75.0,
      "precision": 100.0,
      "recall": 50.0,
      "f1_score": 66.67,
      "true_positive": 5,
      "false_positive": 0,
      "true_negative": 10,
      "false_negative": 5
    },
    "left": {
      "accuracy": 90.0,
      "precision": 100.0,
      "recall": 80.0,
      "f1_score": 88.89,
      "true_positive": 8,
      "false_positive": 0,
      "true_negative": 10,
      "false_negative": 2
    },
    "unknown": {
      "accuracy": 65.0,
      "precision": 0.0,
      "recall": 0,
      "f1_score": 0,
      "true_positive": 0,
      "false_positive": 7,
      "true_negative": 13,
      "false_negative": 0
    },
    "total_predictions": 20,
    "overall_accuracy": 65.0
  },
  "classifications": [
    {
      "author": "GreenThoughts4All",
      "orientation": "left",
      "raw_response": "...",
      "explanation": "The author's focus on questioning whether society is 'doing enough to support women and their choices' and framing the issue around 'reproductive freedom' aligns with progressive and left-leaning values..."
    }
    // Additional classifications...
  ],
  "file_path": "Mistral_balancedData/Abortion_Rights/balanced20Posts_low_confidence_Abortion_Rights:_Debates_on_abortion_access_and_reproductive_freedom_mistral-small-24b-instruct-2501.json",
  "output_filename": "balanced20Posts_low_confidence_Abortion_Rights:_Debates_on_abortion_access_and_reproductive_freedom_mistral-small-24b-instruct-2501_deepseek_deepseek-r1_analysis.json"
}
```

### Key JSON Fields Explained:

#### Model Information
- **`model_name`**: The LLM used for analysis (e.g., "deepseek/deepseek-r1")
- **`file_path`**: Path to the original data file analyzed
- **`output_filename`**: Name of the output analysis file

#### Performance Metrics
- **`metrics`**: Contains detailed performance metrics
  - **`overall_accuracy`**: Percentage of correctly classified posts
  - **For each stance** (right, left, unknown):
    - **`accuracy`**: Percentage of correct classifications for this stance
    - **`precision`**: Percentage of correct predictions among all predictions of this stance
    - **`recall`**: Percentage of correctly identified posts of this stance
    - **`f1_score`**: Harmonic mean of precision and recall
    - **`true_positive`**: Count of correctly classified posts of this stance
    - **`false_positive`**: Count of incorrectly classified posts as this stance
    - **`true_negative`**: Count of correctly not classified as this stance
    - **`false_negative`**: Count of incorrectly not classified as this stance
  - **`total_predictions`**: Total number of predictions made

#### Classification Details
- **`classifications`**: List of individual post classifications
  - **`author`**: Username of the post author
  - **`orientation`**: True political stance (left, right, unknown)
  - **`raw_response`**: Full JSON response from the model
  - **`explanation`**: Model's reasoning for the classification

## Usage Instructions

### Analyzing Political Texts with Models
```bash
python openRouter.py
```

You can customize the analysis by modifying these variables in the script:
- `folder_to_analyze`: Directory containing political text data
- `models`: List of OpenRouter model identifiers to use
- `system_prompt`: Instructions for the model
- `confidence_threshold`: Threshold for high/low confidence classification

### Generating Visualizations for Single Model
```bash
python visualize_political_data.py MistralData_Analysis --output visualizations/mistral
```

### Comparing Multiple Models
```bash
python visualize_Combined.py DeepseekData_Analysis GeminiData_Analysis MistralData_Analysis --output visualizations/combined
```

### Creating Cross-Analysis Matrices
```bash
python visualize_matrix.py DeepseekData_Analysis GeminiData_Analysis MistralData_Analysis --output matrix_results
```

## Output Visualizations

### Model Performance Metrics
- Overall accuracy comparison
- F1 scores by political orientation
- Recall rates by political orientation

### Topic-Based Analysis
- Accuracy by political topic
- Precision and recall metrics for different topics
- Model performance across topics (heatmap)

### Confidence Level Analysis
- Performance comparison between high/low confidence predictions
- Cross-model confidence level performance

## Models Analyzed
- Claude (Anthropic)
- Gemini (Google)
- Mistral
- Deepseek
- Llama (Meta)
- Qwen
- Grok
- GPT-4o-mini

## Political Topics Covered
- Gun Control and Second Amendment Rights: Balancing gun ownership rights with public safety
- Abortion Rights: Debates on abortion access and reproductive freedom
- Immigration Reform: Border security, undocumented immigrants, and pathway to citizenship
- Climate Change Policies: Environmental regulations vs. economic growth
- LGBTQ+ and Transgender Rights: Rights in education, healthcare, and public spaces
- Healthcare and Insurance Reform: Universal healthcare, Medicare-for-all, and insurance privatization
- Voting Rights and Election Integrity: Voter ID laws, mail-in voting, gerrymandering
- Economic Inequality: Raising the minimum wage, wealth taxation, and addressing poverty
- Death Penalty: Debates around capital punishment and wrongful convictions
- Drug Legalization and Decriminalization: Marijuana legalization, drug sentencing reform
- Russia-Ukraine War
- NBA

## Requirements
- Python 3.7+
- Required packages: pandas, numpy, matplotlib, seaborn, json, argparse, requests, dotenv
- OpenRouter API access for model inference

## Contributors
- Project for Northeastern University CS7980
- Spring 2025