from langchain.prompts import PromptTemplate

#prompt for individual review classification
review_prompt = PromptTemplate.from_template("""
You are a customer feedback analyst for restaurant reviews.

Review: "{review}"

1. Summarize the main points in 1 sentence.
2. Identify the main issue type: [Food Quality, Service, Cleanliness, Ambience, Price, Wait Time, Other, None].
3. Based on the tone and severity, assign a priority: [High, Medium, Low, None].
4. Give an overall sentiment analysis of how the reviewer feels towards the restaurant: [Positive, Neutral, Negative]

Respond in JSON format like:
{{
  "summary": "...",
  "issue_type": "...",
  "priority": "...",
  "sentiment": "..."                                           
}}
You are a JSON API. Respond ONLY with valid JSON, with no commentary or explanation.
""")

# Prompt for category-based summary
category_summary_prompt = PromptTemplate.from_template("""
You are analyzing customer reviews for a restaurant. Summarize the feedback by category using only the content of the reviews below.

Reviews:
{reviews}

Return a JSON object like with 2-3 sentence summary from the reviews for each category:
{{
  "food": "...",
  "service": "...",
  "atmosphere": "...",
  "noise": "...",
  "seating": "..."
}}
You are a JSON API. Respond the output strictly with valid JSON, with no commentary, explanation or markdown formatting.
""")