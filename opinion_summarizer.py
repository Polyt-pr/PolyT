import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from difflib import SequenceMatcher

# Load environment variables
load_dotenv()

# Access the API key
api_key = os.getenv('OPENAI_API_KEY')

# Create an OpenAI client
client = OpenAI(api_key=api_key)

sentiment_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "boolean",
                    "description": "True if in favor of the proposition, false if against"
                },
                "core_reason": {
                    "type": "string",
                    "description": "core argument in sentiment analysis"
                },
                "nuance": {
                    "type": "string",
                    "description": "A new nuance for the core reason, if applicable; empty string if not"
                }
            },
            "required": ["sentiment", "core_reason", "nuance"],
            "additionalProperties": False
        }
    }
}

def clear_sentiment_results(file_name):
    file_path = "results/" + file_name
    try:
        with open(file_path, 'w') as f:
            json.dump([], f)
        print(f"All objects removed from {file_path}. The file now contains an empty list.")
    except Exception as e:
        print(f"An error occurred while clearing the file: {str(e)}")

def store_result(result, file_name):
    file_path = "results/" + file_name
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
    
    results.append(result)
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)

def analyze_opinion(question, opinion, aggregated_results):
    try:
        previous_results_str = ", ".join([f"{item['core_reason']} ({item['frequency']} times)" for item in aggregated_results])

        system_message = f"""You are a helpful assistant analyzing opinions on the question: '{question}'.
        Your task is to categorize the given opinion into an appropriate core reason.

        Overall Guidelines:
        1. Determine the sentiment (true if the opinion is for the argument, false if it is against).
        2. Identify a core reason (7-12 words) that captures the essence of the opinion.

        Guidelines for core reasons:
        1. Focus on the primary impact or concern, not specific examples or mechanisms.
        2. Categorize this opinion into an existing core reason if it fits well, or create a new core reason if it's significantly different.
        3. Aim to keep the number of core reasons minimal while ensuring each reason is distinct and meaningful. 
        4. Remember: If a new opinion closely matches an existing core reason, use the exact wording of that reason to maintain consistency.

        Previous core reasons and their frequencies:
        {previous_results_str}

        Provide your response in the required JSON format, including sentiment and core_reason. Leave the nuance field empty for now.

        Opinion: {opinion}
        """
        initial_response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": opinion}
            ],
            response_format=sentiment_schema,
            temperature=0.1
        )

        initial_result = json.loads(initial_response.choices[0].message.content)
        core_reason = initial_result['core_reason']
        sentiment = initial_result['sentiment']
        
        existing_reason = next((item for item in aggregated_results if item['core_reason'] == core_reason), None)
        existing_nuances = existing_reason['nuances'] if existing_reason else []

        nuance_message = f"""Now that you've identified the core reason as "{core_reason}", 
        analyze if a new nuance is needed.

        Original opinion: {opinion} and its sentiment {sentiment}

        Instructions:
        1. Carefully analyze the original opinion and compare it to the existing nuances.

        Here are the existing nuances for this reason, if any:
        {json.dumps(existing_nuances, indent=2)}

        2. Determine if the opinion presents a different perspective or important detail not covered by existing nuances. If no previous nuances exist, generate nuance.
        - REMEMBER: The goal is to represent the opinion wholly, be conservative in assumptions.
        - Don't need to reiterate the original opinion, explicitly state only nuance
        
        3. If new perspective available, compare generated nuance to given core reason and sentiment to ensure the topic matches.

        4. Output:
           - If the opinion offers a significant new perspective, provide a concise new nuance (max 15 words).
           - If the opinion doesn't add significant new information, output an empty string.

        Provide your response as a simple string (the new nuance or an empty string).
        """

        nuance_response = client.chat.completions.create(
            model="gpt-4o-2024-08-06", 
            messages=[
                {"role": "system", "content": nuance_message},
                {"role": "user", "content": opinion}
            ],
            temperature=0.1,
            max_tokens=150
        )

        new_nuance = nuance_response.choices[0].message.content.strip()
        new_nuance = new_nuance.strip('"').replace('\\"', '"')
        initial_result['nuance'] = new_nuance

        return initial_result

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {str(e)}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def is_similar(a, b, threshold=0.7):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

def update_core_reasons(result, aggregated_results):
    core_reason = result['core_reason']
    sentiment = result['sentiment']
    new_nuance = result['nuance']
    
    existing_reason = next((item for item in aggregated_results if item["core_reason"] == core_reason), None)
    
    if existing_reason is None:
        aggregated_results.append({
            "core_reason": core_reason,
            "sentiment": sentiment,
            "frequency": 1,
            "nuances": [new_nuance] if new_nuance else []
        })
    else:
        existing_reason["frequency"] += 1
        if new_nuance:
            existing_reason["nuances"].append(new_nuance)
    
    return aggregated_results

def save_aggregated_results(results, file_name):
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=2)

def main(opinions, original_question):
    individual_results_file = "sentiment_results.json"
    aggregated_results_file = "results/aggregated_sentiment_results.json"
    
    # Clear previous results
    clear_sentiment_results("sentiment_results.json")
    clear_sentiment_results("aggregated_sentiment_results.json")

    aggregated_results = {
        "question": original_question,
        "total_opinions": 0,
        "core_reasons": []
    }
    
    for opinion in opinions:
        result = analyze_opinion(original_question, opinion, aggregated_results["core_reasons"])
        if result:
            store_result(result, individual_results_file)
            aggregated_results["core_reasons"] = update_core_reasons(result, aggregated_results["core_reasons"])
            aggregated_results["total_opinions"] += 1
    
    # Save aggregated results
    save_aggregated_results(aggregated_results, aggregated_results_file)

    print("Sentiment analysis complete. Results have been saved to the respective files.")

if __name__ == "__main__":
    # Example usage
    question = "Should phones be banned from schools?"
    opinions = [
    "Just as more kids began spending more time with their phones, we saw a massive spike in depression and mental illness.",
    "Phones prevent socialization between students during school.",
    "Despite what rules may exist, most students are using their phones during school",
    "Phone usage reduces learning",
    "Having a smartphone with you at all times gives you the ability to instantly communicate with someone else. Students are able to contact parents, guardians or the authorities without much hassle and vice versa.",
    "Smartphones are all about the apps and the amazing things they can do, these apps can be used in a number of creative ways to facilitate their classroom learning experience.",
    "With phones, students can access research, news and videos to enhance their learning.",
    "Smartphones can be utilized for digital harassment in and out of school.",
    "Students could become highly distracted from the many sources of entertainment.",
    "Frequent usage of smartphones has been linked to negative effects on both physical and mental health.",
    "Banning phones would eliminate the problem of cyberbullying during school hours.",
    "The presence of phones in school creates inequality between students who can and cannot afford them.",
    "Phones are essential for students with certain medical conditions to monitor their health.",
    "The use of phones in schools prepares students for the technology-driven workforce they'll enter.",
    "Phones can be used to cheat on tests and assignments, compromising academic integrity.",
    "Banning phones would make it harder for students to coordinate after-school activities and rides home.",
    "Phones can be used to document bullying or other inappropriate behavior in schools.",
    "Allowing phones in school teaches students responsible use of technology.",
    "Banning phones would reduce the risk of theft and property damage in schools.",
    "The radiation from multiple phones in a classroom could potentially be harmful to health.",
    "Phones can be disruptive when they ring or vibrate during class.",
    "Banning phones would make it harder for students to balance part-time jobs and school responsibilities.",
    "Phones can be used to take photos of notes and assignments, helping students stay organized.",
    "Allowing phones in school helps bridge the digital divide for students without internet access at home.",
    "The use of phones in school undermines the authority of teachers and school administration."
    ]
    main(opinions, question)