# 1. Word points dictionary (positive and negative words)
word_points = {
    "good": 2,
    "great": 3,
    "delicious": 3,
    "tasty": 2,
    "fresh": 2,
    "fast": 1,
    "friendly": 2,
    "love": 3,
    "bad": -2,
    "terrible": -3,
    "cold": -2,
    "slow": -2,
    "expensive": -1,
    "stale": -2,
    "awful": -3,
    "hate": -3
}

# 2. Sample comments with their true human labels: (comment, actual_mood)
comments_data = [
    ("The pizza was fresh and delicious!", "HAPPY"),
    ("Service was slow and the soup was cold.", "UNHAPPY"),
    ("The food was not good at all.", "UNHAPPY"),
    ("The fries were not bad today.", "HAPPY"),
    ("I got a sandwich for lunch.", "OK"),
    ("Great burger, friendly staff!", "HAPPY"),
    ("The pasta was terrible and expensive.", "UNHAPPY"),
    ("It was not great, just average.", "OK")  # Tricky case to test mistakes!
]

def analyze_mood(comment):
    # Clean punctuation and convert to lowercase
    clean_text = comment.lower().replace(".", "").replace("!", "").replace(",", "")
    words = clean_text.split()
    
    total_score = 0
    
    for i in range(len(words)):
        word = words[i]
        
        if word in word_points:
            points = word_points[word]
            
            # Check for the "not" flip right before the scored word
            if i > 0 and words[i - 1] == "not":
                points = -points
                
            total_score += points

    # Decide the mood based on the final score
    if total_score > 0:
        return "HAPPY"
    elif total_score < 0:
        return "UNHAPPY"
    else:
        return "OK"

# 3. Run the program and count mistakes
mistakes = 0

print("--- Canteen Comment Analysis ---")
for comment, actual_mood in comments_data:
    predicted_mood = analyze_mood(comment)
    is_mistake = predicted_mood != actual_mood
    
    if is_mistake:
        mistakes += 1
        
    status = "MISTAKE" if is_mistake else "CORRECT"
    print(f"Comment: '{comment}'")
    print(f"  Program: {predicted_mood} | Actual: {actual_mood} -> [{status}]\n")

print(f"Total Mistakes Made: {mistakes} out of {len(comments_data)}")
