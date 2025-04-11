#test_regex.py

import re

# Load regex patterns from your file
def load_bad_word_patterns(file_path="C:/Users/mukth/OneDrive/Desktop/YouTubeSentimentProject/data/Final/bad_words_regex.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return [re.compile(p.strip(), re.IGNORECASE) for p in f if p.strip()]

# Test sentences
test_sentences = [
    "You're a dumb idiot!",
    "Get lost you d@mn troll.",
    "He's such an a$$hole...",
    "U r f4king kidding me!",
    "Screw off you 1d1ot!",
    "Absolute nonsense, nothing wrong here!",
    "Go to h3ll with your fake news!",
    "You m0r0n! Learn something.",
    "FuCk y0u bro"
]

# Load patterns and test them
patterns = load_bad_word_patterns()
print("🔍 Detected Offensive Comments:\n")

for sentence in test_sentences:
    flagged = any(p.search(sentence) for p in patterns)
    status = "🚫 Negative" if flagged else "✅ Positive"
    print(f"{status:7} → {sentence}")
