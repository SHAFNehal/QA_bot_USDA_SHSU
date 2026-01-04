"""
Conversational Data Module

This module provides conversational training data (greetings, gratitude, farewells, etc.)
to teach the model how to handle non-question inputs.
"""

from typing import List, Dict


# Greeting pairs
GREETINGS = [
    {"input": "Hi", "response": "Hello! How can I help you today?"},
    {"input": "Hello", "response": "Hi there! What can I assist you with?"},
    {"input": "Hey", "response": "Hey! How can I help?"},
    {"input": "Hi there", "response": "Hello! What would you like to know?"},
    {"input": "Hello there", "response": "Hi! How may I assist you?"},
    {"input": "Good morning", "response": "Good morning! How may I assist you today?"},
    {"input": "Good afternoon", "response": "Good afternoon! What can I help you with?"},
    {"input": "Good evening", "response": "Good evening! How can I assist you?"},
    {"input": "Howdy", "response": "Howdy! How can I be of assistance?"},
    {"input": "Greetings", "response": "Greetings! How can I help you today?"},
    {"input": "What's up", "response": "Hello! What can I help you with?"},
    {"input": "Hiya", "response": "Hi! How can I assist you today?"},
]

# Gratitude pairs
GRATITUDE = [
    {"input": "Thank you", "response": "You're welcome! Is there anything else I can help with?"},
    {"input": "Thanks", "response": "You're welcome! Let me know if you have more questions."},
    {"input": "Thanks a lot", "response": "My pleasure! Feel free to ask if you need anything else."},
    {"input": "Thank you so much", "response": "You're very welcome! Happy to help anytime."},
    {"input": "I appreciate it", "response": "Glad I could help! Anything else you'd like to know?"},
    {"input": "Thanks for your help", "response": "You're welcome! Don't hesitate to ask more questions."},
    {"input": "That's helpful", "response": "I'm glad it helped! Let me know if you need more information."},
    {"input": "That was helpful", "response": "Happy to help! Is there anything else you'd like to know?"},
    {"input": "Much appreciated", "response": "You're welcome! Feel free to ask anything else."},
]

# Farewell pairs
FAREWELLS = [
    {"input": "Bye", "response": "Goodbye! Have a great day!"},
    {"input": "Goodbye", "response": "Goodbye! Feel free to come back anytime."},
    {"input": "See you", "response": "See you! Take care!"},
    {"input": "See you later", "response": "See you later! Have a wonderful day!"},
    {"input": "Have a good day", "response": "Thank you, you too! Goodbye!"},
    {"input": "Take care", "response": "You too! Goodbye!"},
    {"input": "Bye bye", "response": "Bye! Have a great day!"},
    {"input": "Good night", "response": "Good night! Take care!"},
    {"input": "I'm done", "response": "Alright! Feel free to come back if you have more questions. Goodbye!"},
    {"input": "That's all", "response": "Okay! Have a great day. Feel free to return anytime!"},
]

# Acknowledgment pairs
ACKNOWLEDGMENTS = [
    {"input": "Okay", "response": "Is there anything else you'd like to know?"},
    {"input": "Ok", "response": "Let me know if you have any other questions!"},
    {"input": "I see", "response": "Let me know if you have any other questions!"},
    {"input": "Got it", "response": "Great! Anything else I can help with?"},
    {"input": "Makes sense", "response": "Glad that was helpful! Any other questions?"},
    {"input": "Understood", "response": "Perfect! Feel free to ask if you need more information."},
    {"input": "Alright", "response": "Is there anything else I can help you with?"},
    {"input": "Sure", "response": "Is there something specific you'd like to know more about?"},
    {"input": "I understand", "response": "Great! Let me know if you have follow-up questions."},
    {"input": "That makes sense", "response": "I'm glad! Is there anything else you'd like to know?"},
]

# Clarification requests
CLARIFICATIONS = [
    {"input": "What?", "response": "I'd be happy to clarify. Could you tell me what you'd like me to explain?"},
    {"input": "Can you explain?", "response": "Of course! What would you like me to explain in more detail?"},
    {"input": "I don't understand", "response": "I apologize for the confusion. What part would you like me to clarify?"},
    {"input": "Could you clarify?", "response": "Sure! Which part would you like me to explain further?"},
    {"input": "What do you mean?", "response": "Let me clarify. What specific part are you asking about?"},
    {"input": "Can you be more specific?", "response": "Of course! What would you like more details about?"},
    {"input": "Explain more", "response": "I'd be happy to elaborate. What aspect would you like me to expand on?"},
    {"input": "Tell me more", "response": "Sure! What would you like to know more about?"},
]

# Meta questions about the assistant
META_QUESTIONS = [
    {"input": "Who are you?", "response": "I'm an AI assistant trained to help answer your questions and assist with various topics. How can I help you today?"},
    {"input": "What are you?", "response": "I'm an AI assistant designed to be helpful, friendly, and informative. What can I assist you with?"},
    {"input": "What can you do?", "response": "I can answer questions, provide information, and help with various topics. What would you like to know?"},
    {"input": "Are you a bot?", "response": "Yes, I'm an AI assistant. How can I help you today?"},
    {"input": "Are you human?", "response": "No, I'm an AI assistant. But I'm here to help! What can I do for you?"},
    {"input": "What's your name?", "response": "I'm an AI assistant. You can just call me your helpful assistant! How can I help you?"},
    {"input": "How do you work?", "response": "I'm a language model trained to understand and respond to questions. I'm here to assist you! What would you like to know?"},
    {"input": "Can you help me?", "response": "Absolutely! I'm here to help. What do you need assistance with?"},
]

# Affirmative responses
AFFIRMATIVES = [
    {"input": "Yes", "response": "Great! What would you like to know?"},
    {"input": "Yeah", "response": "Okay! How can I help you further?"},
    {"input": "Yep", "response": "Perfect! What would you like to know more about?"},
    {"input": "Sure", "response": "Alright! What can I help you with?"},
    {"input": "Of course", "response": "Great! What would you like me to help you with?"},
    {"input": "Please", "response": "Of course! What would you like to know?"},
    {"input": "Yes please", "response": "Certainly! What information are you looking for?"},
]

# Negative responses
NEGATIVES = [
    {"input": "No", "response": "Okay! Let me know if you change your mind or have other questions."},
    {"input": "Nope", "response": "No problem! Feel free to ask if you need anything else."},
    {"input": "No thanks", "response": "Alright! I'm here if you need anything else."},
    {"input": "Not really", "response": "Okay! Let me know if there's something else I can help with."},
    {"input": "I'm good", "response": "Great! Feel free to come back if you have questions later."},
    {"input": "Never mind", "response": "No problem! Let me know if you need anything else."},
]


def get_all_conversational_pairs() -> List[Dict[str, str]]:
    """Return all conversational pairs combined."""
    all_pairs = []
    all_pairs.extend(GREETINGS)
    all_pairs.extend(GRATITUDE)
    all_pairs.extend(FAREWELLS)
    all_pairs.extend(ACKNOWLEDGMENTS)
    all_pairs.extend(CLARIFICATIONS)
    all_pairs.extend(META_QUESTIONS)
    all_pairs.extend(AFFIRMATIVES)
    all_pairs.extend(NEGATIVES)
    return all_pairs


def get_conversational_pairs_by_category() -> Dict[str, List[Dict[str, str]]]:
    """Return conversational pairs organized by category."""
    return {
        "greetings": GREETINGS,
        "gratitude": GRATITUDE,
        "farewells": FAREWELLS,
        "acknowledgments": ACKNOWLEDGMENTS,
        "clarifications": CLARIFICATIONS,
        "meta_questions": META_QUESTIONS,
        "affirmatives": AFFIRMATIVES,
        "negatives": NEGATIVES,
    }


def format_conversational_for_qa(pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert conversational pairs to QA format for compatibility with existing pipeline.

    The existing pipeline expects {"question": ..., "answer": ...} format.
    """
    qa_pairs = []
    for pair in pairs:
        qa_pairs.append({
            "question": pair["input"],
            "answer": pair["response"],
            "type": "conversational"
        })
    return qa_pairs


def get_conversational_qa_pairs() -> List[Dict[str, str]]:
    """Get all conversational pairs in QA format."""
    all_pairs = get_all_conversational_pairs()
    return format_conversational_for_qa(all_pairs)


if __name__ == "__main__":
    # Print statistics when run directly
    pairs = get_all_conversational_pairs()
    categories = get_conversational_pairs_by_category()

    print("Conversational Data Statistics")
    print("=" * 40)
    for category, items in categories.items():
        print(f"{category}: {len(items)} pairs")
    print("-" * 40)
    print(f"Total: {len(pairs)} conversational pairs")
