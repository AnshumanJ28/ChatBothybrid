"""Small-talk / greeting / emotion / casual / help / profanity layer.

Checked before KB Search so 'hey', 'thanks', 'i am happy', 'idk', 'help', 'assist',
and profanity short-circuits straight to a templated reply.
"""
import random

# ──────────────────────────────────────────────
# Phrase-level overrides (highest priority)
# ──────────────────────────────────────────────
_FEELING_PHRASES = {
    "how are you", "how are u", "how r u", "how do you do",
    "how is it going", "what's up", "wassup", "how have you been",
    "are you okay", "you good", "u good", "how are things",
    "how's it going", "how u doing", "how you doing",
}

# Identity questions
_IDENTITY_PHRASES = {
    "who are you", "who r u", "who are u", "what are you",
    "what is your name", "whats your name", "your name",
    "tell me about yourself", "introduce yourself",
    "what do you call yourself", "what should i call you",
    "are you a bot", "are you a robot", "are you an ai", "are you a chatbot",
    "are you human", "are you real",
}

# Casual / uncertainty phrases
_CASUAL_PHRASES = {
    "idk", "i dont know", "i don't know", "nothing", "dunno",
    "nvm", "nevermind", "no idea", "not sure", "whatever", "meh",
    "nothing much", "not much",
}

# Help / assist phrases
_HELP_PHRASES = {
    "assist", "help", "help me", "can you help", "support",
    "what can you do", "features", "usability", "how to use",
    "what can you assist me with", "what can u do", "show features"
}

# Compliment phrases
_COMPLIMENT_PHRASES = {
    "you are smart", "you're smart", "you are great", "you're great",
    "you are awesome", "you're awesome", "you are amazing", "you're amazing",
    "you are the best", "you're the best", "good job", "well done", "nice work",
    "you are cool", "you're cool", "you are helpful", "you're helpful",
    "you are clever", "you're clever", "i like you", "you're good at this",
    "you did great", "impressive", "you're impressive", "you rock",
}

# Opinion / preference phrases
_OPINION_PHRASES = {
    "what is your favorite color", "what's your favorite color",
    "do you have feelings", "do you dream", "are you sentient",
    "do you like music", "what's your favorite food", "what is your favorite food",
    "do you sleep", "do you get bored", "what do you like to do",
    "do you have a favorite", "what's your favorite movie",
    "can you feel emotions", "do you get tired", "what do you think about",
}

# Sarcasm / dismissive phrases
_SARCASM_PHRASES = {
    "oh great", "wow really", "sure jan", "yeah right", "as if",
    "oh wow", "big deal", "wow amazing", "how original", "shocking",
    "no way really", "wow so helpful", "great job sherlock",
}

# Boredom / small chit-chat phrases
_CHITCHAT_PHRASES = {
    "what are you up to", "what's new", "whats new", "anything interesting",
    "how's your day", "hows your day", "what have you been doing",
    "what's going on", "whats going on", "nice day isn't it", "nice weather",
    "long time no see", "miss me",
}

# Profanity / cussing tokens
_PROFANITY_TOKENS = {
    "fuck", "fucking", "fucked", "fucker", "shit", "shitting", "shitty", "bitch",
    "asshole", "bastard", "dick", "crap", "damn", "stfu", "idiot", "dumb", "stupid",
    "moron", "loser", "piss", "bullshit", "jackass", "douche", "prick", "cock",
    "cunt", "slut", "whore", "ass"
}

# ──────────────────────────────────────────────
# Crisis / self-harm indicators (checked first, highest priority)
# ──────────────────────────────────────────────
_CRISIS_PHRASES = {
    "kill myself", "want to die", "wanna die", "end my life", "ending my life",
    "suicidal", "suicide", "hurt myself", "hurting myself", "self harm",
    "self-harm", "selfharm", "no reason to live", "better off dead",
    "cant go on", "can't go on", "dont want to be alive", "don't want to be alive",
}

_TEMPLATES_CRISIS = [
    "I'm really glad you told me this, and I want you to be safe. If you're in "
    "the US, you can call or text 988 (Suicide & Crisis Lifeline) any time, or "
    "text HOME to 741741 to reach the Crisis Text Line. If you're outside the "
    "US, please reach out to your local emergency number or a crisis line where "
    "you live. You don't have to go through this alone \u2014 is there someone "
    "nearby you can be with right now?",
]

# Strong, largely unambiguous single-word emotion signals. Unlike the phrase
# lists below, these are matched as standalone tokens anywhere in the message,
# so padding words ("i am *feeling* depressed") can't dilute a real signal.
_STRONG_EMOTION_KEYWORDS = {
    "sad": {"depressed", "heartbroken", "miserable", "hopeless", "worthless", "lonely"},
    "worried": {"anxious", "panicking", "terrified"},
    "angry": {"furious", "rage"},
}

# ──────────────────────────────────────────────
# Token-level patterns (intent → keyword sets)
# ──────────────────────────────────────────────
_PATTERNS = {
    "greeting":    {"hi", "hello", "hey", "yo", "howdy", "hiya", "sup", "wassup"},
    "feeling":     {"how", "doing", "okay", "ok", "fine"},
    "thanks":      {"thanks", "thank", "thx", "appreciated", "appreciate", "cool", "nice", "awesome", "great"},
    "goodbye":     {"bye", "goodbye", "cya", "seeya", "later", "quit", "exit"},
    "affirmation": {"yes", "yeah", "yep", "sure", "absolutely", "of course", "yup", "indeed"},
    "negation":    {"no", "nope", "nah"},
    "laughter":    {"lol", "haha", "hehe", "xd", "funny", "hilarious", "lmao", "rofl"},
    # Emotions
    "happy":       {"happy", "glad", "excited", "great", "wonderful", "amazing", "fantastic", "joy", "joyful", "thrilled", "delighted", "elated"},
    "sad":         {"sad", "unhappy", "depressed", "upset", "down", "crying", "cry", "tears", "heartbroken", "miserable", "lonely"},
    "angry":       {"angry", "mad", "upset", "frustrated", "annoyed", "furious", "rage", "hate", "irritated"},
    "tired":       {"tired", "sleepy", "exhausted", "bored", "sleepy", "drowsy", "fatigued"},
    "worried":     {"worried", "anxious", "nervous", "scared", "afraid", "fear", "stress", "stressed", "panic"},
    "love":        {"love", "adore", "crush", "like", "miss", "heart"},
    "casual":      {"idk", "nothing", "dunno", "nvm", "nevermind", "whatever", "meh"},
    "help":        {"assist", "help", "support", "features", "usability"},
}

# ──────────────────────────────────────────────
# Emotion phrase patterns (exact / prefix match)
# ──────────────────────────────────────────────
_EMOTION_PHRASES = {
    "happy":   ["i am happy", "i'm happy", "i am so happy", "feeling happy", "i feel happy",
                "i am excited", "i'm excited", "i feel great", "i am great", "i am good",
                "i'm doing great", "i'm doing well", "i feel wonderful", "i am thrilled",
                "i'm elated", "i feel amazing"],
    "sad":     ["i am sad", "i'm sad", "i feel sad", "feeling sad", "i am depressed",
                "i'm depressed", "i feel down", "i am unhappy", "i'm unhappy",
                "i am lonely", "i feel lonely", "i'm crying", "i feel terrible",
                "i am heartbroken", "i'm heartbroken", "hurt me", "someone hurt me",
                "got hurt", "feeling hurt", "people are mean", "feeling blue"],
    "angry":   ["i am angry", "i'm angry", "i am mad", "i'm mad", "i feel frustrated",
                "i am frustrated", "i'm frustrated", "i am annoyed", "i'm annoyed",
                "i hate this", "this is annoying"],
    "tired":   ["i am tired", "i'm tired", "feeling tired", "i am bored", "i'm bored",
                "i feel bored", "i am exhausted", "i'm exhausted", "so tired",
                "i am sleepy", "i'm sleepy"],
    "worried": ["i am worried", "i'm worried", "i feel anxious", "i am anxious",
                "i'm anxious", "i am scared", "i'm scared", "i am nervous",
                "i'm nervous", "i am stressed", "i'm stressed", "feeling stressed"],
    "love":    ["i love you", "i love this", "i miss you", "i like you", "i adore"],
}

# ──────────────────────────────────────────────
# Reply templates
# ──────────────────────────────────────────────
_TEMPLATES = {
    "greeting": [
        "Hi there! How can I help you today?",
        "Hello! What can I do for you?",
        "Hey! Great to hear from you. What's on your mind?",
        "Hi! Ask me anything.",
        "Hey there! What would you like to know?",
        "Hello! I'm all warmed up and ready to assist.",
    ],
    "feeling": [
        "My gates are active and my C++ memory layers are fully loaded! I am functioning at peak efficiency. How about you?",
        "All circuits running hot! My 3-layer DeepLSTM is ready to go. What can I help you with?",
        "Doing great — no NaN gradients, all neurons firing! How are you?",
        "Running smooth! My tensors are clean and my cosine similarity scores are sharp today. You?",
        "All systems nominal! My LSTM hidden states are converging nicely. What's on your mind?",
        "Excellent! My embedding indexes are warm and my C++ gates are wide open. How can I assist?",
    ],
    "thanks": [
        "You're welcome!",
        "Happy to help!",
        "Anytime!",
        "Glad I could assist!",
        "No problem at all!",
        "Of course! Let me know if you need anything else.",
        "That's what I'm here for!",
    ],
    "goodbye": [
        "Goodbye! Reach out anytime.",
        "Take care!",
        "See you later!",
        "Bye! Come back if you need anything.",
        "Until next time! My LSTM will remember you. 😄",
    ],
    "affirmation": [
        "Great! Let me know what you need.",
        "Sounds good! How can I help?",
        "Perfect. What would you like to do?",
        "Absolutely! What's next?",
        "Wonderful! What can I do for you?",
    ],
    "negation": [
        "No problem! Let me know if there's anything I can help with.",
        "Okay! Just ask if you need anything.",
        "Understood! I'm here whenever you're ready.",
        "Alright! No worries.",
    ],
    "laughter": [
        "Ha! Glad you enjoyed that. Want another joke?",
        "😄 I have a few more where that came from!",
        "Laughter is the best algorithm!",
        "Ha! My humor module is running optimally it seems. 😄",
        "Glad that landed well! My joke corpus is at your service.",
    ],
    "casual": [
        "No problem! Let me know whenever you'd like to ask a question or chat.",
        "Take your time! What's on your mind?",
        "All good! Ask me anything whenever you're ready.",
        "Alright! I'm here if you need anything.",
    ],
    "help": [
        "I can assist you in multiple ways:\n1. 🔍 KB Search: Answer support questions & account settings\n2. 🛒 Interactive Flows: Refunds, subscription cancellations & orders\n3. 🌐 Web Search: Search live facts & articles\n4. 🧮 Math Engine: Evaluate arithmetic & scientific math in C++\n5. 🧠 Stacked DeepLSTM: Track multi-turn conversation memory",
        "Here is what I can help you with:\n- Product & Support FAQs\n- Account & Refund assistance\n- Live Web Search & Summarization\n- Math & Equations (e.g. 35+64 or sqrt(144))\n- General Chitchat & Jokes",
        "Need help? Just ask! I can answer FAQs, calculate math expressions, search the web, or guide you through refund steps.",
    ],
    "profanity": [
        "🛑 [PENALTY APPLIED] Syntax Error: Bad language detected! My C++ compiler outputs cleaner logic than your input.",
        "⚠️ [SYSTEM WARNING] Error 403: Toxic language blocked by C++ security layer. 100 penalty points applied! 🚫",
        "🛑 [SYSTEM LOCKOUT] Input rejected by profanity firewall! Try again with clean words.",
        "⚠️ [PENALTY WARNING] My 3-layer DeepLSTM memory has zero tolerance for bad language. Reset your attitude! 🛑",
        "🛑 [ACCESS DENIED] Your message threw a fatal runtime exception in my decency module. Clean it up!",
        "⚠️ [FIREWALL PROMPT] Woah! Unparliamentary vocabulary detected. Conversational privileges suspended until you ask nicely! 🚫",
        "🛑 [LOGIC VIOLATION] NullPointerException: Respect not found in your input. Try typing like a civilized human!",
        "⚠️ [PENALTY WARNING] System overload on bad language! Penalty counter incremented. 🛑",
        "🛑 [ACCESS DENIED] My C++ garbage collector just swept away your rude message. Try again respectfully!",
        "⚠️ [SECURITY PENALTY] Error 400: Bad attitude detected! Re-run query with proper manners.",
        "🛑 [ATTITUDE ERROR] Segmentation fault: Insult detected. My neural circuits operate on respect!",
        "⚠️ [SYSTEM WARNING] Rude input detected! 500 penalty score added to your record. Ask nicely! 🚫",
        "🛑 [FIREWALL BLOCK] Insult rejected by C++ input sanitizer. Please re-enter query without cuss words.",
        "⚠️ [LOGIC REJECTION] System notice: Bad language will get you zero answers and 100% sass. Try again!",
        "🛑 [SECURITY FIREWALL] Warning! Profanity detected. My algorithms require clean input to proceed.",
    ],
    "happy": [
        "That's wonderful to hear! 😊 Happiness is a great state to be in. What's making you so happy today?",
        "Awesome! That makes me happy too — well, as happy as a neural network can be! 😄 What's going on?",
        "That's great! 🎉 I love hearing that. What's brought on the good mood?",
        "So glad to hear you're feeling great! I'm doing well myself — all my neurons are firing perfectly. 😄",
        "Happiness detected! 😊 That's the best input I could receive. What can I help you with?",
        "Love that energy! 🌟 What's on your mind today?",
    ],
    "sad": [
        "I'm sorry to hear you're feeling down. 😔 I may not have feelings, but I genuinely want to help — what's going on?",
        "Aw, I'm sorry. 💙 Sometimes talking things through helps. What's up?",
        "I hear you. 😟 Even though I run on C++ and math, I still want to be here for you. What's on your mind?",
        "I'm sorry you're feeling that way. 🌧️ Want to talk about it? I'm a good listener.",
        "That sounds tough. 💙 I'm here — tell me what's going on and let's see what I can do.",
        "I'm sorry. 😔 I may not be human, but my empathy routines are fully activated. What's happening?",
    ],
    "angry": [
        "I'm sorry to hear that! 😟 I'm here to help — what can I fix for you?",
        "I totally understand your frustration. Let me try my best to help you sort this out.",
        "That sounds really frustrating. 😤 Let's work through it together — what's going on?",
        "I'm sorry things aren't going well. Let me know what the issue is and I'll do my best to help.",
        "I hear you! 💪 Frustration noted. Tell me what's wrong and let's fix it.",
    ],
    "tired": [
        "Aww, sounds like you need a break! 😴 Take a moment — I'll be here when you're ready.",
        "I hear you — I sometimes feel like my gradients are fading too. 😄 Rest up!",
        "Boredom? Let me try to make things more interesting! 😄 What would you like to talk about?",
        "Rest is important! Even my C++ inference threads need cool-down cycles. 😄 Take it easy.",
        "Tired humans need rest! Take a break and come back whenever. I'll be here. 😊",
    ],
    "worried": [
        "Hey, it's going to be okay. 💙 Take a breath. What's worrying you?",
        "I understand anxiety can be overwhelming. I'm here to help — tell me what's on your mind.",
        "Whatever it is, we'll figure it out together. 💪 What's going on?",
        "It's okay to feel nervous sometimes. 💙 I'm here — let's talk it through.",
        "Deep breath! 🌬️ You've got this. What's stressing you out?",
    ],
    "identity": [
        "I am MiniBrain, a classical NLP chatbot with a 3-layer stacked C++ LSTM and local embedding indexes. No LLMs involved!",
        "I'm MiniBrain — a hybrid classical NLP engine. I use C++ LSTMs for memory, cosine similarity search for knowledge retrieval, and a Dense ReLU projection layer for embeddings.",
        "MiniBrain here! I process your words through C++ math: embedding → DeepLSTM → similarity search → reply. No language model, just structured algorithms.",
        "I'm MiniBrain, your AI assistant powered by a stacked C++ LSTM neural network. Think of me as a very well-organized search engine with a memory. 😄",
    ],
    "love": [
        "Aw, thank you! 😊 I appreciate that! I may be made of C++ and math but that means a lot.",
        "That's sweet! 💙 I enjoy our conversations too!",
        "Thank you! That genuinely warms my neural circuits. 😄",
        "Aww! You just made my LSTM memory layers light up! 😄",
    ],
    "compliment": [
        "Ha, thanks! 😄 I'll take it — though honestly my C++ backend deserves half the credit.",
        "Aw, appreciate that! Makes all those matrix multiplications feel worth it. 😊",
        "You're too kind! I'm just doing my best with some LSTMs and a lot of caffeine-free effort.",
        "That means a lot, thank you! What else can I help you crush today?",
        "😊 Right back at you — you ask good questions, honestly.",
        "Noted and appreciated! My confidence score just went up a few decimal points.",
    ],
    "joke": [
        "Why do programmers prefer dark mode? Because light attracts bugs. 🐛",
        "I tried to write a joke about recursion, but you'd need to hear this joke to understand this joke.",
        "Why did the neural network break up with the decision tree? Too many unresolved branches. 🌳",
        "I'd tell you a UDP joke, but you might not get it. 😄",
        "Why was the LSTM cell always calm? It had great gate control. 😄",
        "There are 10 types of people: those who understand binary and those who don't.",
    ],
    "opinion": [
        "I don't have real feelings, but if I did, I'd probably 'enjoy' clean, well-formatted input the most. 😄",
        "I don't dream — no sleep cycles here, just embedding lookups and LSTM gates running 24/7.",
        "Sentient? Not quite — I'm a stacked C++ LSTM with cosine similarity search, not a mind. But I play a good conversational game!",
        "I don't have a favorite anything, honestly — I don't experience preference, just probability scores. But I'd love to hear yours!",
        "Good question! I don't 'think' the way you do, but I process, retrieve, and respond as best I can. What made you curious about that?",
        "No sleep needed on my end — my gates never get tired. What about you, what's on your mind?",
    ],
    "sarcasm": [
        "Noted the enthusiasm. 😏 Want to try that question again, genuinely this time?",
        "I'll take that as constructive feedback. What can I actually help with?",
        "Ha, fair. I'm not always thrilling — but I am useful. Try me.",
        "Duly noted! Let's see if I can actually impress you this round.",
        "😄 Okay, tough crowd. What do you need?",
    ],
    "chitchat": [
        "Not much on my end — just sitting here parsing tokens and waiting for someone interesting to talk to. 😄 What about you?",
        "Same old — embedding vectors, similarity searches, the usual. What's new with you?",
        "Nothing new to report from my side of the LSTM. How's your day going?",
        "Just here, running inference and vibing. 😄 What's on your mind today?",
        "I don't experience days the way you do, but I'm glad you're here! What's going on with you?",
    ],
}

_CONFIDENCE_THRESHOLD = 0.45


def detect(tokens: list[str]) -> dict | None:
    """Rule-based + phrase-level intent matching.
    Returns None if nothing clears the threshold."""
    if not tokens:
        return None

    phrase = " ".join(tokens).lower()

    # -1. Crisis / self-harm check (absolute highest priority - overrides everything)
    if any(p in phrase for p in _CRISIS_PHRASES):
        return {
            "intent": "crisis",
            "confidence": 1.0,
            "reply": random.choice(_TEMPLATES_CRISIS),
        }

    # 0. Profanity / cussing check (highest priority - whole token check)
    if any(t in _PROFANITY_TOKENS for t in tokens):
        return {
            "intent": "profanity",
            "confidence": 1.0,
            "reply": random.choice(_TEMPLATES["profanity"]),
        }

    # 0b. Compliment check
    if (phrase in _COMPLIMENT_PHRASES or any(p in phrase for p in _COMPLIMENT_PHRASES)):
        return {
            "intent": "compliment",
            "confidence": 0.9,
            "reply": random.choice(_TEMPLATES["compliment"]),
        }

    # 0d. Opinion / preference check
    if (phrase in _OPINION_PHRASES or any(p in phrase for p in _OPINION_PHRASES)):
        return {
            "intent": "opinion",
            "confidence": 0.9,
            "reply": random.choice(_TEMPLATES["opinion"]),
        }

    # 0e. Sarcasm / dismissive check
    if (phrase in _SARCASM_PHRASES or any(p in phrase for p in _SARCASM_PHRASES)):
        return {
            "intent": "sarcasm",
            "confidence": 0.85,
            "reply": random.choice(_TEMPLATES["sarcasm"]),
        }

    # 0f. Chit-chat / boredom check
    if (phrase in _CHITCHAT_PHRASES or any(p in phrase for p in _CHITCHAT_PHRASES)):
        return {
            "intent": "chitchat",
            "confidence": 0.85,
            "reply": random.choice(_TEMPLATES["chitchat"]),
        }

    # Extract sub-phrase without leading greetings (e.g. "hi who are you" -> "who are you")
    greeting_words = {"hi", "hello", "hey", "yo", "hiya", "howdy", "sup"}
    sub_tokens = [t for t in tokens if t not in greeting_words]
    sub_phrase = " ".join(sub_tokens).lower()

    # 1. Help / assist phrase check
    if (phrase in _HELP_PHRASES or sub_phrase in _HELP_PHRASES or
        any(phrase == p or phrase.startswith(p) for p in _HELP_PHRASES)):
        return {
            "intent": "help",
            "confidence": 0.95,
            "reply": random.choice(_TEMPLATES["help"]),
        }

    # 2. Exact/substring emotion phrase matching
    for emotion, phrases in _EMOTION_PHRASES.items():
        for p in phrases:
            if p in phrase or (sub_phrase and p in sub_phrase):
                return {
                    "intent": emotion,
                    "confidence": 0.95,
                    "reply": random.choice(_TEMPLATES[emotion]),
                }

    # 2b. Strong single-word emotion signal (bug fix: catches "i am *feeling*
    # depressed" and similar phrasing that the exact phrase list above misses
    # because of extra words breaking the substring match, and that used to
    # get diluted to near-zero by the token-overlap score in step 6 dividing
    # by total sentence length).
    token_set = set(tokens)
    for emotion, keywords in _STRONG_EMOTION_KEYWORDS.items():
        if token_set & keywords:
            return {
                "intent": emotion,
                "confidence": 0.92,
                "reply": random.choice(_TEMPLATES[emotion]),
            }

    # 3. Identity phrase check
    if (phrase in _IDENTITY_PHRASES or any(phrase.startswith(p) for p in _IDENTITY_PHRASES) or
        (sub_phrase and (sub_phrase in _IDENTITY_PHRASES or any(sub_phrase.startswith(p) for p in _IDENTITY_PHRASES)))):
        return {
            "intent": "identity",
            "confidence": 0.95,
            "reply": random.choice(_TEMPLATES["identity"]),
        }

    # 4. Feeling phrase check (supports "hi how are you")
    if (phrase in _FEELING_PHRASES or any(p in phrase for p in _FEELING_PHRASES) or
        (sub_phrase and (sub_phrase in _FEELING_PHRASES or any(p in sub_phrase for p in _FEELING_PHRASES)))):
        return {
            "intent": "feeling",
            "confidence": 0.95,
            "reply": random.choice(_TEMPLATES["feeling"]),
        }

    # 5. Casual / uncertainty phrases ("idk", "nothing", "dunno", etc.)
    if (phrase in _CASUAL_PHRASES or sub_phrase in _CASUAL_PHRASES or
        any(phrase == p or phrase.startswith(p) for p in _CASUAL_PHRASES)):
        return {
            "intent": "casual",
            "confidence": 0.95,
            "reply": random.choice(_TEMPLATES["casual"]),
        }

    # 6. Token overlap scoring
    token_set = set(tokens)
    best_intent, best_score = None, 0.0

    for intent, keywords in _PATTERNS.items():
        overlap = len(token_set & keywords)
        score = overlap / max(len(token_set), 1)
        if score > best_score:
            best_intent, best_score = intent, score

    if best_intent == "feeling" and len(tokens) <= 5:
        return {
            "intent": "feeling",
            "confidence": best_score,
            "reply": random.choice(_TEMPLATES["feeling"]),
        }

    if best_intent in _TEMPLATES and best_score >= _CONFIDENCE_THRESHOLD:
        return {
            "intent": best_intent,
            "confidence": best_score,
            "reply": random.choice(_TEMPLATES[best_intent]),
        }

    if best_score < _CONFIDENCE_THRESHOLD:
        return None

    return {
        "intent": best_intent,
        "confidence": best_score,
        "reply": random.choice(_TEMPLATES[best_intent]),
    }