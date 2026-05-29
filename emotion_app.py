import random
from pathlib import Path

import whisper
from transformers import pipeline

# Base directory
BASE_DIR = Path.cwd()

# Whisper local model path
WHISPER_MODEL_PATH = BASE_DIR / "whisper_models" / "base.pt"
if not WHISPER_MODEL_PATH.exists():
    raise FileNotFoundError(f"Whisper model not found: {WHISPER_MODEL_PATH}")

# Force CPU for Whisper
whisper_model = whisper.load_model(str(WHISPER_MODEL_PATH)).to("cpu")

# Emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=-1
)

EMOJI_MAP = {
    "joy": "😁",
    "neutral": "😐",
    "sadness": "😭",
    "anger": "🤬",
    "disgust": "🤢",
    "fear": "😨",
    "surprise": "🫨",
}

FEEDBACK = {
    "joy": [
        "I can feel your happiness through the screen 🤩",
        "This sounds like a really lovely moment 🌈",
        "Hold onto this feeling - it’s precious 🥰",
        "Your joy is quietly glowing 💛",
        "Moments like this deserve to be celebrated 🥳",
        "Let yourself enjoy the happiness - you deserve it 💋",
        "This kind of happiness suits you perfectly 🌟",
        "It’s nice to see something going right for you 🤍",
    ],
    "neutral": [
        "I’m here with you - thanks for sharing 🤞🏻",
        "It’s okay to just feel... okay sometimes 🌿",
        "Not every moment needs a big emotion - this is valid too 💭",
        "A calm state can be a safe place to rest 🧸",
        "Steady days matter more than we realize 🌙",
        "Being neutral doesn’t mean being empty 🫶🏻",
        "You’re allowed to just exist without pressure 🫵🏻",
        "Quiet moments can still be meaningful 🌻",
    ],
    "sadness": [
        "I’m really sorry you’re feeling this way 💙",
        "It’s okay to feel sad — you don’t have to rush through it 😔",
        "You’re not weak for feeling this, you’re human 🌧️",
        "Take your time. I’m here with you 👀",
        "Even heavy feelings pass, slowly but surely 🌱",
        "You don’t have to carry this alone ☂️",
        "It’s okay to rest when things feel heavy 🛌",
        "Your feelings deserve kindness too 💫",
    ],
    "anger": [
        "It sounds like something really crossed a line 😠",
        "Your feelings make sense - let’s slow things down together 🌬️",
        "Take a breath. You deserve a moment of calm 🧘",
        "Strong emotions often mean something important matters 🔥",
        "Pause for a second — clarity comes after the storm 🌊",
        "You’re allowed to be angry without letting it control you 🍀",
        "It’s okay to protect your boundaries 🐚",
        "Let’s give your mind a little space right now 🧠",
    ],
    "disgust": [
        "That reaction is completely understandable 🤝",
        "Some things just don’t sit right, and that’s okay 🤐",
        "It’s fine to step back from things that feel wrong 🚧",
        "Trust your boundaries — they’re there to protect you 🛡️",
        "Your discomfort is a signal worth listening to 👂🏻",
        "You’re allowed to say ‘no’ to what feels unpleasant ❌",
        "You don’t need to tolerate what makes you uncomfortable 🤕",
        "Taking distance can be a form of self-care 🚶‍♀️",
    ],
    "fear": [
        "It’s okay to feel scared - fear doesn’t mean failure 🧚‍♀️",
        "You’re facing something uncertain, and that takes courage ✊",
        "Take a slow breath - you’re safe in this moment ❣️",
        "Fear often shows up before growth 🧩",
        "You don’t have to be fearless to be brave 🩰",
        "One step at a time - you’re doing your best 👣",
        "It’s okay to move slowly when things feel overwhelming 🐢",
        "You’re stronger than this moment feels 🤘",
    ],
    "surprise": [
        "Wow, that was unexpected 🙀",
        "Surprises can shake us, in good or strange ways 🤓",
        "Give yourself a second to take it in 🫥",
        "Life loves plot twists sometimes 🎢",
        "It’s okay if you don’t know how to feel yet 🤗",
        "Moments like this often stay memorable 🤭",
        "Take a breath - you can process this at your own pace 💖",
        "Unexpected doesn’t mean bad, just different 🌈",
    ],
}

def predict_emotion(text: str):
    """
    Input: text (str)
    Output: emotion, confidence, scores_dict, emoji, feedback
    """
    outputs = emotion_classifier(text)

    scores_list = outputs[0] if isinstance(outputs, list) and outputs and isinstance(outputs[0], list) else outputs
    scores = {o["label"]: float(o["score"]) for o in scores_list}

    emotion = max(scores, key=scores.get)
    confidence = scores[emotion]
    emoji = EMOJI_MAP.get(emotion, "🙂")
    feedback = random.choice(FEEDBACK.get(emotion, ["Thanks for sharing 💞"]))

    return emotion, confidence, scores, emoji, feedback

def transcribe_audio(audio_file: str):
    """
    Input: audio file path
    Output: transcription text
    """
    result = whisper_model.transcribe(audio_file)
    return result.get("text", "")
