import os
import time
import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
from ark_commands.ark_commands import ARKCommands

# --- Phrases prédéfinies ---
responses = {
    "dire bonjour ou demander comment ça va": "Salut ! Ça va bien, merci ! Et toi ?",
    "remercier poliment": "Avec plaisir !",
    "demander le nom de l'assistant": "Je suis ton petit assistant vocal ARK, j'existe pour t'aider dans tes tâches du quotidien."
}

# --- Phrases pour mise en veille ---
sleep_trigger = "dire au revoir ou demander à ARK de se mettre en veille"

# --- Charger le modèle ---
model = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    device='cpu'
)

# --- Pré-calculer les embeddings en numpy pour éviter le warning ---
response_texts = list(responses.keys())
response_embeddings = model.encode(response_texts, convert_to_numpy=True)  # shape = (n_responses, dim)
sleep_embedding = model.encode([sleep_trigger], convert_to_numpy=True)     # shape = (1, dim)

def get_best_response(user_text):
    """Trouve la réponse prédéfinie la plus proche."""
    user_embedding = model.encode([user_text], convert_to_numpy=True)  # shape = (1, dim)
    scores = util.cos_sim(user_embedding, response_embeddings)[0]       # shape = (n_responses,)
    best_idx = scores.argmax()
    return responses[response_texts[best_idx]]

def is_sleep_command(user_text, threshold=0.6):
    """Vérifie si la phrase demande la mise en veille."""
    user_embedding = model.encode([user_text], convert_to_numpy=True)  # shape = (1, dim)
    score = util.cos_sim(user_embedding, sleep_embedding)[0][0]         # scalar
    return score >= threshold

# --- Initialiser la reconnaissance vocale ---
mic_index = 1
r = sr.Recognizer()

def listen(timeout=None, phrase_limit=5):
    """Écoute une phrase avec un timeout optionnel."""
    with sr.Microphone(device_index=mic_index) as source:
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            return r.recognize_google(audio, language="fr-FR").lower()  # type: ignore
        except sr.WaitTimeoutError:
            return ""
        except:
            return ""

# --- Initialiser commandes ---
commands = ARKCommands(model=model)

# --- Boucle principale ---
active = False
last_active_time = 0
chat_duration = 15  # secondes après activation

print("ARK prêt ! Dites 'activation' pour l'activer.")

while True:
    if not active:
        phrase = listen(timeout=None, phrase_limit=5)
        if not phrase:
            continue
        if "activation" in phrase:
            active = True
            last_active_time = time.time()
            print("ARK activé ! Je t'écoute...")
        elif "stop" in phrase:
            print("ARK désactivé. À bientôt !")
            break
    else:
        if time.time() - last_active_time > chat_duration:
            print("ARK se remet en veille...")
            active = False
            continue

        phrase = listen(timeout=3, phrase_limit=5)
        if not phrase:
            continue

        print("Tu as dit :", phrase)

        # --- Vérif mise en veille ---
        if is_sleep_command(phrase):
            print("Bot : À bientôt !")
            active = False
            continue

        # --- Détection des commandes ---
        result = commands.get_best_command(phrase)
        if result:
            print("ARK :", result)
        else:
            response = get_best_response(phrase)
            print("Bot :", response)

        last_active_time = time.time()
