import os
import time
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
from ark_commands.ark_commands import ARKCommands
from ark_responses.ark_responses import ARKResponses

def load_model_with_progress():
    """Charge le modèle avec indicateur de progression."""
    print("🤖 Chargement du modèle ARK...")
    start_time = time.time()
    
    # Utiliser un modèle plus léger et rapide
    model = SentenceTransformer(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        device='cpu'
    )
    
    load_time = time.time() - start_time
    print(f"✅ Modèle chargé en {load_time:.2f}s")
    return model

def initialize_components():
    """Initialise tous les composants avec feedback utilisateur."""
    print("🚀 Initialisation d'ARK en cours...")
    
    # Chargement du modèle
    model = load_model_with_progress()
    
    # Initialisation des composants
    print("📝 Chargement des réponses...")
    ark_responses = ARKResponses(model)
    
    print("⚡ Chargement des commandes...")
    commands = ARKCommands(model=model)
    
    print("🎤 Configuration du microphone...")
    # Initialiser la reconnaissance vocale
    r = sr.Recognizer()
    
    return model, ark_responses, commands, r

def listen(r, timeout=None, phrase_limit=5, mic_index=1):
    """Écoute une phrase avec un timeout optionnel."""
    with sr.Microphone(device_index=mic_index) as source:
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            return r.recognize_google(audio, language="fr-FR").lower()
        except sr.WaitTimeoutError:
            return ""
        except Exception as e:
            # print(f"Erreur de reconnaissance: {e}")  # Debug si nécessaire
            return ""

def main():
    """Fonction principale optimisée."""
    # Initialisation avec feedback
    model, ark_responses, commands, r = initialize_components()
    
    # Configuration
    mic_index = 1
    active = False
    last_active_time = 0
    chat_duration = 15  # secondes après activation
    
    print("🎯 ARK prêt ! Dites 'activation' pour l'activer.")
    print("💡 Conseil: Dites 'stop' pour quitter complètement.")
    
    try:
        while True:
            if not active:
                phrase = listen(r, timeout=None, phrase_limit=5, mic_index=mic_index)
                if not phrase:
                    continue
                    
                if "activation" in phrase:
                    active = True
                    last_active_time = time.time()
                    print("✨ ARK activé ! Je t'écoute...")
                elif "stop" in phrase:
                    print("👋 ARK désactivé. À bientôt !")
                    break
            else:
                # Vérification du timeout
                if time.time() - last_active_time > chat_duration:
                    print("😴 ARK se remet en veille...")
                    active = False
                    continue

                phrase = listen(r, timeout=3, phrase_limit=5, mic_index=mic_index)
                if not phrase:
                    continue

                print("\n👤 Tu as dit :", phrase)

                # Vérification mise en veille
                if ark_responses.is_sleep_command(phrase):
                    print("🤖 ARK : À bientôt !")
                    active = False
                    continue

                # Détection des commandes
                result = commands.get_best_command(phrase)
                if result:
                    print("🤖 ARK :", result)
                else:
                    # Utiliser les réponses prédéfinies
                    response = ark_responses.get_best_response(phrase)
                    print("🤖 ARK :", response)

                last_active_time = time.time()
                
    except KeyboardInterrupt:
        print("\n👋 ARK interrompu par l'utilisateur. À bientôt !")
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")
    finally:
        print("🔄 Nettoyage en cours...")

if __name__ == "__main__":
    main()