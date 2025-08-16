from sentence_transformers import SentenceTransformer, util

class ARKResponses:
    """Gère les réponses prédéfinies et la détection des commandes de mise en veille d'ARK."""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
        
        # Réponses prédéfinies pour les conversations basiques
        self.responses = {
            "dire bonjour ou demander comment ça va": "Salut ! Ça va bien, merci ! Et toi ?",
            "remercier poliment": "Avec plaisir !",
            "demander le nom de l'assistant": "Je suis ton petit assistant vocal ARK, j'existe pour t'aider dans tes tâches du quotidien."
        }
        
        # Phrase déclencheur pour la mise en veille
        self.sleep_trigger = "dire au revoir ou demander à ARK de se mettre en veille ou bien de se reposer ou de sleep"
        self.unknown_response = "Désolé, je n'ai pas compris. Peux-tu reformuler ?"
        
        # Pré-calculer les embeddings pour optimiser les performances
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Pré-calcule les embeddings pour éviter de les recalculer à chaque requête."""
        self.response_texts = list(self.responses.keys())
        self.response_embeddings = self.model.encode(self.response_texts, convert_to_numpy=True)
        self.sleep_embedding = self.model.encode([self.sleep_trigger], convert_to_numpy=True)
    
    def get_best_response(self, user_text: str, threshold: float = 0.6) -> str:
        """
        Trouve la réponse prédéfinie la plus proche sémantiquement du texte utilisateur.
        Retourne une réponse "inconnue" si la similarité est trop faible.
        
        Args:
            user_text (str): Le texte de l'utilisateur à analyser
            threshold (float): Seuil minimum de similarité (0.0 à 1.0)
            
        Returns:
            str: La réponse la plus appropriée ou unknown_response
        """
        user_embedding = self.model.encode([user_text], convert_to_numpy=True)
        scores = util.cos_sim(user_embedding, self.response_embeddings)[0]
        best_idx = scores.argmax()
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            return self.responses[self.response_texts[best_idx]]
        else:
            return self.unknown_response
    
    def is_sleep_command(self, user_text: str, threshold: float = 0.6) -> bool:
        """
        Vérifie si la phrase utilisateur demande la mise en veille d'ARK.
        
        Args:
            user_text (str): Le texte de l'utilisateur à analyser
            threshold (float): Seuil de similarité (0.0 à 1.0)
            
        Returns:
            bool: True si c'est une commande de mise en veille, False sinon
        """
        user_embedding = self.model.encode([user_text], convert_to_numpy=True)
        score = util.cos_sim(user_embedding, self.sleep_embedding)[0][0]
        return float(score) >= threshold
    
    def add_response(self, trigger: str, response: str):
        """
        Ajoute une nouvelle réponse prédéfinie.
        
        Args:
            trigger (str): La phrase déclencheur
            response (str): La réponse à donner
        """
        self.responses[trigger] = response
        # Recalculer les embeddings
        self._precompute_embeddings()
    
    def remove_response(self, trigger: str) -> bool:
        """
        Supprime une réponse prédéfinie.
        
        Args:
            trigger (str): La phrase déclencheur à supprimer
            
        Returns:
            bool: True si supprimé avec succès, False si non trouvé
        """
        if trigger in self.responses:
            del self.responses[trigger]
            self._precompute_embeddings()
            return True
        return False
    
    def get_all_responses(self) -> dict:
        """
        Retourne toutes les réponses prédéfinies.
        
        Returns:
            dict: Dictionnaire des réponses {trigger: response}
        """
        return self.responses.copy()
    
    def set_sleep_trigger(self, new_trigger: str):
        """
        Modifie la phrase déclencheur pour la mise en veille.
        
        Args:
            new_trigger (str): Nouvelle phrase déclencheur
        """
        self.sleep_trigger = new_trigger
        self.sleep_embedding = self.model.encode([self.sleep_trigger], convert_to_numpy=True)