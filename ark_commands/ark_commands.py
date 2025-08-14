import os
from typing import Optional
from sentence_transformers import SentenceTransformer, util

class ARKCommands:
    def __init__(self, model: SentenceTransformer):
        self.model = model

        # Commandes avec phrase clé unique
        self.commands_map = {
            "compte le nombre de fichiers dans le dossier": self._count_files_in_folder,
            "liste les fichiers d’un dossier": self._list_files
        }

        # Préparer embeddings en numpy 2D pour éviter le warning
        self.command_texts = list(self.commands_map.keys())
        self.command_functions = list(self.commands_map.values())
        self.command_embeddings = self.model.encode(
            self.command_texts, 
            convert_to_numpy=True, 
            batch_size=32,    # au lieu de tout faire d’un coup
            show_progress_bar=False
        )

        folder = "C:/Users/auvra/Images"
        print(os.path.exists(folder))      # True si le chemin existe
        print(os.access(folder, os.R_OK)) # True si tu peux lire le dossier


        # Dossier par défaut si aucun chemin trouvé
        self.default_base_path = os.path.expanduser("~")

    def get_best_command(self, phrase: str, threshold: float = 0.6) -> Optional[str]:
        """Détecte la commande et exécute avec le dossier trouvé."""
        folder_path = self.extract_folder_from_phrase(phrase) or self.default_base_path

        # Garder embedding 2D pour cos_sim
        user_emb = self.model.encode([phrase], convert_to_numpy=True)  # shape = [1, dim]

        # Calculer similarité cosinus
        scores = util.cos_sim(user_emb, self.command_embeddings)[0]  # shape = [n_commands]
        best_idx = scores.argmax()

        if scores[best_idx] >= threshold:
            return self.command_functions[best_idx](folder_path)
        return None

    def extract_folder_from_phrase(self, phrase: str, current_path: Optional[str] = None) -> str:
        """Cherche un nom de dossier dans la phrase, dans le dossier courant."""
        current_path = current_path or self.default_base_path

        if "dans" not in phrase:
            return current_path

        folder_name = phrase.split("dans", 1)[1].strip().lower()

        try:
            for entry in os.listdir(current_path):
                entry_path = os.path.join(current_path, entry)
                if os.path.isdir(entry_path) and entry.lower() == folder_name:
                    return entry_path
        except PermissionError:
            pass

        # Si rien trouvé, retourne le dossier courant
        return current_path



    # -------------------------
    # Fonctions des commandes
    # -------------------------
    def _count_files_in_folder(self, folder_path: str):
        try:
            entries = os.listdir(folder_path)
            return f"{len(entries)} éléments trouvés dans {folder_path}."
        except FileNotFoundError:
            return "Dossier introuvable."
        except PermissionError:
            return "Accès refusé."

    def _list_files(self, folder_path: str):
        try:
            files = os.listdir(folder_path)
            return "\n".join(files) if files else "Dossier vide."
        except FileNotFoundError:
            return "Dossier introuvable."
        except PermissionError:
            return "Accès refusé."
