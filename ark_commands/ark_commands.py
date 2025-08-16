import os
from typing import Optional, List
from sentence_transformers import SentenceTransformer, util
from ark_commands.subject_extractor import SubjectOfCommands, SubjectType, ExtractedSubject

class ARKCommands:
    def __init__(self, model: SentenceTransformer, base_path: str = ""):
        self.model = model
        self.subject_manager = SubjectOfCommands(model, base_path)

        # Commandes avec contexte détaillé
        self.commands = {
            "combien de fichiers documents images dans le dossier": self._count_command,
            "compter le nombre d'éléments fichiers photos vidéos": self._count_command,
            "quel est le nombre total de documents pdf": self._count_command,
            "lister afficher tous les fichiers du répertoire": self._list_command,
            "montrer voir les documents images vidéos du dossier": self._list_command,
            "afficher le contenu la liste des éléments": self._list_command,
            "voir tous les fichiers photos dans le dossier": self._list_command
        }
        
        # Embeddings des commandes
        self.cmd_embeddings = self.model.encode(list(self.commands.keys()))

    def get_best_command(self, phrase: str, threshold: float = 0.4) -> Optional[str]:
        try:
            # Analyser les sujets
            subjects = self.subject_manager.analyze_phrase(phrase)
            if not subjects:
                return None

            # Détection de commande par IA
            user_emb = self.model.encode([phrase])
            scores = util.cos_sim(user_emb, self.cmd_embeddings)[0]
            best_score = scores.max()
            
            if best_score >= threshold:
                best_cmd = list(self.commands.keys())[scores.argmax()]
                return self.commands[best_cmd](subjects)
            
            # Fallback par mots-clés - seulement si score IA pas trop bas
            if best_score >= 0.2:  # Seuil minimal pour considérer le fallback
                phrase_lower = phrase.lower()
                if any(w in phrase_lower for w in ["combien", "nombre"]):
                    return self._count_command(subjects)
                elif any(w in phrase_lower for w in ["liste", "afficher", "voir"]):
                    return self._list_command(subjects)
            
            # Demande incompatible avec ARKCommands
            return None
        except Exception:
            # En cas d'erreur, retourner None plutôt que de planter
            return None

    def _count_command(self, subjects: List[ExtractedSubject]) -> str:
        results = []
        for subject in subjects:
            count = self.subject_manager.count_by_subject(subject)
            location = os.path.basename(subject.location) or "racine"
            results.append(f"{count} {subject.subject_type.value} dans {location}")
        return " | ".join(results)

    def _list_command(self, subjects: List[ExtractedSubject]) -> str:
        results = []
        for subject in subjects:
            files = self.subject_manager.get_files_by_subject(subject)[:10]  # Max 10
            location = os.path.basename(subject.location) or "racine"
            
            if files:
                file_list = "\n".join([f"  • {f}" for f in files])
                results.append(f"{subject.subject_type.value.title()} dans {location}:\n{file_list}")
            else:
                results.append(f"Aucun {subject.subject_type.value} dans {location}")
        
        return "\n\n".join(results)