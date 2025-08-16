import os
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from sentence_transformers import SentenceTransformer, util
from ark_commands.utils import remove_accents

class SubjectType(Enum):
    FILES = "fichiers"
    FOLDERS = "dossiers"
    IMAGES = "images"
    VIDEOS = "videos"
    DOCUMENTS = "documents"
    MUSIC = "musique"
    ARCHIVES = "archives"
    EXECUTABLES = "executables"
    TEXT_FILES = "texte"

@dataclass
class ExtractedSubject:
    subject_type: SubjectType
    location: str
    filters: List[str]
    count_requested: bool = False
    list_requested: bool = False
    confidence: float = 0.0

class SubjectExtractor:
    def __init__(self, model: SentenceTransformer, base_path: str = ""):
        self.model = model
        self.base_path = base_path or os.path.expanduser("~")
        
        # Exemples pour chaque type (IA)
        self.subject_examples = {
            SubjectType.FILES: ["montrer les fichiers", "lister les éléments"],
            SubjectType.FOLDERS: ["voir les dossiers", "lister les répertoires"],
            SubjectType.IMAGES: ["montrer les photos", "images jpg png"],
            SubjectType.VIDEOS: ["lister les vidéos mp4", "voir les films"],
            SubjectType.DOCUMENTS: ["documents pdf", "fichiers texte"],
            SubjectType.MUSIC: ["musique mp3", "fichiers audio"],
            SubjectType.ARCHIVES: ["archives zip", "fichiers compressés"],
            SubjectType.EXECUTABLES: ["programmes exe", "applications"],
            SubjectType.TEXT_FILES: ["fichiers txt", "notes texte"]
        }
        
        self.action_examples = {
            'count': ["combien de", "nombre de", "compter"],
            'list': ["lister", "montrer", "afficher"]
        }
        
        # Extensions par type
        self.type_extensions = {
            SubjectType.IMAGES: ['.jpg', '.png', '.gif', '.webp'],
            SubjectType.VIDEOS: ['.mp4', '.avi', '.mkv', '.mov'],
            SubjectType.DOCUMENTS: ['.pdf', '.doc', '.docx', '.txt'],
            SubjectType.MUSIC: ['.mp3', '.wav', '.flac'],
            SubjectType.ARCHIVES: ['.zip', '.rar', '.7z'],
            SubjectType.EXECUTABLES: ['.exe', '.app', '.deb'],
            SubjectType.TEXT_FILES: ['.txt', '.md', '.log']
        }
        
        # Pré-calcul des embeddings
        self._precompute_embeddings()

    def _precompute_embeddings(self):
        self.subject_embeddings = {
            stype: self.model.encode(examples) 
            for stype, examples in self.subject_examples.items()
        }
        self.action_embeddings = {
            action: self.model.encode(examples) 
            for action, examples in self.action_examples.items()
        }

    def extract_subjects(self, phrase: str) -> List[ExtractedSubject]:
        phrase_clean = remove_accents(phrase.lower())
        user_emb = self.model.encode([phrase])
        
        # Détection IA + fallback
        subjects = self._detect_subjects_ai(user_emb, phrase_clean)
        count_req, list_req = self._detect_actions_ai(user_emb, phrase_clean)
        location = self._extract_location(phrase_clean)
        filters = self._extract_filters(phrase_clean)
        
        return [ExtractedSubject(
            subject_type=stype, location=location, filters=filters,
            count_requested=count_req, list_requested=list_req, confidence=conf
        ) for stype, conf in subjects]

    def _detect_subjects_ai(self, user_emb, phrase: str) -> List[Tuple[SubjectType, float]]:
        results = []
        for stype, embeddings in self.subject_embeddings.items():
            sim = util.cos_sim(user_emb, embeddings)[0].max().item()
            if sim > 0.3:
                results.append((stype, sim))
        
        # Fallback mots-clés si rien détecté
        if not results:
            keywords = {
                SubjectType.IMAGES: ["image", "photo", "jpg", "png"],
                SubjectType.VIDEOS: ["video", "mp4", "film"],
                SubjectType.DOCUMENTS: ["document", "pdf", "doc"],
                SubjectType.FOLDERS: ["dossier", "folder"],
                SubjectType.MUSIC: ["music", "mp3", "audio"]
            }
            for stype, words in keywords.items():
                if any(w in phrase for w in words):
                    results.append((stype, 0.5))
            if not results:
                results.append((SubjectType.FILES, 0.4))
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _detect_actions_ai(self, user_emb, phrase: str) -> Tuple[bool, bool]:
        count_sim = util.cos_sim(user_emb, self.action_embeddings['count'])[0].max().item()
        list_sim = util.cos_sim(user_emb, self.action_embeddings['list'])[0].max().item()
        
        count_req = count_sim > 0.4 or any(w in phrase for w in ["combien", "nombre"])
        list_req = list_sim > 0.4 or any(w in phrase for w in ["liste", "afficher", "voir"])
        
        return count_req, list_req

    def _extract_location(self, phrase: str) -> str:
        # Simple extraction par mots-clés
        for keyword in ["dans", "dans le", "sur"]:
            if keyword in phrase:
                parts = phrase.split(keyword, 1)
                if len(parts) > 1:
                    remaining = parts[1].strip()
                    if remaining:  # Vérifier que la partie après le mot-clé n'est pas vide
                        folder_name = remaining.split()[0] if remaining.split() else ""
                        if folder_name:
                            folder_path = self._find_folder(folder_name)
                            if folder_path:
                                return folder_path
        return self.base_path

    def _find_folder(self, folder_name: str) -> Optional[str]:
        try:
            for entry in os.listdir(self.base_path):
                entry_path = os.path.join(self.base_path, entry)
                if os.path.isdir(entry_path) and folder_name.lower() in entry.lower():
                    return entry_path
        except (PermissionError, FileNotFoundError):
            pass
        return None

    def _extract_filters(self, phrase: str) -> List[str]:
        # Extensions et noms entre guillemets
        filters = re.findall(r'\b\w+\.\w{2,4}\b', phrase)  # Extensions
        filters.extend(re.findall(r'"([^"]+)"', phrase))   # Noms entre guillemets
        return filters


class SubjectOfCommands:
    def __init__(self, model: SentenceTransformer, base_path: str = ""):
        self.base_path = base_path or os.path.expanduser("~")
        self.extractor = SubjectExtractor(model, self.base_path)
        self.subjects: List[ExtractedSubject] = []

    def analyze_phrase(self, phrase: str) -> List[ExtractedSubject]:
        self.subjects = self.extractor.extract_subjects(phrase)
        return self.subjects

    def get_primary_subject(self) -> Optional[ExtractedSubject]:
        return max(self.subjects, key=lambda x: x.confidence) if self.subjects else None

    def get_files_by_subject(self, subject: ExtractedSubject) -> List[str]:
        try:
            entries = os.listdir(subject.location)
            filtered = []
            
            for entry in entries:
                path = os.path.join(subject.location, entry)
                
                # Filtrage par type
                if subject.subject_type == SubjectType.FOLDERS and not os.path.isdir(path):
                    continue
                if subject.subject_type != SubjectType.FOLDERS and not os.path.isfile(path):
                    continue
                
                # Filtrage par extension
                if (subject.subject_type in self.extractor.type_extensions and 
                    os.path.splitext(entry)[1].lower() not in self.extractor.type_extensions[subject.subject_type]):
                    continue
                
                # Filtres personnalisés
                if subject.filters and not any(f.lower() in entry.lower() for f in subject.filters):
                    continue
                
                filtered.append(entry)
            
            return filtered
        except (FileNotFoundError, PermissionError):
            return []

    def count_by_subject(self, subject: ExtractedSubject) -> int:
        return len(self.get_files_by_subject(subject))