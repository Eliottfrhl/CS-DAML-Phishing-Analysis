# src/utils/url_parser.py
from __future__ import annotations
from typing import List
import re
import tldextract
import wordninja

class URLParser:
    """
    Utilitaires génériques de parsing/cleaning/segmentation d'URL.
    Aucune dépendance à un modèle : peut être réutilisé dans n'importe quel pipeline.
    """

    @staticmethod
    def strip_scheme_www(url: str) -> str:
        """Supprime schéma (http/https/ftp) et 'www.'"""
        u = re.sub(r'^\w+://', '', url.lower())
        u = re.sub(r'^www\.', '', u)
        return u

    @staticmethod
    def simple_clean(url: str) -> str:
        """
        Nettoyage léger pour approches char-ngrams : on garde les séparateurs
        qui portent de l'info (. / ? & = - _ : #), on enlève les espaces.
        """
        u = URLParser.strip_scheme_www(url)
        u = re.sub(r'\s+', '', u)
        return u

    @staticmethod
    def parse_domain_words(url: str) -> List[str]:
        """
        Découpe le domaine en mots probables via wordninja. Retourne [] si absent.
        """
        ext = tldextract.extract(url)
        domain = ext.domain or ""
        return wordninja.split(domain) if domain else []

    @staticmethod
    def tokenize_words(url: str) -> List[str]:
        """
        Tokenisation 'mots' pour embeddings (FastText, etc.) :
        - tokens du domaine (wordninja)
        - sous-domaine (split . et -) + suffix (tld)
        - path/query grossière (split par séparateurs), avec wordninja si segments longs
        - normalisation simple (chiffres -> <num>, lower)
        """
        tokens = []

        # 1) domain -> tokens
        tokens.extend(URLParser.parse_domain_words(url))

        # 2) subdomain + suffix
        try:
            ext = tldextract.extract(url)
            if ext.subdomain:
                for part in re.split(r'[.\-]+', ext.subdomain):
                    if part:
                        tokens.append(part)
            if ext.suffix:
                tokens.append(ext.suffix)
        except Exception:
            pass

        # 3) path & query
        u = URLParser.strip_scheme_www(url)
        parts = re.split(r'[/\?\&=\.\-_\:\#]+', u)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) >= 10:
                subs = wordninja.split(p)
                if len(subs) > 1:
                    tokens.extend([s for s in subs if s])
                else:
                    tokens.append(p)
            else:
                tokens.append(p)

        # 4) normalisation chiffres / lower
        norm = []
        for t in tokens:
            t = t.lower()
            if t.isdigit():
                norm.append("<num>")
            else:
                norm.append(t)
        return norm

    @staticmethod
    def parse(url: str) -> list[str]:
        """
        Compat util : comportement original demandé (renvoie tokens de domaine).
        Gardé pour rétro-compatibilité.
        """
        return URLParser.parse_domain_words(url)
