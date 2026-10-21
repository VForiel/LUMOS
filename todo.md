# TODO — questions et actions pour Vincent

Uniquement ce qui attend une réponse ou une action de ta part ; tout ce qui est traité est retiré (l'historique
est dans les commits et les ADR). Pour répondre : une ligne commençant par `->` sous l'entrée.

> Fichier à la racine à ta demande (exception à la règle du CLAUDE.md).

## Questions

- [ ] **Publier `lumos-maxima` 1.0 sur PyPI ?** (dernière étape de la phase 15, P15.5) Tout est prêt et vérifié :
  CI verte, wheel construite et installée dans un environnement neuf (interface, catalogue, préréglages), instance
  publique déployée (version affichée corrigée). Une version publiée sur PyPI ne peut plus être réutilisée, d'où
  la question. Sur ton feu vert, je : renomme `[Unreleased]` en `[1.0.0]` dans le CHANGELOG, pousse le tag
  `v1.0.0`, crée la *release* GitHub (qui publie sur PyPI par *trusted publishing*), puis déploie le VPS sur
  `v1.0.0`. Un essai préalable sur TestPyPI est possible si tu préfères.

## Actions coté humain

- [ ] **Connecteur Claude.ai « Lumos »** : il échouait parce que son lien était mort avec ta session de navigateur
  (« unknown or revoked connector link »). Corrigé et déployé : un lien vit maintenant 90 jours, survit à la
  fin de la session et aux redémarrages. À faire une fois : créer un nouveau lien (panneau Assistant › Créer un
  lien de connecteur) et remplacer l'URL du connecteur dans Claude.ai.
