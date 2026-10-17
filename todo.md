# TODO — questions et tâches pour Vincent

Fichier tenu par l'agent pendant le développement autonome (phases 10 à 14, à partir du 17/10/2026).
Chaque entrée indique le contexte, ce que j'ai fait en attendant et ce dont j'ai besoin.
Pour répondre : un commit distant modifiant ce fichier, dont le message commence par `asupr` ou bien une édition locale du fichier sur une ligne commençant par "->"

> Ce fichier est à la racine à ta demande (exception à la règle « pas de fichier à la racine » du CLAUDE.md).
> Supprime-le ou déplace-le quand tout est traité.

## Réponses du 17/10 traitées

VFSciTools intégré, push après chaque étape, notebooks exécutés dans la doc, tailles des mires, spectre solaire
CALSPEC, CSV exoplanet.eu, durée de construction RTD et Colab : validés (« Ok pour tout »), rien à faire.

## Phase 13 terminée (17/10)

Couche de service et serveur MCP : `lumos mcp` (stdio et HTTP), ressources, prompts, opération avec
confirmation humaine, réponses pour LLM, guide « AI assistant », évaluation UC1/UC8/UC9 (ADR-0015 à 0017).
Le SDK MCP installé est la version 2 (`MCPServer`) : l'extra `mcp` demande désormais `mcp>=2.3,<3`.

## Tâches que je ne peux pas faire moi-même

- [ ] **Connexion de `gh`** : GitHub CLI est installé (winget) mais doit être connecté une fois, de façon
  interactive : `gh auth login` dans un nouveau terminal. Ensuite je pourrai suivre la CI.
- [ ] **Critère d'acceptation v0.6** : la commande `claude` est maintenant installée (winget
  `Anthropic.ClaudeCode`, dans un nouveau terminal). Pour valider :
  `claude mcp add lumos -- D:\LUMOS\.venv\Scripts\lumos.exe mcp`, puis dans `claude` :
  « Charge le préréglage coronographe de LUMOS et donne-moi le contraste brut entre 3 et 4 λ/D ».
  Optionnel, même principe automatisé (consomme un peu de ton forfait) : `python tools/mcp_eval.py --claude-code`.

## Décisions du 17/10 (2e série)

- Pas de clé API Claude : l'évaluation hebdomadaire est supprimée ; les scénarios tournent à chaque CI avec
  leurs solutions scriptées (gratuit) et, à la demande, avec Claude Code et ton abonnement (ADR-0017 amendé).
- Traductions françaises des nouvelles pages : en P15.
- HTTP local uniquement sans authentification jusqu'à P14 : validé.
- Pyright assoupli pour `tests/` seulement : n'affecte ni le code de la bibliothèque ni ses vérifications.
