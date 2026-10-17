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

- [ ] **Critère d'acceptation v0.6** : depuis Claude Code, construire, exécuter et analyser le préréglage
  coronographe en langage naturel. Les tests le vérifient avec un client MCP scripté, mais pas avec Claude Code
  lui-même. Pour l'essayer : `claude mcp add lumos -- D:\LUMOS\.venv\Scripts\lumos.exe mcp`, puis demander
  « Charge le préréglage coronographe de LUMOS et donne-moi le contraste brut entre 3 et 4 λ/D ».
- [ ] **Secret `ANTHROPIC_API_KEY`** dans les paramètres du dépôt GitHub, pour que le workflow hebdomadaire
  « MCP evaluation » fasse résoudre les scénarios par Claude (sans lui, il ne lance que les solutions scriptées).
- [ ] **Nouveau PC** : `gh` (GitHub CLI) n'est pas installé, je n'ai donc pas pu vérifier l'état de la CI. Le
  `.venv` est recréé (Python 3.13, extras dev, docs, jax, catalogs, hardware, web, mcp, validation, hdf5 ;
  `anthropic` en plus pour l'évaluation) ; pas de GPU, les tests GPU sont désélectionnés comme avant.
- [ ] **CI à surveiller** : `pydantic` n'était pas déclaré alors que `lumos.service` l'utilise (corrigé) ; la CI
  installe maintenant l'extra `mcp`. Pyright 1.1.414 signalait des erreurs dans des tests existants : j'ai assoupli
  trois règles pour `tests/` seulement (commit `build:`).

## Choix techniques à valider

- **Traductions françaises** : les nouvelles pages (`docs/assistant/*`) ne sont pas encore traduites, comme les
  guides du matériel avant elles. Je propose de tout traduire en P15 (« docs EN/FR complètes »).
- **HTTP sans authentification** : `lumos mcp --transport http` n'écoute que sur la machine locale, sauf
  `--allow-remote`, et refuse toujours l'opération du matériel aux clients distants, jusqu'à l'authentification
  de P14.
