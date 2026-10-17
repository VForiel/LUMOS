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

## Phase 14 terminée (17/10)

Interface web complète : backend FastAPI (`lumos web`), authentification (mot de passe Argon2id, clé SSH par
défi `ssh-keygen -Y sign`, `lumos users`, `lumos login`), profils local/public et quotas, éditeur React Flow
(formulaires à unités, annuler/rétablir, groupes, copier-coller, blocs), exécution et inspection,
préréglages et visite guidée EN/FR, panneau opération, panneau assistant (MCP sur `/mcp`), image Docker et
guide d'hébergement (ADR-0018, ADR-0019 — le plan l'appelait « ADR-0008 », numéro déjà pris). 30 tests
navigateur (Playwright) tournent en CI, ainsi qu'un test de parité Python/web/MCP. Pour l'essayer :
`cd web; npm ci; npm run build` puis `lumos web --open` (ou `docker compose -f deploy/compose.yaml up --build`).

Corrigés au passage : schéma JSON des réglages quantités, recherche de doc, course de l'arrêt d'urgence du
pilote LL-LS100 (fin de mouvement au lieu d'interruption), tests MCP/pilote instables en CI, CI rouge (pyright
sans `pyvisa`, formatage).

## Tâches que je ne peux pas faire moi-même

- [ ] **Critère d'acceptation v0.7 « UC5 sur l'instance publique »** : il faut un VPS et un nom DNS. Tout est
  prêt dans `deploy/public/` (Caddy + TLS automatique, quotas pour 4 cœurs) et décrit dans
  `docs/web/hosting.md`. Le même parcours est testé en CI sur une instance au profil `public` locale.
- [ ] **Docker** n'est pas installé sur ta machine : l'image est construite et démarrée par le job CI `docker`
  seulement. Si tu veux la tester : installer Docker Desktop puis la commande compose ci-dessus.
- [ ] **Panneau assistant** : il demande une clé API Anthropic (gardée dans l'onglet du navigateur, facturée sur
  le compte de la clé). Tu as indiqué ne pas en avoir : le panneau reste optionnel (désactivable avec
  `assistant = false`) ; sans clé, Claude Code peut se brancher sur `/mcp` d'une instance avec un jeton de
  `lumos login` (voir `docs/web/assistant.md`). Les tests utilisent un faux serveur Claude.

- [x] **Connexion de `gh`** : GitHub CLI est installé (winget) mais doit être connecté une fois, de façon
  interactive : `gh auth login` dans un nouveau terminal. Ensuite je pourrai suivre la CI.
- [x] **Critère d'acceptation v0.6** : la commande `claude` est maintenant installée (winget
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

## Décisions prises en phase 14 (à valider ou corriger)

- Galerie de préréglages + visite guidée affichées dans le canevas vide **à la première visite seulement**.
- Attribution « React Flow » laissée visible (licence MIT, ses auteurs demandent de ne la masquer qu'avec un
  abonnement Pro).
- Dossiers `web/` (interface) et `deploy/` (Docker, hébergement) à la racine du dépôt, comme dans
  l'arborescence cible du cahier des charges ; CLAUDE.md mis à jour.
- L'image Docker utilise le profil local avec `--listen-all` (écoute 0.0.0.0 dans le conteneur, port publié
  sur 127.0.0.1 seulement) ; le profil local refuse toujours toute autre adresse hors conteneur.
