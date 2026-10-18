# TODO — questions et tâches pour Vincent

Fichier tenu par l'agent pendant le développement autonome (phases 10 à 14, à partir du 17/10/2026).
Chaque entrée indique le contexte, ce que j'ai fait en attendant et ce dont j'ai besoin.
Pour répondre : un commit distant modifiant ce fichier, dont le message commence par `asupr` ou bien une édition locale du fichier sur une ligne commençant par "->"

> Ce fichier est à la racine à ta demande (exception à la règle « pas de fichier à la racine » du CLAUDE.md).
> Supprime-le ou déplace-le quand tout est traité.

## À faire maintenant (18/10) — une commande dans ton terminal VPS

- [x] **Autoriser ma clé SSH sur le VPS** (fait, merci) : je n'ai pas accès au terminal VS Code connecté au VPS, et le serveur
  refuse la connexion sans mot de passe. J'ai créé la clé `~/.ssh/id_ed25519_vps1` et le profil `VPS1`
  (`~/.ssh/config`, utilisateur `ubuntu`). Colle cette ligne **dans le terminal connecté au VPS** :

  ```
  mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHg599ulXRUtCGxmszHR4q6nPhrLlzcgvvLAbMy49qb+ vince@windows->VPS1' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
  ```

  Ensuite `ssh VPS1` marche depuis PowerShell et Git Bash ; je teste régulièrement et je déploie dès que ça passe
  (Docker, clone du dépôt, Caddy et TLS pour `lumos.vincent.foriel.pro`).
-> C'est fait
- [ ] **WSL n'est pas installé** sur ta machine (`wsl -l` : « Le sous-système Windows pour Linux n'est pas
  installé »). Après `wsl --install`, dans WSL : `mkdir -p ~/.ssh && cp /mnt/c/Users/vince/.ssh/{config,id_ed25519_vps1*} ~/.ssh/ && chmod 600 ~/.ssh/id_ed25519_vps1 ~/.ssh/config`.
-> j'ai installé le WSL via wsl --install et j'ai essayé d'installer une distribution ubuntu depuis le MS store mais quand je le lance il me fait une erreur indiquant une fonctionnalité manquante

## Où j'en suis (18/10, arrêt pour quota)

- Phase 15 : **P15.1 fait** (revue d'API, ADR-0021, gel 1.x). **P15.2 en cours** : traductions FR faites pour le
  guide utilisateur, l'interface web, l'assistant et le matériel ; restent les notebooks (`examples/`) puis la
  resynchronisation EN (deux phrases FR plus complètes que l'anglais : `web/index`, `web/development`, et
  `assistant/setup` ligne 93 à mettre à jour en anglais). Outil : `tools/translate_po.py`.
- Restent P15.3 (performance), P15.4 (qualité, sécurité, CSP à prévoir pour les fournisseurs de l'assistant),
  P15.5 (PyPI : il faudra ton accord et un compte ; VPS).
- **VPS : instance publique en ligne** sur https://lumos.vincent.foriel.pro (Docker, Caddy + certificat
  Let's Encrypt, profil public, opération désactivée, quotas pour 2 cœurs / 4 Go dans `~/lumos-vps.toml`).
  Préréglage exécuté et lien de connecteur MCP vérifiés depuis Internet. Mise à jour :
  `ssh VPS1 'cd LUMOS && git pull && docker compose -f deploy/public/compose.yaml up -d --build'`.
  À faire : la version affichée est 0.0.0 (image construite sans historique git) ; tester le connecteur depuis
  Claude.ai / ChatGPT avec ton abonnement (panneau Assistant › Créer un lien de connecteur).

## Retours du 18/10 traités (P14.12 à P14.15)

- **Éditeur** : mode de pointage (main/sélection, touches H et V), badge « Non sauvegardé » + confirmation
  (nouveau contexte, fichier, préréglage, visite, fermeture de la page), suppression d'un lien seul (bouton au
  milieu du lien sélectionné), panneaux gauche/droite/bas redimensionnables et masquables (tailles mémorisées).
- **Études** : graine du contexte dans l'inspecteur (saisie ou tirage), paramètres libres avec a priori (bouton
  dé), dialogue *Étude* : balayage (plages lin/log, grille ou paires), Monte-Carlo (bruit indépendant par
  tirage), onglet *Étude* (courbe, histogramme, tableau, CSV). Préréglage **Monte-Carlo tolerancing** et
  notebook 11 (l'accord avec la loi du faisceau gaussien est vérifié).
- **Boucles** : il n'existait aucune vraie boucle de rétroaction. Ajout des rétroactions (ADR-0020, format
  `.lumos` 1.1), de `ctx.loop`, d'un analyseur Shack-Hartmann et d'un contrôleur intégrateur. Préréglage
  **Adaptive optics loop** (8,2 m, bande H) et notebook 12 : résidu 468 → 123 nm, l'erreur de fitting attendue
  étant 121 nm ; pic de la PSF ×14.
- **Assistant** : Claude, OpenAI, Gemini, OpenRouter et tout point d'accès compatible OpenAI (clés gardées dans
  l'onglet). Réponse à ta question sur la connexion sans clé : **ni Anthropic ni OpenAI ni Google ne
  permettent à une application tierce d'utiliser un abonnement** (Claude.ai, ChatGPT, Gemini) ; seules les clés
  d'API y donnent accès. J'ai donc fait les deux choses possibles :
  - **connexion OpenRouter** (vrai OAuth, pas de clé à copier ; OpenRouter donne accès à Claude, GPT, Gemini…) ;
  - **lien de connecteur** : LUMOS s'ajoute comme connecteur MCP dans Claude.ai ou ChatGPT, dont l'assistant
    travaille alors sur ton contexte ouvert (tu vois ses modifications en direct) **avec ton abonnement**.
    Cela demande l'instance publique en HTTPS (Claude.ai et ChatGPT doivent l'atteindre) : à tester dès que le
    VPS sera en ligne.

## Décisions du 18/10 (à valider)

- Les retours ont été traités comme des étapes P14.12 à P14.15 avant la phase 15 (plan mis à jour).
- Format `.lumos` passé en **1.1** (rétroactions) avant le gel de la v1.0 ; les fichiers 1.0 restent lus.
- Une graine tirée au hasard fait désormais 32 bits (128 bits auparavant : inexacte en JSON et dans le navigateur).
- Un balayage garde le même bruit à chaque point par défaut ; un Monte-Carlo tire un bruit indépendant par point.
- Une boucle garde son état final (commandes du miroir, horloge) ; *Annuler* le rétablit.
- Notebooks de la phase 16 renumérotés 13 et 14.

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
- [x] ~~**Docker**~~ : tu m'autorises à l'installer moi-même ; je le fais sur le VPS (pas besoin sur ta machine).
- [ ] (ancien) **Docker** n'est pas installé sur ta machine : l'image est construite et démarrée par le job CI `docker`
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
