# TODO — questions et tâches pour Vincent

Fichier tenu par l'agent pendant le développement autonome (phases 10 à 14, à partir du 17/10/2026).
Chaque entrée indique le contexte, ce que j'ai fait en attendant et ce dont j'ai besoin.
Pour répondre : un commit distant modifiant ce fichier, dont le message commence par `asupr`.

> Ce fichier est à la racine à ta demande (exception à la règle « pas de fichier à la racine » du CLAUDE.md).
> Supprime-le ou déplace-le quand tout est traité.

## Suite donnée à tes réponses du 17/10

- **VFSciTools** : intégré à LUMOS (`lumos._vendor.pltedit` et `lumos._vendor.vfst_archive`, copiés tels quels
  depuis le commit `b0f862e`, licence MIT). `vfscitools` n'est plus une dépendance, donc RTD et la CI peuvent
  installer LUMOS. Amendement dans l'ADR-0010. Quand VFSciTools sera sur PyPI, on pourra revenir à la
  dépendance.
- **Push** : je pousse sur `origin/main` après chaque étape.
- **Notebooks dans la doc** : exécutés à la construction (`nb_execution_mode = "cache"`, la construction échoue
  si un notebook échoue), figures visibles. Chaque page commence par des boutons **Colab** et **Binder**.
  Sur Colab, une cellule d'installation (`tools/add_colab_setup.py`) installe LUMOS depuis GitHub et récupère
  les préréglages ; Binder utilise `.binder/`.
- **Mires** : tailles en angle (grille angulaire), en longueur (grille spatiale) ou en pixels (nombre ou
  `u.pix`), par exemple `SiemensStar(radius=50 * u.mas)`.
- **Spectre solaire** : je garde CALSPEC `sun_reference_stis_002` (Bohlin et al. 2001), de qualité équivalente
  à E490 sur 0,12–2,7 µm et lisible sans dépendance.
- **exoplanet.eu** : réessayé le 17/10. Le TAP répond en HTTP (port 80) mais refuse toujours HTTPS ; la couche
  réseau n'accepte que HTTPS vérifié, donc je garde le CSV complet.

## Tâches que je ne peux pas faire moi-même

- [ ] **Durée de construction RTD** : les notebooks sont maintenant exécutés par RTD (quelques minutes par langue
  en local). Si la construction dépasse la limite de RTD, il faudra repasser en `"off"` ou exclure les plus longs
  (`nb_execution_excludepatterns`).
- [ ] **Colab** : la cellule d'installation passe par `git+https://github.com/VForiel/LUMOS`. Elle ne marchera
  que si le dépôt est public.

## Choix techniques à valider

(rien pour l'instant)
