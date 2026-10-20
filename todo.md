# TODO — questions et actions pour Vincent

Uniquement ce qui attend une réponse ou une action de ta part ; tout ce qui est traité est retiré (l'historique
est dans les commits et les ADR). Pour répondre : une ligne commençant par `->` sous l'entrée.

> Fichier à la racine à ta demande (exception à la règle du CLAUDE.md).

## Actions de ton côté

- [ ] **Connecteur Claude.ai « Lumos »** : il échouait parce que son lien était mort avec ta session de navigateur
  (« unknown or revoked connector link »). Corrigé et déployé : un lien vit maintenant 90 jours, survit à la
  fin de la session et aux redémarrages. À faire une fois : créer un nouveau lien (panneau Assistant › Créer un
  lien de connecteur) et remplacer l'URL du connecteur dans Claude.ai.
- [ ] **PyPI (pour P15.5, pas urgent)** : créer un compte sur https://pypi.org (2FA), puis y déclarer un
  *trusted publisher* en attente : projet `lumos-maxima`, dépôt `VForiel/LUMOS`, workflow `publish.yml`,
  environnement `pypi` (et de même sur https://test.pypi.org avec l'environnement `testpypi`). Aucun mot de passe ni jeton à me transmettre.
- [ ] **WSL (optionnel)** : la virtualisation marche maintenant, mais aucune distribution n'est installée
  (`wsl -l -v`). Si tu veux Ubuntu : `wsl --install -d Ubuntu` (crée ton utilisateur Linux), puis
  `mkdir -p ~/.ssh && cp /mnt/c/Users/vince/.ssh/{config,id_ed25519_vps1*} ~/.ssh/ && chmod 600 ~/.ssh/id_ed25519_vps1 ~/.ssh/config`.
