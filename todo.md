# TODO — questions et tâches pour Vincent

Fichier tenu par l'agent pendant le développement autonome (phases 3 à 15).
Chaque entrée indique le contexte, ce que j'ai fait en attendant et ce dont j'ai besoin.

> Ce fichier est à la racine à ta demande (exception à la règle « pas de fichier à la racine » du CLAUDE.md).
> Supprime-le ou déplace-le quand tout est traité.

## Tâches que je ne peux pas faire moi-même

- [ ] **Publier `vfscitools` sur PyPI** (nécessaire pour P7.3/P7.4 : `.plt` via pltedit, `vfst.archive`).
  En attendant : intégration derrière un import optionnel, avec repli documenté (voir P7).

## À examiner plus tard

- **JAX sur GPU plus lent que CuPy** sur la même GTX 1060 (propagation `asm` 1024² : 36 ms contre 13 ms,
  machine au repos ; voir `docs/learn/performance.md`) : profiler le chemin JAX (transferts hôte, cache des
  noyaux, compilation) avant d'en faire le backend GPU préféré, comme le prévoit l'ADR-0004.
