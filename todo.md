# TODO — questions et tâches pour Vincent

Fichier tenu par l'agent pendant le développement autonome (phases 3 à 15).
Chaque entrée indique le contexte, ce que j'ai fait en attendant et ce dont j'ai besoin.

> Ce fichier est à la racine à ta demande (exception à la règle « pas de fichier à la racine » du CLAUDE.md).
> Supprime-le ou déplace-le quand tout est traité.

## Tâches que je ne peux pas faire moi-même

- [ ] **Publier `vfscitools` sur PyPI** (nécessaire pour P7.3/P7.4 : `.plt` via pltedit, `vfst.archive`).
  En attendant : intégration derrière un import optionnel, avec repli documenté (voir P7).

- [ ] **Mesurer le gain GPU** (critère d'acceptation de la phase 4). Cette machine a bien un GPU (GeForce
  GTX 1060 6 Go, pilote 582.28, CUDA 13.0), mais aucun backend de LUMOS ne peut l'utiliser : `jaxlib` est la
  version CPU (JAX ne publie pas de version CUDA pour Windows natif, seulement Linux/WSL2) et ni CuPy ni
  PyTorch ne sont installés, d'où `gpus=()` dans `lumos.hardware_report()`. Pour l'utiliser : installer
  l'extra `[gpu]` (`cupy-cuda12x` ; le pilote CUDA 13 exécute les programmes CUDA 12, et CUDA 12 gère
  encore l'architecture Pascal de la 1060), avec le CUDA Toolkit 12.x ou les bibliothèques CUDA installées
  par pip ; ou JAX CUDA sous WSL2. Ensuite :
  `python -m pytest -m benchmark tests/benchmarks/test_bench_propagation.py`, et reporter les temps dans
  `docs/learn/performance.md`. Dis-moi si tu veux que je fasse l'installation de CuPy dans le `.venv`.
