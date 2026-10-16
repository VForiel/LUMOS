# TODO — questions et tâches pour Vincent

Fichier tenu par l'agent pendant le développement autonome (phases 3 à 15).
Chaque entrée indique le contexte, ce que j'ai fait en attendant et ce dont j'ai besoin.

> Ce fichier est à la racine à ta demande (exception à la règle « pas de fichier à la racine » du CLAUDE.md).
> Supprime-le ou déplace-le quand tout est traité.

## Tâches que je ne peux pas faire moi-même

- [ ] **Publier `vfscitools` sur PyPI** (nécessaire pour P7.3/P7.4 : `.plt` via pltedit, `vfst.archive`).
  En attendant : intégration derrière un import optionnel, avec repli documenté (voir P7).

- [ ] **Mesurer le gain GPU** (critère d'acceptation de la phase 4). Le GPU (GTX 1060 6 Go) est maintenant
  utilisable : CuPy (+ bibliothèques CUDA 12 de pip) dans le `.venv` Windows, et JAX CUDA dans Ubuntu/WSL
  (clone dans `/root/LUMOS` ; distribution déplacée sur `D:\WSL\Ubuntu` car C: était plein). Les deux suites
  passent sur GPU. Mes mesures ont été faussées par Helldivers 2 en cours d'exécution (CPU à 74 %, GPU
  occupé) : je ne les ai pas reportées. À relancer machine au repos, puis reporter dans
  `docs/learn/performance.md` :
  - Windows : `.venv\Scripts\python.exe -m pytest -m benchmark tests/benchmarks/test_bench_propagation.py`
  - WSL : `wsl -d Ubuntu -u root -- bash -lc "cd ~/LUMOS && .venv/bin/python -m pytest -m benchmark tests/benchmarks/test_bench_propagation.py"`

- [ ] **Créer ton utilisateur Ubuntu** : la distribution a été initialisée en `root`, sans interaction.
  `wsl -d Ubuntu -u root -- adduser <nom>`, puis `ubuntu config --default-user <nom>`.

## À examiner plus tard

- **JAX sur GPU plus lent que CuPy** sur la même carte (propagation `asm` 1024² : ~93 ms contre ~20-30 ms,
  mesures faites sous charge) : profiler le chemin JAX (transferts hôte, cache des noyaux, double précision).
