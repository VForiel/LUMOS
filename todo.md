# TODO — questions et tâches pour Vincent

Fichier tenu par l'agent pendant le développement autonome (phases 3 à 15).
Chaque entrée indique le contexte, ce que j'ai fait en attendant et ce dont j'ai besoin.

> Ce fichier est à la racine à ta demande (exception à la règle « pas de fichier à la racine » du CLAUDE.md).
> Supprime-le ou déplace-le quand tout est traité.

## Tâches que je ne peux pas faire moi-même

- [ ] **Publier `vfscitools` sur PyPI** (nécessaire pour P7.3/P7.4 : `.plt` via pltedit, `vfst.archive`).
  En attendant : intégration derrière un import optionnel, avec repli documenté (voir P7).

- [ ] **Mesurer le gain GPU** (critère d'acceptation de la phase 4) : je n'ai pas de GPU sur cette machine.
  Lance `python -m pytest -m benchmark tests/benchmarks/test_bench_propagation.py` sur une machine avec GPU
  CUDA (`pip install -e ".[dev,jax,gpu]"` ou JAX CUDA) et reporte les temps dans `docs/learn/performance.md`.

## Choix techniques à valider

- **P4.4 : Fresnel « deux étapes »** remplacé par un Fresnel en une étape par MFT (`fresnel_mft`), qui donne
  directement n'importe quel échantillonnage de sortie (but visé par le schéma à deux étapes). Idem pour
  l'ASM mis à l'échelle : MFT inverse plutôt que CZT. Détails et justification dans l'ADR-0005. Dis-moi si tu
  tiens au schéma à deux étapes ou à la CZT.

## Remarques

- P2.5 : bug trouvé et corrigé pendant P3.2. Le hook d'import JAX était consommé par
  `importlib.util.find_spec("jax")` (appelé par `backend.available()`), si bien que `Values` n'était pas
  enregistré comme pytree quand JAX était ensuite chargé par le backend. Un test de non-régression a été
  ajouté (commits `7782853`, `f8c004c`).
