# TODO — questions et tâches pour Vincent

Fichier tenu par l'agent pendant le développement autonome (phases 7 à 14, nuit du 16 au 17/10/2026).
Chaque entrée indique le contexte, ce que j'ai fait en attendant et ce dont j'ai besoin.

> Ce fichier est à la racine à ta demande (exception à la règle « pas de fichier à la racine » du CLAUDE.md).
> Supprime-le ou déplace-le quand tout est traité.

## Tâches que je ne peux pas faire moi-même

- [ ] **Publier `vfscitools` sur PyPI.** C'est maintenant une dépendance de base (`vfscitools>=0.1` dans
  `pyproject.toml`, NF-2) : `vfst.archive` crée les archives et `pltedit.save` les `.plt` (ADR-0010). En local
  je l'ai installé en mode éditable depuis `G:\VFSciTools`. **La CI GitHub échouera à l'installation tant que
  le paquet n'est pas sur PyPI.**
- [ ] **Pousser les commits** (`git push`) : je n'ai rien poussé.

## Choix techniques à valider

- [ ] **Notebooks dans la doc** : ils sont copiés de `examples/` vers `docs/examples/notebooks/` à la
  construction et rendus **sans sorties** (`nb_execution_mode = "off"`). Ils sont exécutés par les tests
  (`tests/integration/test_notebooks.py`, ~75 s). Veux-tu qu'ils soient exécutés à la construction de la doc
  (RTD : plus long, figures visibles) ? Il suffit de passer à `nb_execution_mode = "cache"`.
- [ ] **Géométrie des mires** (`SiemensStar`, `Checkerboard`, `BarTarget`) exprimée **en pixels** de leur grille
  pour servir à la fois les scènes du ciel (pixel angulaire) et de laboratoire (pixel en µm). Alternative :
  deux jeux de paramètres (angle / longueur).
- [ ] **Scènes de laboratoire** : une carte de radiance à distance finie est vue comme à l'infini (pixels de
  `dx/d` rad, radiance conservée) ; la courbure des ondes d'un objet proche est négligée (ADR-0011). Les
  sources proches sont à modéliser comme champs (laser, lampe).
- [ ] **Spectre solaire de référence** : j'ai utilisé le spectre CALSPEC `sun_reference_stis_002` (Bohlin et al.
  2001, 0,12–2,7 µm) au lieu de la table ASTM E490 demandée (REQ-CAT-2), distribuée en tableur `.xls` par le
  NREL. Si tu tiens à E490, il faudra choisir une source de fichier lisible (CSV) ou accepter `openpyxl`/`xlrd`.
- [ ] **exoplanet.eu** : son serveur TAP (`voparis-tap-planeto.obspm.fr`) refusait les connexions le 16/10 ;
  le fournisseur lit le catalogue CSV complet (3,7 Mo, mis en cache). À revoir si le TAP revient.
- [ ] **Catalogues** : pas d'astroquery/pyvo (TAP et HTTP directs via la couche réseau sécurisée) ; l'extra
  `[catalogs]` ne contient plus que `requests` (ADR-0012). Les tests rejouent des réponses réelles
  enregistrées (`tools/record_catalog_fixtures.py`) ; aucun test `network` en direct n'existe pour l'instant.
- [ ] **Stratégie d'imagerie étendue « auto »** : sonde numérique d'invariance par translation (PSF hors axe
  recentrée vs PSF sur l'axe, tolérance 1 %) plutôt que des règles par type d'élément (principe 2). Le
  mode anisoplanatique n'est jamais choisi automatiquement (seulement sur demande).
