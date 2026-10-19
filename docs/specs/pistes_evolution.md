# Pistes d'évolution — retours du 19/10/2026 (à discuter, rien n'est implémenté)

Trois demandes de la revue du 19/10 demandent d'abord une décision. Pour chacune : le constat, ce qui est
faisable, une proposition, l'effort estimé et les questions à trancher. Une fois tranchées, elles deviendront des
étapes du plan (avec une ADR chacune).

---

## 1. Ports visuels des composants multi-canaux (MMI, puces photoniques)

### Constat

Un `ModalField` porte N canaux sur **un seul port** (`in`, `out`) : c'est un bus. Le réglage `channels` choisit
les canaux sur lesquels agit un composant (coupleur sur 0 et 2…). C'est compact et exact pour le calcul, mais un
MMI 4×4 dessiné avec une seule entrée et une seule sortie ne ressemble pas au composant, et câbler une puce
oblige à raisonner en indices de canaux.

### Options

| | A. Ports indexés dans le cœur | B. Ports dessinés, bus conservé | C. Blocs Démux / Mux explicites |
|---|---|---|---|
| Principe | un port peut avoir N voies : `mmi.in[2]` se connecte à `guide.out` ; le moteur assemble le bus | le bloc dessine N broches regroupées par une accolade ; une connexion porte toujours le bus entier | `Demux(N)` éclate un bus en N ports, `Mux(N)` les regroupe |
| Rendu | un fil par guide d'onde, comme un schéma de puce | lisible, mais les fils restent des bus | correct, mais chargé (2 blocs de plus par composant) |
| Cœur | connexions vers une voie, fusion des voies (ordre, longueurs d'onde), validation | aucun changement | deux éléments de plus |
| Format `.lumos` | 1.3 (`"to": "mmi.in[2]"`) | inchangé | inchangé |
| Effort | 3 à 4 jours (cœur, service, éditeur, docs) | 1 jour | 1 jour |

### Proposition

**A**, limité aux ports de signaux `ModalField` : c'est la seule option qui donne le schéma qu'on attend d'une
puce, et elle reste compatible (une connexion sans indice continue de porter le bus entier). Le nombre de voies
d'un port vient du composant : `MMI.n_ports`, la forme de `TransferMatrix`, le nombre de canaux d'un groupe. Les
réglages `channels` restent pour le Python et les documents existants.

### Questions

1. Un port à voies peut-il recevoir à la fois un bus et des voies isolées (mélange interdit, ou voies libres
   complétées par le bus) ?
2. Une voie non connectée : erreur, ou canal vide (amplitude nulle) ?
3. Faut-il étendre l'idée aux autres signaux à plusieurs composantes (faisceaux incohérents, cubes) ? Ma
   recommandation : non, pas de cas d'usage aujourd'hui.

---

## 2. Composants créés par l'utilisateur, éditeur intégré et catalogue (« marketplace »)

### Ce qui existe

- **Mes blocs** : un groupe sauvegardé est réutilisable (palette, fichier `.lumos` de type `group`) — c'est déjà un
  composant utilisateur sans code.
- **Greffons** : un paquet Python qui déclare le point d'entrée `lumos.elements` enregistre ses types
  (`Registry.load_plugins`, étape P1.5) ; **`lumos.processing.script`** fait d'une fonction un élément (ADR-0013).

### Faisabilité d'un éditeur intégré

Oui, en deux vues qui ne sont pas réversibles l'une dans l'autre (comme tu le pressens) :

- **Vue simple (formulaire)** : nom, catégorie, description ; ports (type de signal) ; paramètres (unité, bornes,
  valeur par défaut, description) ; et un **comportement choisi dans des gabarits** : matrice de transfert,
  gain/atténuation, déphasage, transmission spectrale tabulée, réponse impulsionnelle, ou « composite » (un
  groupe). L'éditeur produit un document déclaratif (`.lumos` de type `component`), sans code exécutable. Couvre la
  majorité des besoins de banc.
- **Vue code** : une classe `Element` en Python (`transfer(values, inputs)` pure), avec l'aide du formulaire pour
  déclarer ports et paramètres. Passer du code au formulaire n'est possible que si le code suit un gabarit.

### Le point dur : la sécurité

Exécuter le code d'un utilisateur sur l'**instance publique**, c'est lui donner la machine (accès disque, réseau,
autres sessions). Les possibilités :

1. **Code seulement en local** (profil `local`, la machine de l'utilisateur) ; sur l'instance publique, seuls les
   composants **déclaratifs** et ceux installés par l'administrateur. Simple, sûr. **Recommandé.**
2. Code exécuté **dans le navigateur** (Pyodide/WebAssembly) : pas de risque serveur, mais NumPy seulement (pas
   JAX/GPU), lent, et il faudrait faire tourner tout le calcul côté client.
3. Bac à sable serveur (processus isolé : nsjail/gVisor, sans réseau, quotas) : faisable mais lourd à maintenir et
   à auditer (P15.4).

### Règles de validité vérifiées (commande `lumos component check` et éditeur)

Identifiant stable (`auteur.nom`), version et migrations ; ports typés ; chaque paramètre physique a une unité, des
bornes cohérentes et une description ; transfert pur (pas d'état caché, pas d'E/S, pas de `pickle`) ; aller-retour
`.lumos` identique ; schéma JSON valide ; au moins un test avec une valeur de référence indépendante (conservation
de l'énergie, cas limite analytique) ; documentation (docstring NumPy) ; licence déclarée.

### Catalogue

Un **index public** (dépôt Git : une fiche par composant, auteur, version, licence, tests passés) consultable depuis
la palette (« Catalogue »). Les composants déclaratifs s'importent d'un clic ; les composants à code s'installent
comme paquets Python (`pip install lumos-component-…`, nommage réservé sur PyPI) en local. Une CI de l'index
exécute `lumos component check` sur chaque contribution. Une page de doc « Créer et partager un composant »
décrit les deux voies.

### Effort

Vue simple + format `component` + validation : ~1 semaine. Catalogue (index, CI, palette) : ~1 semaine. Vue code
(local) : 3 jours. ADR à écrire (format, sécurité).

### Questions

1. D'accord pour **pas de code utilisateur sur l'instance publique** (option 1) ?
2. Le catalogue : dépôt GitHub dédié (contributions par pull request) — ou autre hébergement ?
3. Quels gabarits de comportement en priorité ?

---

## 3. Analyses procédurales (blocs d'instructions)

### Exemple de la revue

« Fermer un obturateur, prendre des images, les moyenner pour faire un *dark*, puis l'utiliser pour les mesures
suivantes. » C'est une **procédure** : une suite d'actions et de mesures dans le temps, avec des résultats
intermédiaires réutilisés — différente du graphe (un calcul) et des études (balayage, Monte-Carlo, boucle).

### Ce qui existe

Actions des éléments (`shutter.close()`, `stage.move_to(...)`) avec temps prédit/mesuré ; **séquences**
enregistrées et rejouées (ADR-0014) ; traitements (`Stack`, `dark`, `flat_field`) ; boucles et études ; le tout
journalisé et identique en simulation et en opération.

### Proposition : un onglet « Procédures »

Une procédure est une liste d'**instructions** (blocs empilés, façon Scratch/Blockly, pas un graphe) :

- *Action* : appeler une action d'un élément (fermer l'obturateur, déplacer la platine de 2 mm) ;
- *Mesure* : calculer des ports (`camera.out`) et **nommer** le résultat (`darks`) ;
- *Répéter N fois* / *pour chaque valeur* (boucle sur une liste ou une plage) ;
- *Traiter* : appliquer une fonction de traitement à des résultats nommés (`dark = moyenne(darks)`) ;
- *Utiliser* : injecter un résultat nommé dans un élément ou un paramètre (soustraction du dark, gain) ;
- *Attendre*, *Si / sinon* sur une valeur, *Enregistrer* (archive), *Afficher* (navigateur de résultats).

L'exemple devient : `fermer(obturateur)` → `répéter 20 : mesurer camera.out → darks` →
`dark = moyenne(darks)` → `ouvrir(obturateur)` → `répéter 10 : mesurer camera.out → images` →
`science = soustraire(images, dark)`.

Une procédure est un document JSON (format `procedure`, sans code), exécuté par le service comme une tâche
(progression, annulation, journal, quotas), en simulation comme en opération (confirmation humaine des actions
matérielles, comme aujourd'hui). Le **mode avancé** est un script Python qui utilise l'API (`ctx.run`,
actions, traitements) : comme pour les composants, il ne revient pas vers les blocs, et il ne s'exécute qu'en
local. L'éditeur peut proposer « Exporter en Python » (blocs → script), pas l'inverse.

### Effort

Format et moteur d'exécution (service, MCP, API web) : ~1 semaine ; éditeur de blocs : ~1 semaine (bibliothèque
Blockly, licence Apache-2.0, ou éditeur maison en liste) ; doc, préréglage « dark puis mesure », notebook.

### Questions

1. Blocs empilés (Blockly) ou liste d'étapes éditable (plus simple, moins ludique) ?
2. Les résultats nommés doivent-ils pouvoir alimenter les **études** (ex. un Monte-Carlo sur une procédure
   entière) ?
3. Le mode script sur l'instance publique : même réponse que pour les composants (non) ?

---

## Où les placer

Ces trois chantiers ajoutent des fonctions ; aucun n'est nécessaire à la v1.0. Je propose de finir la phase 15
(v1.0), puis d'en faire une **phase 18 « Extensibilité »** : ports à voies (format 1.3), composants déclaratifs et
catalogue, procédures. L'ordre recommandé : procédures (forte valeur pour les bancs réels), ports à voies,
composants.
