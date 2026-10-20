# Pistes d'évolution — retours du 19/10/2026 (à discuter, rien n'est implémenté)

Trois demandes de la revue du 19/10 demandent d'abord une décision. Pour chacune : le constat, ce qui est
faisable, une proposition, l'effort estimé et les questions à trancher. Une fois tranchées, elles deviendront des
étapes du plan (avec une ADR chacune).

> **Décision du 20/10 — où s'exécute le code.** Le calcul et le code utilisateur s'exécutent **toujours côté
> serveur** (le serveur de l'instance, ou la machine de l'utilisateur en profil `local`) ; aucun calcul dans le
> navigateur (Pyodide/WebAssembly écarté). L'**instance publique reste limitée** : pas de code utilisateur
> (scripts, composants à code, mode script des procédures), pas d'opération, stockage et calcul sous quotas. Les
> fonctions complètes passent par une instance locale ou privée. Raison : un seul environnement d'exécution à
> maintenir et à tester ; l'exécution dans le navigateur aurait demandé une couche de transport en double, une
> exécution sans threads, un cœur restreint aux paquets disponibles dans Pyodide, sans JAX ni GPU, et n'aurait de
> toute façon pas permis l'opération (pas de TCP ni de VISA dans un navigateur).

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

### Proposition (retenue le 20/10) : B + C

Combiner B et C : le bus reste la règle, et on ne l'éclate que là où les voies doivent recevoir des composants
différents. Une première proposition retenait A ; B + C donne presque le même résultat sans toucher au cœur ni au
format.

- **Le bus par défaut (B).** Un composant multi-canaux dessine ses N broches regroupées par une accolade. Une
  connexion porte le bus entier, comme aujourd'hui en Python (`ctx.connect(guide.port("out"), mmi.port("in"))`). Un MMI câblé
  en bus reste un seul fil.
- **L'éclatement explicite (C).** `Demux(N)` éclate un bus en N ports à une voie, et `Mux(N)` regroupe N voies en
  un bus. Ce sont deux éléments purs du cœur, utilisables aussi en Python et par le MCP, et utiles ailleurs (par
  exemple traiter séparément les bras d'un `TelescopeArray`, ou envoyer une partie des canaux d'un bus vers une
  autre puce). Ils ne découpent que l'axe des voies spatiales (voir la question 3).
- **Ce que fait `Mux`.** C'est là qu'est la seule logique nouvelle : assembler les voies dans l'ordre des ports,
  vérifier qu'elles sont compatibles (grille, longueurs d'onde) et compléter une voie non connectée par un canal
  vide (amplitude nulle), avec un avertissement à la validation.
- **Pas de mélange.** Un port reçoit soit un bus, soit la sortie d'un `Mux` ; il n'y a rien à fusionner dans le
  moteur.
- **Le format ne change pas.** Les documents `.lumos` restent en 1.2 : `Demux` et `Mux` sont des éléments
  ordinaires. Les réglages `channels` restent valables.

Deux étapes :

1. **B + C explicites.** On dessine les broches en accolade, on ajoute les éléments `Demux` / `Mux` (cœur,
   palette, docs), et on teste `Mux` sur ses invariants (aller-retour `Mux ∘ Demux` = identité, conservation de
   l'énergie). Effort : 1,5 à 2 jours.
2. **Raccourci « Éclater le port » (optionnel).** L'éditeur insère un `Demux` (ou un `Mux` en sortie) **accolé** au
   bloc : il se déplace et se supprime avec lui, et l'accolade devient visuellement le démultiplexeur. On obtient
   le rendu de A (un fil par guide d'onde) ; le document contient toujours des `Demux` / `Mux` explicites, donc le
   Python et le MCP voient exactement ce qui est dans le fichier. Effort : environ 1 jour.

En Python, le bus reste la règle et le câblage reste compact. Ce n'est verbeux que si l'utilisateur câble voie par
voie : il crée alors lui-même `Demux` et `Mux`, alors que l'option A aurait permis d'écrire `mmi.port("in")[2]`.

### Questions

1. ~~Mélange d'un bus et de voies isolées sur un même port~~ : sans objet avec B + C (un port reçoit soit un bus,
   soit la sortie d'un `Mux`).
2. ~~Voie non connectée~~ : canal vide (amplitude nulle) avec un avertissement à la validation.
3. ~~Extension aux autres signaux à plusieurs composantes~~ (tranché le 20/10). Un bus regroupe des signaux
   **séparés dans l'espace**, sur lesquels on peut placer des composants différents : guides d'onde d'une puce,
   bras d'un interféromètre. `Demux` / `Mux` découpent donc **uniquement l'axe des voies spatiales** ; chaque voie
   garde toutes ses sources et toutes ses longueurs d'onde.
   - **Composantes spectrales** : pas de bus. Séparer les longueurs d'onde est le rôle d'un composant physique
     (prisme, réseau, filtre dichroïque), pas d'un démultiplexeur logique.
   - **Composantes incohérentes** (sources d'un `IncoherentBundle`, axe des sources d'un `ModalField`) : pas de
     bus non plus. Elles se superposent dans le même faisceau, et aucun composant ne peut agir sur l'une sans
     agir sur les autres.
   - **Sortie d'un `TelescopeArray`** : c'est un bus. Elle est déjà un `ModalField` dont les voies sont les
     télescopes (`T1`…`Tn`). Le bloc dessine une broche par télescope, et un `Demux` permet de traiter chaque bras
     séparément (ligne à retard, fibre, atténuateur) avant de tout regrouper avec un `Mux` vers le recombineur.
   - **Plus tard** : un bus de `Wavefront` (un front d'onde par ouverture, en plan pupille) servirait aux
     recombinaisons de type Fizeau ou à une optique adaptative par bras. Aucun élément ne produit encore ce
     signal ; à reprendre quand un cas d'usage apparaîtra.

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

1. **Code seulement sur une instance locale ou privée** (profil `local`, ou instance privée dont l'administrateur
   l'autorise explicitement, comme `operation_enabled`) ; sur l'instance publique, seuls les composants
   **déclaratifs** et ceux installés par l'administrateur. Simple, sûr. **Retenu (décision du 20/10).**
2. ~~Code exécuté **dans le navigateur** (Pyodide/WebAssembly)~~ : **écarté** le 20/10 (voir la décision en tête
   du document).
3. ~~Bac à sable serveur~~ (processus isolé : nsjail/gVisor, sans réseau, quotas) : **écarté**, lourd à maintenir
   et à auditer (P15.4).

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

1. ~~D'accord pour **pas de code utilisateur sur l'instance publique** (option 1) ?~~ **Tranché le 20/10 : oui.**
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
actions, traitements) : comme pour les composants, il ne revient pas vers les blocs, et il ne s'exécute que sur
une instance locale ou privée (jamais sur l'instance publique). L'éditeur peut proposer « Exporter en Python » (blocs → script), pas l'inverse.

### Effort

Format et moteur d'exécution (service, MCP, API web) : ~1 semaine ; éditeur de blocs : ~1 semaine (bibliothèque
Blockly, licence Apache-2.0, ou éditeur maison en liste) ; doc, préréglage « dark puis mesure », notebook.

### Questions

1. Blocs empilés (Blockly) ou liste d'étapes éditable (plus simple, moins ludique) ?
2. Les résultats nommés doivent-ils pouvoir alimenter les **études** (ex. un Monte-Carlo sur une procédure
   entière) ?
3. ~~Le mode script sur l'instance publique : même réponse que pour les composants (non) ?~~ **Tranché le 20/10 :
   non**, il s'exécute côté serveur sur une instance locale ou privée seulement ; l'instance publique n'a que les
   blocs.

---

## Où les placer

Ces trois chantiers ajoutent des fonctions ; aucun n'est nécessaire à la v1.0. Je propose de finir la phase 15
(v1.0), puis d'en faire une **phase 18 « Extensibilité »** : ports à voies (format 1.3), composants déclaratifs et
catalogue, procédures. L'ordre recommandé : procédures (forte valeur pour les bancs réels), ports à voies,
composants.
