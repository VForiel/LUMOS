# Pistes d'évolution — retours du 19/10/2026 (tranchées le 20/10, étapes P15.6 à P15.16 du plan)

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

## 1. Bus, ports multi-canaux et injection des télescopes

### Constat

- **Rendu.** Un `ModalField` porte N canaux sur **un seul port** (`in`, `out`) : c'est un bus. Le réglage
  `channels` choisit les canaux sur lesquels agit un composant (coupleur sur 0 et 2…). C'est compact et exact pour
  le calcul, mais un MMI 4×4 dessiné avec une seule entrée et une seule sortie ne ressemble pas au composant, et
  câbler une puce oblige à raisonner en indices de canaux.
- **Modèle.** L'axe des canaux d'un `ModalField` mélange deux notions : des guides séparés dans l'espace (sortie
  d'un `TelescopeArray`, guides d'une puce) et les modes d'un même guide (port `modes` d'une `Fiber`).
- **`TelescopeArray`.** Il renvoie directement un `ModalField` : l'élément mélange la collecte en espace libre et
  l'injection dans des guides monomodes, supposée parfaite. Conséquences physiques :
  - le couplage n'est réduit que par `throughput`, alors qu'une pupille d'Airy couplée dans une fibre monomode
    plafonne vers 80 % (Shaklan & Roddier 1988, Appl. Opt. 27, 2334) ;
  - l'amplitude injectée ne dépend pas de l'angle de la source : il n'y a pas de lobe d'antenne de la fibre
    (environ λ/D), donc une source hors axe est couplée comme une source sur l'axe, ce qui fausse les cartes de
    transmission des nulleurs (Bracewell, kernel nuller, LIFE) ;
  - on ne peut rien insérer par bras avant l'injection (aberration, optique adaptative, pointage).

  Pour un seul télescope, la chaîne explicite existe déjà : `Telescope` (un `IncoherentBundle` en plan pupille)
  puis `Fiber` (intégrale de recouvrement, port `modes`).

### Options pour le rendu

| | A. Ports indexés dans le cœur | B. Ports dessinés, bus conservé | C. Blocs Démux / Mux explicites |
|---|---|---|---|
| Principe | un port peut avoir N voies : `mmi.in[2]` se connecte à `guide.out` ; le moteur assemble le bus | le bloc dessine N broches regroupées par une accolade ; une connexion porte toujours le bus entier | `Demux(N)` éclate un bus en N ports, `Mux(N)` les regroupe |
| Rendu | un fil par guide d'onde, comme un schéma de puce | lisible, mais les fils restent des bus | correct, mais chargé (2 blocs de plus par composant) |
| Cœur | connexions vers une voie, assemblage des voies (ordre, longueurs d'onde), validation | aucun changement | deux éléments de plus |
| Format `.lumos` | 1.3 (`"to": "mmi.in[2]"`) | inchangé | inchangé |
| Effort | 3 à 4 jours (cœur, service, éditeur, docs) | 1 jour | 1 jour |

### Décisions du 20/10

#### 1.1 Rendu : B + C

Le bus reste la règle, et on ne l'éclate que là où les voies doivent recevoir des composants différents. Une
première proposition retenait A ; B + C donne presque le même résultat sans ajouter de connexions vers une voie
dans le moteur.

- **Le bus par défaut (B).** Un composant multi-canaux dessine ses N broches regroupées par une accolade. Une
  connexion porte le bus entier, comme aujourd'hui en Python
  (`ctx.connect(guide.port("out"), mmi.port("in"))`). Un MMI câblé en bus reste un seul fil.
- **L'éclatement explicite (C).** `Demux(N)` éclate un bus en N ports à une voie, et `Mux(N)` regroupe N voies en
  un bus. Ce sont deux éléments purs du cœur, utilisables aussi en Python et par le MCP, et utiles ailleurs (par
  exemple traiter séparément les bras d'un `TelescopeArray`, ou envoyer une partie des voies d'un bus vers une
  autre puce).
- **`Mux` ne mélange rien.** Il range les voies côte à côte dans l'ordre de ses ports, sans somme ni
  interférence ; seul un composant physique (MMI, coupleur, lame séparatrice, recombineur) mélange des voies. Une
  voie non connectée devient une voie vide (amplitude nulle), avec un avertissement à la validation.
- **Pas de mélange de connexions.** Un port reçoit soit un bus, soit la sortie d'un `Mux`.

En Python, le bus reste la règle et le câblage reste compact. Ce n'est verbeux que si l'utilisateur câble voie par
voie : il crée alors lui-même `Demux` et `Mux`, alors que l'option A aurait permis d'écrire `mmi.port("in")[2]`.

#### 1.2 Modèle des signaux : une voie = un signal, un bus = N voies

- **Une voie guidée** est un `ModalField` d'un seul guide, de forme `(sources, λ, modes)` ; `modes = 1` pour un
  guide monomode. L'axe des modes ne désigne plus jamais des guides différents. Une lanterne photonique, par
  exemple, transforme une voie à K modes en un bus de K voies monomodes.
- **Une voie en espace libre** est un `Wavefront`, un `Wavecube` ou un `IncoherentBundle`.
- **Un bus** regroupe N voies du même type de signal, séparées dans l'espace. Il se comporte comme une séquence :
  `len(bus)`, `bus[i]`, des noms de voies (`T1`…`Tn`).
- **Un composant à une voie appliqué à un bus agit sur chaque voie**, comme N copies identiques : une `Fiber`
  branchée sur un bus de faisceaux renvoie un bus de voies guidées. Pour traiter les voies différemment, on passe
  par `Demux`.
- **Un recombineur** (MMI, coupleur, kernel nuller) applique sa matrice S au vecteur de tous les modes de toutes
  ses voies d'entrée. Pour interférer, les voies doivent décrire **les mêmes sources dans le même ordre** ; le
  recombineur le vérifie (identifiant de la liste de sources porté par chaque voie) au lieu de le supposer.

**Compactage implicite, pour tous les bus (guidés et en espace libre).** Quand les voies partagent certaines
propriétés, le bus les stocke dans **un seul tableau** ; sinon, il garde une liste de signaux. Le choix est fait
par le bus, à la construction et après chaque `Mux`, et l'interface est la même dans les deux cas : les éléments
ne voient qu'un bus de N voies.

| Bus | Conditions du compactage | Tableau |
|---|---|---|
| guidé | mêmes sources, mêmes longueurs d'onde et poids spectraux, même unité de puissance (le nombre de modes peut varier d'une voie à l'autre) | `(sources, λ, Σ modes)` et la position de chaque voie |
| espace libre | même type de signal, même grille, mêmes sources, mêmes longueurs d'onde et poids spectraux, même polarisation, même unité de puissance | `(voies, sources, λ, ny, nx)` (et l'axe de polarisation s'il existe) |

Avantages : un recombineur applique sa matrice sans copier les voies ; `Demux` / `Mux` deviennent des découpages
et des concaténations ; un composant appliqué à un bus compacté est vectorisé sur l'axe des voies (par exemple les
N intégrales de recouvrement d'une `Fiber` en un seul calcul). Pour un réseau de télescopes identiques, le bus
de sortie est toujours compacté.

**Ce qui est un bus et ce qui n'en est pas un.** Un bus regroupe des signaux **séparés dans l'espace**, sur
lesquels on peut placer des composants différents : guides d'onde d'une puce, bras d'un interféromètre.
`Demux` / `Mux` découpent donc **uniquement l'axe des voies** ; chaque voie garde toutes ses sources et toutes ses
longueurs d'onde.

- **Composantes spectrales** : pas de bus. Séparer les longueurs d'onde est le rôle d'un composant physique
  (prisme, réseau, filtre dichroïque), pas d'un démultiplexeur logique.
- **Composantes incohérentes** (sources d'un `IncoherentBundle`, axe des sources d'un `ModalField`) : pas de bus
  non plus. Elles se superposent dans le même faisceau, et aucun composant ne peut agir sur l'une sans agir sur
  les autres.
- **Modes d'un guide** : pas de bus. Ils partagent le même guide ; c'est un composant physique (lanterne
  photonique) qui les sépare en voies.

#### 1.3 `TelescopeArray` : collecte en espace libre, injection explicite

- **`TelescopeArray` renvoie un bus de faisceaux en plan pupille**, un `IncoherentBundle` par télescope : la
  pupille de chaque télescope (comme `Telescope`) avec la phase géométrique de son bras. Les télescopes étant
  identiques, le bus est compacté.
- **L'injection devient un élément du schéma** : `TelescopeArray → Fiber → puce` pour le cas courant (une fibre
  identique par bras), ou `TelescopeArray → Demux → (optique par bras, Fiber) → Mux → puce` pour traiter les bras
  séparément. Le couplage réel et le lobe d'antenne de la fibre sont alors pris en compte.
- **Le modèle analytique actuel est conservé sous un nom explicite** (proposition : `IdealInjectionArray`), comme
  modèle rapide documenté et non comme comportement par défaut. Il reste utile quand le nombre de sources est
  grand (disque exozodiacal de milliers de points), où échantillonner N pupilles × M sources × nλ coûte cher.
- **Préréglages** : `bracewell.lumos` et `kernel_nuller.lumos` branchent `TelescopeArray` directement sur la puce.
  Le convertisseur vers le format 1.3 remplace cet élément par `IdealInjectionArray`, ce qui garde les résultats
  à l'identique ; les préréglages livrés passent à la chaîne explicite.

### Étapes

Une ADR pour le modèle des signaux et des bus (1.2 et 1.3), écrite avant l'étape 2.

1. **Rendu B + C.** Broches en accolade, éléments `Demux` / `Mux` (cœur, palette, docs). Tests des invariants :
   aller-retour `Mux ∘ Demux` = identité, conservation de l'énergie, voie vide pour une entrée non connectée.
   Effort : 1,5 à 2 jours.
2. **Modèle des bus.** Voie guidée `(sources, λ, modes)`, bus guidé et bus en espace libre avec compactage
   implicite, recombineurs sur bus avec vérification des sources, composants à une voie appliqués à un bus,
   migration des éléments photoniques, de `Fiber` et des détecteurs. Format `.lumos` 1.3 avec convertisseur
   (les réglages `channels` désignent désormais des voies). Tests : un bus compacté et le même bus en liste donnent
   les mêmes résultats ; le compactage est refusé dès qu'une propriété partagée diffère. Effort : 3 à 4 jours.
3. **`TelescopeArray` en espace libre.** Bus d'`IncoherentBundle`, `IdealInjectionArray`, migration des
   préréglages et des notebooks. Tests de référence : couplage d'une pupille d'Airy dans une fibre monomode
   (Shaklan & Roddier 1988) et lobe d'antenne en fonction de l'angle de la source ; `IdealInjectionArray`
   identique à l'ancien `TelescopeArray`. Effort : 1,5 à 2 jours.
4. **Raccourci « Éclater le port » (optionnel).** L'éditeur insère un `Demux` (ou un `Mux` en sortie) **accolé** au
   bloc : il se déplace et se supprime avec lui, et l'accolade devient visuellement le démultiplexeur. On obtient
   le rendu de A (un fil par guide d'onde) ; le document contient toujours des `Demux` / `Mux` explicites, donc le
   Python et le MCP voient exactement ce qui est dans le fichier. Effort : environ 1 jour.
5. **`FizeauCombiner`.** Recombine un bus de faisceaux en plan pupille en **une seule pupille de sortie** : chaque
   voie est placée à la position de son télescope, à une échelle commune (pupille de sortie homothétique de la
   pupille d'entrée, condition de Traub 1986, J. Opt. Soc. Am. A 3, 1491, pour garder un champ étendu). La sortie
   est un `IncoherentBundle` unique, que les éléments de propagation existants amènent au plan focal. Pour cela,
   chaque voie du bus porte la position projetée de son télescope (donnée par `TelescopeArray` à chaque instant).
   Tests de référence : deux ouvertures donnent la figure d'Airy d'une ouverture modulée par des franges en
   cos²(π B θ / λ) (Born & Wolf, *Principles of Optics*, § 8.6) ; énergie conservée entre
   les voies d'entrée et la pupille de sortie. Effort : 1,5 à 2 jours.

### Questions

1. ~~Mélange d'un bus et de voies isolées sur un même port~~ : sans objet avec B + C (un port reçoit soit un bus,
   soit la sortie d'un `Mux`).
2. ~~Voie non connectée~~ : voie vide (amplitude nulle) avec un avertissement à la validation.
3. ~~Extension aux autres signaux à plusieurs composantes~~ : un bus regroupe uniquement des voies séparées dans
   l'espace, guidées ou en espace libre (voir 1.2).
4. ~~Nom du modèle rapide~~ : `IdealInjectionArray` (retenu le 20/10).
5. ~~Recombinaison de type Fizeau~~ : un élément dédié, `FizeauCombiner` (retenu le 20/10, étape 5).

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

Un **index public** (une fiche par composant, auteur, version, licence, tests passés) consultable depuis la palette
(« Catalogue »). **Décision du 20/10 :** il vit dans le dépôt LUMOS, dans un dossier `community/` réservé aux
contributions de la communauté (composants, préréglages partagés, ressources de l'instance publique), plutôt que
dans un dépôt séparé : une seule CI, une seule revue, et les contributions suivent l'évolution du format. La CI
filtre par chemin (une contribution dans `community/` ne relance que ses propres vérifications), et chaque fiche
déclare sa licence et la version minimale du format `.lumos`. Les composants déclaratifs s'importent d'un clic ; les composants à code s'installent
comme paquets Python (`pip install lumos-component-…`, nommage réservé sur PyPI) en local. Une CI de l'index
exécute `lumos component check` sur chaque contribution. Une page de doc « Créer et partager un composant »
décrit les deux voies.

### Effort

Vue simple + format `component` + validation : ~1 semaine. Catalogue (index, CI, palette) : ~1 semaine. Vue code
(local) : 3 jours. ADR à écrire (format, sécurité).

### Questions

1. ~~D'accord pour **pas de code utilisateur sur l'instance publique** (option 1) ?~~ **Tranché le 20/10 : oui.**
2. ~~Hébergement du catalogue~~ : dossier `community/` du dépôt LUMOS, contributions par pull request (tranché le
   20/10).
3. ~~Gabarits prioritaires~~ (tranché le 20/10) : matrice de transfert ; gain ou atténuation et déphasage ;
   transmission spectrale tabulée ; composite (un groupe). La réponse impulsionnelle vient plus tard.

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

1. ~~Blocs empilés ou liste d'étapes~~ : **Blockly** (tranché le 20/10).
2. ~~Résultats nommés et études~~ : **oui** (tranché le 20/10). Une procédure peut être le corps d'une étude
   (balayage, Monte-Carlo) et ses résultats nommés en sont les sorties. Les itérations indépendantes (bloc
   *pour chaque valeur* sans dépendance entre itérations, études sur une procédure) s'exécutent en parallèle sur
   les *workers* du service, dans la limite des quotas.
3. ~~Le mode script sur l'instance publique : même réponse que pour les composants (non) ?~~ **Tranché le 20/10 :
   non**, il s'exécute côté serveur sur une instance locale ou privée seulement ; l'instance publique n'a que les
   blocs.

---

## Où les placer

~~Première proposition : finir la phase 15 (v1.0), puis en faire une phase 18 « Extensibilité ».~~

**Décision du 20/10 : les trois chantiers font partie de la v1.0.** Sans eux, les cas d'usage possibles depuis
l'interface web restent trop limités pour que la démonstration (et donc le produit) convainque. Ils passent donc
**avant** la performance (P15.3), la qualité et la sécurité (P15.4) et la publication (P15.5), qui s'appliquent
ensuite à eux aussi :

- **performance** : exécution parallèle des boucles et des études de procédures sur les *workers*, vectorisation
  des bus compactés, mesure des coûts de `Fiber` et de `FizeauCombiner` ;
- **sécurité** : vérifier que les zones de code (composants à code, mode script des procédures) sont
  **inaccessibles sur l'instance publique** (service, API web, MCP, chargement de documents), que les documents
  déclaratifs (`component`, `procedure`) ne peuvent rien exécuter, et que les quotas s'appliquent aux procédures ;
- **publication** : la v1.0 publiée inclut le format 1.3, les procédures et le catalogue.

**Ordre retenu** (étapes P15.6 à P15.16 du plan, placées avant P15.3) : le modèle des bus d'abord, car il change
les signaux et le format dont dépendent les deux autres chantiers (le modèle passe aussi avant le rendu B + C, pour
ne pas écrire `Demux` / `Mux` deux fois) ; puis les procédures, qui apportent le plus à la démonstration ; enfin
les composants et le catalogue.

Le format `.lumos` 1.3 reste dans le cadre du gel de P15.1 : c'est une version mineure, et le convertisseur lit
les documents 1.x antérieurs.
