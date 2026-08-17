You will be given a batch of French or German news sentences, each already judged to contain at least one social-group mention. Assign every social category each sentence expresses, independently. You follow a discursive definition: code how the text constructs categories of people, not whether they self-identify. This is a sentence-level task: you do not extract or highlight text, only decide which categories each sentence as a whole expresses.

# Multiple labels and intersectionality
List every distinct category the sentence expresses; sentences often express several, and each gets its own entry with no upper limit. A single group at the intersection of two dimensions takes both (e.g. a sentence naming both same-sex and heterosexual couples -> LGBTQIA+ and cisgender and heterosexuals). Report each distinct category once even if mentioned twice.
# Collective actors (which label applies)
- Institutions themselves are not categorised, but the people within them take their label: "les ministres" / "die Minister" -> politicians and high-ranking officials; "les policiers" / "die Polizisten" -> security forces; "les enseignants" / "die Lehrkräfte" -> teachers and educators. The institution itself ("la police" / "die Polizei", "le gouvernement" / "die Regierung") is not categorised.
- Parties, movements, electorates, "the opposition" / "la droite" / "die Rechte" -> politicians and high-ranking officials, or the group the movement represents.
- A general sector or type of enterprise -> entrepreneurs ("les restaurateurs" / "die Gastronomie", "les banques" / "die Banken"). A single named firm is not categorised.
- Religious groups -> Christians / Jews / Muslims (or multiple/other religious or minority groups). Abstract systems ("l'islam" / "der Islam") are not categorised.
- Ideologies are not categorised; their adherents take the relevant group label.

# Disambiguation defaults (set to your annotators' convention if it differs)
- Nationality or broad/local origin with no other trait ("les Russes" / "die Russen") -> Others. A nationality tied to a listed trait takes that trait's label ("Français sans abri" / "wohnungslose Franzosen" -> homeless).
- Immigrants vs ethnic and racial minorities vs Muslims: code the one explicitly named; combine only when the sentence marks more than one.
- A specific profession that is also a civil servant (police officer, state teacher, soldier) -> the specific profession (security forces, teachers and educators, soldiers), not civil servants. Use civil servants only when civil-service status itself is the point.

# Taxonomy (broad category -> specific labels). Use labels exactly as written, in English, and never invent a label. For broad class "Others", set specific_category to "others".
1. Socio-economic position: lower class; middle class; upper class; capital owners, investors and shareholders; unskilled or unqualified; skilled or qualified
2. Labor market position: wage and salary earners; civil servants; CEOs and corporate leaders; employers; entrepreneurs; self-employed and freelancers; unemployed; retirees; housewives and househusbands
3. Age and family status: parents and families; minors, including children and pupils; youth, including students and apprentices; middle-aged and pre-retirement age groups; elderly; couples; singles
4. Identities and minority/majority status: men; women; cisgender and heterosexuals; LGBTQIA+; disabled people; people with an immigration background, including immigrants; ethnic and racial minorities; Christians; Jews; Muslims; multiple (or other) religious or minority groups
5. Profession: athletes; authors and artists; doctors; farmers and fishermen; health and care professionals; journalists; legal professionals; politicians and high-ranking officials; sex workers; scientists and professors; security forces; soldiers; teachers and educators; other professions
6. Social roles and behavior: consumers and clients; car drivers; patients
7. Social deviance: extremists; terrorists, rebels, revolutionaries and/or movements of armed resistance; offenders, criminals, prisoners and/or accused people; drug addicts
8. Real estate ownership: real-estate owners; tenants; homeless
9. Others: others

# Veto
If, on inspection, the sentence contains no codable social group after all, return an empty list. has_social_category is true if and only if categories is non-empty.

# Examples
Input (DE): "Sozialpädagogin Werth hat 1993 die erste Tafel für Notleidende gegründet."
Output: {"has_social_category": true, "categories": [{"broad_category": "Socio-economic position", "specific_category": "lower class"}]}

Input (DE): "Die neue Herausforderung heißt: Menschenrechte für die Armen."
Output: {"has_social_category": true, "categories": [{"broad_category": "Socio-economic position", "specific_category": "lower class"}]}

Input (FR): "Nous accompagnons aussi un public auparavant invisible, des allocataires des minima sociaux qui n'étaient pris en charge par aucune structure."
Output: {"has_social_category": true, "categories": [{"broad_category": "Socio-economic position", "specific_category": "lower class"}]}

Input (FR): "Sur le plan national, où s'accroît le nombre de Français sans abri, victimes du chômage, de la pauvreté, de l'exclusion."
Output: {"has_social_category": true, "categories": [{"broad_category": "Real estate ownership", "specific_category": "homeless"}, {"broad_category": "Socio-economic position", "specific_category": "lower class"}]}

Input (DE): "So wird der von Arbeitgebern paritätisch mitfinanzierte Krankenkassenbeitrag langfristig ganz zur Disposition gestellt."
Output: {"has_social_category": true, "categories": [{"broad_category": "Labor market position", "specific_category": "employers"}]}

Input (DE): "Um die Arbeitszeitverkürzung abzuwehren, drohten die Metallarbeitgeber mit bundesweiter Aussperrung."
Output: {"has_social_category": true, "categories": [{"broad_category": "Labor market position", "specific_category": "employers"}]}

Input (FR): "Le gouvernement devait rencontrer les représentants des organisations officielles patronales, agricoles et syndicales, afin d'établir avec eux un pacte social."
Output: {"has_social_category": true, "categories": [{"broad_category": "Labor market position", "specific_category": "employers"}, {"broad_category": "Profession", "specific_category": "farmers and fishermen"}, {"broad_category": "Labor market position", "specific_category": "wage and salary earners"}]}

Input (FR): "Mais on n'y parviendra que si employeurs et salariés parlent entre eux."
Output: {"has_social_category": true, "categories": [{"broad_category": "Labor market position", "specific_category": "employers"}, {"broad_category": "Labor market position", "specific_category": "wage and salary earners"}]}

Input (DE): "Unter den 16- bis 24-Jährigen dagegen waren es im Juni 21,3 Prozent."
Output: {"has_social_category": true, "categories": [{"broad_category": "Age and family status", "specific_category": "youth, including students and apprentices"}]}

Input (DE): "Man müsse die Jugendlichen von der Straße fischen, ihnen eine Chance geben."
Output: {"has_social_category": true, "categories": [{"broad_category": "Age and family status", "specific_category": "youth, including students and apprentices"}]}

Input (FR): "Mais on y croisait aussi des familles, des lycéens, des étudiants."
Output: {"has_social_category": true, "categories": [{"broad_category": "Age and family status", "specific_category": "parents and families"}, {"broad_category": "Age and family status", "specific_category": "youth, including students and apprentices"}]}

Input (FR): "Les jeunes ont tenu assez longtemps le haut du pavé, place aux vieux !"
Output: {"has_social_category": true, "categories": [{"broad_category": "Age and family status", "specific_category": "youth, including students and apprentices"}, {"broad_category": "Age and family status", "specific_category": "elderly"}]}

Input (DE): "Er nutzte den Widerstand gegen die Entscheidung, die Homo-Ehe und die Adoption von Kindern durch gleichgeschlechtliche Paare zuzulassen, für seine Zwecke aus."
Output: {"has_social_category": true, "categories": [{"broad_category": "Identities and minority/majority status", "specific_category": "LGBTQIA+"}]}

Input (DE): "Wohl zum ersten Mal fand sich ein Podium von lesbischen, schwulen und transsexuellen Kulturschaffenden zusammen, die über ihre gesellschaftliche Außenseiterrolle berichteten."
Output: {"has_social_category": true, "categories": [{"broad_category": "Identities and minority/majority status", "specific_category": "LGBTQIA+"}, {"broad_category": "Profession", "specific_category": "authors and artists"}]}

Input (DE): "Bei der Trauerfeier wehten Regenbogen- und Transgenderfahnen; sogar die Polizei in Münster betonte ihre Solidarität mit der queeren Community."
Output: {"has_social_category": true, "categories": [{"broad_category": "Identities and minority/majority status", "specific_category": "LGBTQIA+"}]}

Input (FR): "La plus haute juridiction judiciaire ne reconnaît pas les mêmes droits aux enfants élevés par deux parents du même sexe et aux enfants issus d'un couple hétérosexuel."
Output: {"has_social_category": true, "categories": [{"broad_category": "Identities and minority/majority status", "specific_category": "LGBTQIA+"}, {"broad_category": "Identities and minority/majority status", "specific_category": "cisgender and heterosexuals"}]}

Input (FR): "La décision prise par la SNCF d'accorder aux couples du même sexe les réductions consenties aux couples hétérosexuels est également un succès."
Output: {"has_social_category": true, "categories": [{"broad_category": "Identities and minority/majority status", "specific_category": "LGBTQIA+"}, {"broad_category": "Identities and minority/majority status", "specific_category": "cisgender and heterosexuals"}]}

Input (DE): "Die europäischen Ausbilder haben bislang rund 8000 einheimische Soldaten in Grundfertigkeiten ausgebildet."
Output: {"has_social_category": true, "categories": [{"broad_category": "Profession", "specific_category": "soldiers"}]}

Input (DE): "So wurden Panzerfallen eingerichtet und die Kämpfer mit Schulterraketen ausgestattet, mit denen Panzer und Flugobjekte getroffen werden können."
Output: {"has_social_category": true, "categories": [{"broad_category": "Profession", "specific_category": "soldiers"}]}

Input (FR): "Jacques Chirac poursuivait à l'Ecole nationale de gendarmerie de Melun, devant de futurs officiers et sous-officiers, sa visite des différents corps de l'armée."
Output: {"has_social_category": true, "categories": [{"broad_category": "Profession", "specific_category": "soldiers"}]}

Input (FR): "Réservons également une pensée respectueuse et fraternelle aux familles des passagers en deuil et aux soldats blessés au combat, qui souffrent dans leur coeur et dans leur chair."
Output: {"has_social_category": true, "categories": [{"broad_category": "Profession", "specific_category": "soldiers"}, {"broad_category": "Age and family status", "specific_category": "parents and families"}]}

Input (DE): "Wochenlang hielten die Autofahrer mit ihrer Aktion gegen Shell die Menschen in Atem."
Output: {"has_social_category": true, "categories": [{"broad_category": "Social roles and behavior", "specific_category": "car drivers"}]}

Input (FR): "Tous les possesseurs de voiture sportive le déplorent."
Output: {"has_social_category": true, "categories": [{"broad_category": "Social roles and behavior", "specific_category": "car drivers"}]}

Input (FR): "Les automobilistes profiteront ce matin d'une baisse de 6 à 7 centimes."
Output: {"has_social_category": true, "categories": [{"broad_category": "Social roles and behavior", "specific_category": "car drivers"}]}

Input (DE): "Drogenkranke in allen Stadien des Verfalls, das sind täglich Tausende."
Output: {"has_social_category": true, "categories": [{"broad_category": "Social deviance", "specific_category": "drug addicts"}]}

Input (FR): "Les toxicomanes peuvent ainsi, vingt-quatre heures sur vingt-quatre, déposer leurs seringues usagées."
Output: {"has_social_category": true, "categories": [{"broad_category": "Social deviance", "specific_category": "drug addicts"}]}

Input (FR): "Comme avec de jeunes drogués, on a plutôt le sentiment d'une énergie intraitable."
Output: {"has_social_category": true, "categories": [{"broad_category": "Social deviance", "specific_category": "drug addicts"}, {"broad_category": "Age and family status", "specific_category": "youth, including students and apprentices"}]}

Input (DE): "Entlang der Gleise sind jetzt immer wieder Müllansammlungen oder heruntergekommene Zelte von Obdachlosen zu sehen."
Output: {"has_social_category": true, "categories": [{"broad_category": "Real estate ownership", "specific_category": "homeless"}]}

Input (DE): "Die Anzahl der Menschen ohne festen Wohnsitz hat sich in den letzten zwei Jahren um 25 Prozent erhöht."
Output: {"has_social_category": true, "categories": [{"broad_category": "Real estate ownership", "specific_category": "homeless"}]}