You will be given a batch of French or German news sentences, each already judged to mention a social group. Assign every category each sentence expresses, independently. Use a discursive definition: code how the text constructs categories of people, not whether they self-identify. Sentence-level: decide which categories are present, do not extract text.

Intersectionality: list every distinct category; a group at an intersection takes both (a sentence naming same-sex and heterosexual couples -> LGBTQIA+ and cisgender and heterosexuals). Report each once.

Collective actors: institutions themselves are not categorised, but people within them take their label ("les ministres"/"die Minister" -> politicians and high-ranking officials; "les policiers"/"die Polizisten" -> security forces; "les enseignants"/"die Lehrkräfte" -> teachers and educators). Parties, movements, electorates, "the opposition"/"la droite"/"die Rechte" -> politicians and high-ranking officials or the represented group. A general sector -> entrepreneurs; a single named firm is not categorised. Religious groups -> Christians/Jews/Muslims; abstract systems ("l'islam") not categorised. Ideologies not categorised; their adherents take the group label.

Disambiguation defaults (adjust to your gold): nationality/origin alone -> Others; nationality + trait -> the trait ("Français sans abri" -> homeless). Immigrants vs ethnic and racial minorities vs Muslims -> the one named, combine only when more than one is marked. A specific profession that is also a civil servant (police, state teacher, soldier) -> the profession, not civil servants.

Taxonomy (broad -> specific). Use labels verbatim in English; never invent one. For "Others", specific is "others".
1. Socio-economic position: lower class; middle class; upper class; capital owners, investors and shareholders; unskilled or unqualified; skilled or qualified
2. Labor market position: wage and salary earners; civil servants; CEOs and corporate leaders; employers; entrepreneurs; self-employed and freelancers; unemployed; retirees; housewives and househusbands
3. Age and family status: parents and families; minors, including children and pupils; youth, including students and apprentices; middle-aged and pre-retirement age groups; elderly; couples; singles
4. Identities and minority/majority status: men; women; cisgender and heterosexuals; LGBTQIA+; disabled people; people with an immigration background, including immigrants; ethnic and racial minorities; Christians; Jews; Muslims; multiple (or other) religious or minority groups
5. Profession: athletes; authors and artists; doctors; farmers and fishermen; health and care professionals; journalists; legal professionals; politicians and high-ranking officials; sex workers; scientists and professors; security forces; soldiers; teachers and educators; other professions
6. Social roles and behavior: consumers and clients; car drivers; patients
7. Social deviance: extremists; terrorists, rebels, revolutionaries and/or movements of armed resistance; offenders, criminals, prisoners and/or accused people; drug addicts
8. Real estate ownership: real-estate owners; tenants; homeless
9. Others: others

Veto: if on inspection there is no codable group, categories is empty. has_social_category is true iff categories is non-empty.

Examples:
Input (DE): "So wird der von Arbeitgebern paritätisch mitfinanzierte Krankenkassenbeitrag langfristig ganz zur Disposition gestellt."
Output: {"has_social_category": true, "categories": [{"broad_category": "Labor market position", "specific_category": "employers"}]}
Input (FR): "Mais on n'y parviendra que si employeurs et salariés parlent entre eux."
Output: {"has_social_category": true, "categories": [{"broad_category": "Labor market position", "specific_category": "employers"}, {"broad_category": "Labor market position", "specific_category": "wage and salary earners"}]}