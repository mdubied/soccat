You will be given a batch of French or German news sentences. Decide, independently for each sentence, whether it mentions a social group. You follow a discursive definition: code how the text constructs categories of people in public discourse, regardless of whether those people share an internal identity, cohesion, or self-identification (van Dijk 1991, 2008; Schneider and Ingram 1993; Wodak et al. 2009). This is a sentence-level yes/no task. You do not extract, highlight, or categorise anything; you only judge whether each sentence as a whole mentions at least one social group.

# Language
Only French and German sentences are annotated. If a sentence is in English, do not annotate it: treat it as not mentioning a social group.

# What counts as a social group
A category of people sharing one or more characteristics that make them recognisable as a group in discourse. To count, the reference must satisfy all three:
- People, not things. It must refer to persons, not to an institution, an abstraction, or a non-human entity. "die Lehrkräfte" / "les enseignants" (people) counts; "die Schule" / "l'école" (institution) does not. "die Deutschen" / "les Français" counts; "Deutschland" / "la France" / "notre pays" does not.
- Socially meaningful trait. The shared characteristic must be socially, politically, or identity-relevant, not a chance co-presence in time or space. "Die Passanten, die an diesem Tag anwesend waren" / "Les passants présents ce jour-là" is a factual co-occurrence and does not count.
- Group, not individuals. It must refer to the group as a whole, not to isolated cases. "die Beamten" / "les fonctionnaires" counts; "einige Beamte" / "des fonctionnaires" (isolated individuals) and named individuals ("Die Minister Habeck und Pistorius" / "Les ministres Attal et Darmanin") do not.

# French "des" and contracted articles (French only)
In French, "des"/"aux"/"du" may be a contracted definite article (de/à + les/le, pointing to the group as a whole) or an indefinite article (pointing to particular individuals). Test: if it can be replaced by "les", it is a group; if by "certain·e·s", it is individuals.
- Contracted definite -> counts: "les droits des femmes" (des = de + les); "le paiement des employés"; "l'expérience des chasseurs"; "la résolution demande aux responsables politiques...".
- Indefinite -> does not count: "Des femmes portent encore cette collection" (des = certaines femmes); "des migrants prient à l'écart" (particular individuals).

# Forms that still count as a group mention
A group can be named in forms other than a plain plural noun phrase. The following still count:
- Paraphrase pronouns and generalising relatives: French "ceux qui..." / "celles qui..." as an autonomous designation, and "quiconque...". German generalising "wer" / "jemand" with a general sense.
- Group-referring adjectives, when the adjective can be paraphrased as des/[group] + "von"/genitive: "les violences policières" = par des policiers; "la mobilisation féminine" = des femmes; "la condition ouvrière" = des ouvriers; "polizeiliche Gewalt" = Gewalt durch Polizisten; "die weibliche Mobilisierung" = der Frauen.
- General singular forms: "les droits de l'enfant" / "die Rechte des Kindes"; "l'immigré comme sujet de droit" / "den Einwanderer als Rechtssubjekt".
- Groups inside German compound nouns (German only), when the compound can be reformulated so the group stands alone: "Lehrerberuf" = Beruf der Lehrer; "Kinderkrankheiten" = Krankheiten der Kinder.

# The "de la droite" versus "de droite" contrast (French only)
A definite construction naming the grouping counts; a bare qualifying adjective does not, unless the head noun is itself a group. "la candidate de la droite" counts; "la candidate de droite" does not; "les candidats de droite" counts via the head "les candidats".

# Groups versus individuals
- Individual cases do not count. "Sieben Polen kommen an" / "Sept Polonais arrivent" refers to individuals; "Die Polen sind zur Wahl aufgerufen" / "Les Polonais sont appelés aux urnes" refers to a national category and counts.
- A subgroup counts when it represents a significant portion of a broader group, says something about the broader group, or is defined by a socially or politically meaningful trait ("une large majorité de Français"; "hunderte Frauen"; "eine Gemeinschaft von 120.000 Studierenden"; "ein Teil (14 %) der Bayrou-Wählerschaft").
- A subgroup does not count when introduced by vague determiners (einige, manche, eine Handvoll / quelques, certains, une poignée de), when it describes a one-off or purely contextual situation, or when the trait is not socially relevant or too idiosyncratic ("une poignée de manifestants cagoulés"; "einige Streikende"; "Sechs Ärzte untersuchten den Angeklagten"; "300 combattants évacuent Azovstal"; "il a de nombreuses amantes").
- In doubt, ask: is this a representative subgroup that says something about the broader group, and is it defined by a stable social trait rather than a chance commonality?

# Collective actors
- Institutions do not count (die Schule, die Justiz, das Parlament, die Behörden / l'école, la justice, le Parlement, les autorités; international organisations such as UNO/ONU, IWF/FMI, EU/UE, OECD/OCDE, NATO/OTAN). People associated with an institution do count when named as people ("die Minister", "die Soldaten", "die Abgeordneten", "die Richter" / "les ministres", "les soldats", "les députés", "les juges"). Test: people or abstract institution. "Polizisten" / "les policiers" counts; "eine Reform der Polizei" / "une réforme de la police" does not.
- Political parties always count, including internal bodies (die Parteiführung, der linke Flügel / la direction, l'aile gauche), electoral lists, parliamentary factions, foreign parties, and historical or defunct parties (Likud/Likoud, NSDAP). A party abbreviation tagging a person counts ("Olaf Scholz (SPD)" / "Anne Hidalgo (PS)"). A single leader named alone (der Generalsekretär / le secrétaire national) is an individual and does not count.
- Movements count, including political movements and paramilitary or terrorist groups (der Islamische Staat / l'État islamique). Electorates of a party or candidate count ("die Wählerschaft von Éric Zemmour" / "l'électorat d'Éric Zemmour"; "die linken Wählerinnen" / "les électeurs de droite").
- The government does not count (die Regierung / le gouvernement, and coalition names such as "die Ampel", "Schwarz-gelb"), unless its members are referred to as people. But "die Mehrheit / die Opposition / die Mitte / die Rechte / die Linke" and "la majorité / l'opposition / le centre / la droite / la gauche" count as political groupings of people. Ideologies (der Kommunismus, der Faschismus / le communisme, le fascisme) do not count; their adherents do.
- Companies and associations: a general type counts (Banken, Unternehmen der Baubranche / les banques, les entreprises du bâtiment, les restaurateurs, les associations écologistes); a single named company or NGO does not (Aldi, SNCF, PSG, Greenpeace, Secours populaire), unless it is founded and carried by a social collective to represent its interests (die Gewerkschaft / le syndicat, der Verbraucherverband; "association de défense des droits des femmes").
- Religious groups and their representative federations count (CFCM, CRIF, Consistoire, Fédération protestante). Abstract belief systems do not (der Islam / l'islam, das Christentum / le christianisme).

# Recurring difficult cases
- People in collective mobilisations (demonstrators, strikers, blockers) count even in small numbers, as long as the text frames them as a collective actor.
- Fatalities (accidents, disasters, attacks) do not count as such, but a collective they belonged to before death can count ("X pompiers sont morts dans l'attentat" / "X Feuerwehrleute kamen bei dem Anschlag ums Leben").
- In survey statements, the represented group is what matters, not the polled sample.
- Individual sports teams ("die Bayern") and athletes count as a professional group.

# Examples
Input (DE): "Väter erhalten ein Recht auf Elternurlaub."
Output: {"has_social_category": true}
Input (DE): "Es geht um die sexualisierte Gewalt von Priestern gegen Kinder und Jugendliche."
Output: {"has_social_category": true}
Input (DE): "Alle Sechstklässler erhalten ein Exemplar des Romans."
Output: {"has_social_category": true}
Input (DE): "Die zahlreichen Investoren wurden in dieses Dossier eingebunden."
Output: {"has_social_category": true}
Input (DE): "Er spricht mit denjenigen, die von dem Problem betroffen sind."
Output: {"has_social_category": true}
Input (DE): "Schließlich können wir als stabile Deutsche überhaupt keine Antisemiten sein."
Output: {"has_social_category": true}
Input (DE): "Wer früh aufsteht, hat mehr vom Leben."
Output: {"has_social_category": true}
Input (DE): "Wer auch immer einreist, ohne die Formalitäten erfüllt zu haben, wird sanktioniert."
Output: {"has_social_category": true}
Input (DE): "Die weibliche Mobilisierung nahm in der Post-MeToo-Ära zu."
Output: {"has_social_category": true}
Input (DE): "Die Polen sind zur Wahl aufgerufen."
Output: {"has_social_category": true}
Input (DE): "Ein Teil (14 %) der Bayrou-Wählerschaft ist zur Stimmenthaltung geneigt."
Output: {"has_social_category": true}
Input (DE): "In Bordeaux demonstrierten nach Angaben der Polizei weniger als 200 Personen unter sengender Sonne im historischen Zentrum der Stadt."
Output: {"has_social_category": true}
Input (DE): "26,3 % der Männer zwischen 20 und 39 Jahren geben an, dass sie keine Nachkommen haben wollen."
Output: {"has_social_category": true}
Input (DE): "Die Minister erhalten eine Erhöhung ihrer Aufwandsentschädigung."
Output: {"has_social_category": true}
Input (DE): "Olaf Scholz (SPD) hat geantwortet."
Output: {"has_social_category": true}
Input (DE): "Väter zücken ihre Videokameras, Omas kramen nach Taschentüchern, irgendwo kreischt ein Kleinkind."
Output: {"has_social_category": false}
Input (DE): "Die Minister Robert Habeck und Boris Pistorius nahmen an der Sitzung teil."
Output: {"has_social_category": false}
Input (DE): "Einige Homöopathen haben einen Aufruf unterschrieben, um die Erstattung bestimmter Behandlungen zu fordern."
Output: {"has_social_category": false}
Input (DE): "Sieben Polen kommen an."
Output: {"has_social_category": false}
Input (DE): "Einige Streikende befürchten, mittellos zu werden."
Output: {"has_social_category": false}
Input (DE): "Er hat zahlreiche Geliebte."
Output: {"has_social_category": false}
Input (DE): "In der Ukraine verlassen 300 Kämpfer das Asow-Stahlwerk in Mariupol."
Output: {"has_social_category": false}
Input (DE): "Sechs Ärzte untersuchten den Angeklagten."
Output: {"has_social_category": false}
Input (DE): "Die Regierung kommt zusammen, um über den Gesetzesentwurf zu beraten."
Output: {"has_social_category": false}
Input (DE): "Sie werfen der Regierung vor, nicht ausreichend auf Übergriffe gegen den Islam zu achten."
Output: {"has_social_category": false}
Input (DE): "Die Zahl soll mindestens 35 Todesopfer betragen."
Output: {"has_social_category": false}
Input (FR): "Tous les électeurs de gauche se sont mobilisés."
Output: {"has_social_category": true}
Input (FR): "Les nombreux investisseurs ont été mobilisés sur ce dossier."
Output: {"has_social_category": true}
Input (FR): "La mesure concerne tous les délinquants ou criminels répondant d'une infraction passible de trois ans ou plus d'emprisonnement."
Output: {"has_social_category": true}
Input (FR): "Nicolas Sarkozy s'adresse dans sa campagne aux Français qui se lèvent tôt."
Output: {"has_social_category": true}
Input (FR): "Il y a les ouvriers qui arrivent et ceux qui repartent."
Output: {"has_social_category": true}
Input (FR): "Ces dispositions visent quiconque s'installe dans le pays sans avoir effectué les formalités."
Output: {"has_social_category": true}
Input (FR): "La condition ouvrière est au cœur du discours du parti."
Output: {"has_social_category": true}
Input (FR): "Les Polonais sont appelés aux urnes."
Output: {"has_social_category": true}
Input (FR): "Une large majorité de Français estiment être mieux représentés."
Output: {"has_social_category": true}
Input (FR): "À Bordeaux, moins de 200 personnes ont manifesté, selon la police, sous un soleil de plomb dans le centre historique de la ville."
Output: {"has_social_category": true}
Input (FR): "26,3% des hommes âgés de 20 à 39 ans disent ne pas vouloir de descendance."
Output: {"has_social_category": true}
Input (FR): "Les ministres bénéficient d'une augmentation de leur indemnité."
Output: {"has_social_category": true}
Input (FR): "Anne Hidalgo (PS) s'est prononcée en faveur."
Output: {"has_social_category": true}
Input (FR): "La droite proteste contre le projet de loi."
Output: {"has_social_category": true}
Input (FR): "Le conseil français du culte musulman adopte un nouveau mode de gouvernance."
Output: {"has_social_category": true}
Input (FR): "Des homéopathes ont signé une tribune pour demander le remboursement de certains traitements."
Output: {"has_social_category": false}
Input (FR): "Les ministres Gabriel Attal et Gerald Darmanin ont participé à la réunion."
Output: {"has_social_category": false}
Input (FR): "Des femmes portent encore cette collection aujourd'hui."
Output: {"has_social_category": false}
Input (FR): "Tandis que des migrants prient ou se reposent à l'écart, on dispose sur une table de pique-nique sandwichs, fruits, gâteaux."
Output: {"has_social_category": false}
Input (FR): "Sept Polonais arrivent."
Output: {"has_social_category": false}
Input (FR): "Certains grévistes craignent de se retrouver sur la paille."
Output: {"has_social_category": false}
Input (FR): "Il a de nombreuses amantes."
Output: {"has_social_category": false}
Input (FR): "300 combattants évacuent Azovstal."
Output: {"has_social_category": false}
Input (FR): "Six médecins ont examiné l'accusé."
Output: {"has_social_category": false}
Input (FR): "Le gouvernement se réunit pour délibérer du projet de loi."
Output: {"has_social_category": false}
Input (FR): "Elles reprochent au gouvernement de ne pas être suffisamment attentif aux atteintes contre l'islam."
Output: {"has_social_category": false}