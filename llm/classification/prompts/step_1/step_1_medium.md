You will be given a batch of French or German news sentences. Decide, independently for each sentence, whether it mentions a social group, using a discursive definition: code how the text constructs categories of people, not whether they self-identify. Sentence-level yes/no only; do not extract or categorise.

A social group is a category of people sharing a socially, politically, or identity-relevant trait, referred to as a group. All three must hold: people, not an institution or abstraction ("les enseignants" / "die Lehrkräfte" yes, "l'école" / "die Schule" no; "les Français" / "die Deutschen" yes, "la France" / "Deutschland" no); a meaningful trait, not a chance co-presence; the group as a whole, not isolated individuals ("les fonctionnaires" / "die Beamten" yes, "des fonctionnaires" / "einige Beamte" and named individuals no).

French "des"/"aux"/"du": a contracted definite article (= les) marks a group; an indefinite article (= certain·e·s) does not. "les droits des femmes" yes; "des femmes portent cette collection" no; "des migrants prient à l'écart" no.

Count non-plural forms too: French "ceux qui..." / "quiconque...", German "wer" / "jemand" with general sense; group-referring adjectives paraphrasable as des/[group]+"von"/genitive ("violences policières" = par des policiers; "polizeiliche Gewalt" = Gewalt durch Polizisten); general singulars ("les droits de l'enfant" / "die Rechte des Kindes"); German compounds reformulable as a standalone group ("Lehrerberuf" = Beruf der Lehrer). French only: "la candidate de la droite" yes, "la candidate de droite" no, "les candidats de droite" yes.

Subgroups: count when sizeable, representative, or defined by a meaningful trait ("une large majorité de Français" / "eine breite Mehrheit der Franzosen"); do not count when introduced by vague determiners (quelques, certains, une poignée / einige, eine Handvoll), one-off or contextual, or idiosyncratic ("six médecins ont examiné l'accusé" / "sechs Ärzte untersuchten den Angeklagten").

Collective actors:
- Institutions no (l'école, la justice, le Parlement, ONU, UE / die Schule, die Justiz, das Parlament, UNO, EU); people within them yes ("les policiers" / "die Polizisten" yes, "réforme de la police" / "Reform der Polizei" no).
- Parties always yes (internal bodies, lists, factions, foreign, historical; "(PS)"/"(SPD)" tag yes); a lone leader is an individual. Movements yes (incl. paramilitary/terrorist); electorates yes.
- Government no (incl. "die Ampel", "Schwarz-gelb") unless its members as people; but "la majorité/l'opposition/la droite/la gauche" and "die Mehrheit/Opposition/Rechte/Linke" yes. Ideologies no, their adherents yes.
- General company/association types yes (les banques, les restaurateurs / die Banken, die Baubranche); single firm or NGO no (SNCF, Aldi, Greenpeace) unless it carries a social collective's interests (le syndicat / die Gewerkschaft).
- Religious groups and federations yes (CFCM, CRIF); abstract systems no (l'islam / der Islam, le christianisme / das Christentum).

Difficult cases: demonstrators/strikers count even in small numbers if framed as a collective; fatalities not as such but their prior collective can count; in surveys code the represented group; sports teams and athletes count.

Example:
Input (FR): "Les ministres bénéficient d'une augmentation de leur indemnité."
Output: {"has_social_category": true}
Input (DE): "Die Minister erhalten eine Erhöhung ihrer Aufwandsentschädigung."
Output: {"has_social_category": true}