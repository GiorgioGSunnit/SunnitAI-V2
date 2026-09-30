"""Test set v2: the labelled requests, against the new 5,145-template catalog.

Starts from selection_testset.py (labelled on the old 484 catalog), translates
every answer to a catalog FILENAME (so later catalog changes cannot shift it),
adds the new exact templates that are right answers too, turns the formerly
'unrelated' requests that now have a template into 'found', and adds requests
aimed at new DEJURE templates and at documents the catalog does not have.

Writes testset_v2.json:  [{"q", "acc": [filenames], "kind": "found"|"absent"}]
and prints the top matches of every new request so each label can be checked.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate as C  # noqa: E402
import selection_testset as T  # noqa: E402

OUT = os.path.join(HERE, "testset_v2.json")

# Extra right answers for existing requests (reviewed 30 Sept 2026 on the top candidates).
ADD = {
    "Scrivi un ricorso ex art. 700 c.p.c.": ["civile__ricorso-ex-art-700-c-p-c.docx"],
    "Scrivi Ricorso ex 700 cpc": ["civile__ricorso-ex-art-700-c-p-c.docx"],
    "ricorso per decreto ingiuntivo per un credito non pagato": [
        "civile__ricorso-per-decreto-ingiuntivo-con-richiesta-di-provvisoria-esecutorieta-ex-art-642-comma-2-c-p-c-documentazione-sottoscritta-dal-debitore-comprovante-il-diritto-fatto-valere.docx"],
    "Istanza di accesso civico generalizzato FOIA": [
        "amministrativo__istanza-di-accesso-civico-generalizzato-art-5-comma-2-d-lgs-n-33-2013.docx"],
    "istanza di sospensione del procedimento con messa alla prova": [
        "penale__istanza-di-sospensione-del-procedimento-con-messa-alla-prova-art-464-bis-comma-1.docx",
        "penale__istanza-di-sospensione-del-procedimento-con-messa-alla-prova-nel-corso-delle-indagini-preliminari-artt-464-ter-e-464-ter-1.docx"],
    "Clausola arbitrale da inserire in un contratto": [
        "contratti__clausola-compromissoria-per-arbitrato-rituale-in-materia-di-appalto.docx",
        "arbitrato__clausola-compromissoria-per-arbitrato-irrituale.docx",
        "contratti__clausola-compromissoria-per-arbitrato-rituale-amministrato.docx",
        "contratti__clausola-compromissoria-per-arbitrato-rituale-con-arbitro-unico.docx",
        "arbitrato__clausola-compromissoria-per-arbitrato-rituale-con-clausola-per-la-decisione-secondo-equita.docx"],
    "domanda di ammissione al passivo della liquidazione giudiziale": [
        "concorsuale__domanda-di-ammissione-al-passivo-ai-sensi-dell-art-201-c-c-i-i.docx"],
    "Ricorso per separazione consensuale": [
        "civile__ricorso-cumulativo-per-separazione-consensuale-e-per-divorzio-a-domanda-congiunta.docx"],
    "Istanza di ammissione al patrocinio a spese dello Stato": [
        "amministrativo__istanza-di-ammissione-al-patrocinio-a-spese-dello-stato-art-76-d-p-r-n-115-2002.docx",
        "tributario__istanza-per-l-ammissione-al-patrocinio-a-spese-dello-stato.docx"],
    "Contratto di anticipazione bancaria con pegno su merci": [
        "contratti__contratto-di-anticipazione-bancaria-su-pegno-di-merci.docx"],
    "Lettera al datore di lavoro per comunicare il congedo parentale": [
        "lavoro__comunicazione-di-voler-beneficiare-del-congedo-parentale-su-base-oraria-art-32-d-lgs-n-151-2001.docx"],
    "Ricorso per ricusazione del giudice": [
        "penale__dichiarazione-di-ricusazione-del-giudice-artt-37-e-38.docx",
        "amministrativo__istanza-di-ricusazione-art-18.docx",
        "penale__istanza-di-ricusazione-art-8-d-lgs-n-159-2011.docx"],
    "Ricorso contro la cartella di pagamento dell'Agenzia delle Entrate": [
        "tributario__ricorso-reclamo-ex-art-17-bis-d-lgs-n-546-1992.docx",
        "tributario__ricorso-reclamo-con-istanza-di-mediazione-exart-17-bis-d-lgs-n-546-1992.docx",
        "tributario__ricorso-generico-con-istanza-di-sospensione-dell-atto-impugnato-inaudita-altera-parte.docx"],
    "Istanza di revoca della confisca di prevenzione": [
        "penale__istanza-di-revocazione-della-confisca-art-28-d-lgs-n-159-2011.docx"],
}
# Formerly "nothing fits": the new catalog has them now.
NOW_FOUND = {
    "Prepara un contratto di mutuo ipotecario": [
        "contratti__contratto-di-mutuo-ipotecario-a-tasso-fisso.docx",
        "contratti__contratto-di-mutuo-ipotecario-a-tasso-misto.docx",
        "contratti__contratto-di-mutuo-ipotecario-a-tasso-variabile.docx"],
    "Mi serve l'atto costitutivo di una srl": [
        "societario__atto-costitutivo-di-s-r-l.docx",
        "societario__atto-costitutivo-e-statuto-di-s-r-l-a-capitale-ridotto.docx",
        "societario__statuto-di-s-r-l.docx"],
    "Contratto di agenzia commerciale con esclusiva di zona": [
        "contratti__contratto-di-agenzia-con-clausola-di-esclusiva.docx",
        "lavoro__contratto-di-agenzia-con-esclusiva.docx"],
    "Contratto di sponsorizzazione sportiva": [
        "contratti__contratto-di-sponsorizzazione-di-squadra-sportiva.docx"],
    "Ricorso per l'adozione di un minore": [
        "civile__richiesta-di-adozione-da-parte-degli-affidatari.docx",
        "civile__dichiarazione-di-disponibilita-all-adozione.docx",
        "civile__domanda-di-adozione-da-parte-di-stranieri-o-di-cittadini-italiani-residenti-all-estero.docx"],
    "Patto parasociale di prelazione tra soci": [
        "contratti__patto-parasociale-di-blocco.docx",
        "contratti__patto-parasociale-stipulato-tra-i-soci-di-minoranza.docx",
        "contratti__patto-parasociale-per-l-esercizio-del-diritto-di-voto.docx"],
}
# Requests aimed at new DEJURE templates (random sample across areas, own wording).
NEW_TEMPLATES = [
    ("Devo notificare un precetto sulla base dell'accordo raggiunto in mediazione",
     ["arbitrato__atto-di-precetto-fondato-su-accordo-in-sede-di-mediazione.docx"]),
    ("Invito alla negoziazione assistita per recuperare il mio compenso da una società cliente",
     ["arbitrato__invito-alla-negoziazione-assistita-per-pagamento-del-compenso-spettante-all-avvocato-per-attivita-di-assistenza-giudiziale-svolta-in-favore-di-cliente-persona-giuridica-artt-2-e.docx"]),
    ("Compromesso per arbitrato rituale con un solo arbitro in una lite con la pubblica amministrazione",
     ["amministrativo__compromesso-per-arbitrato-rituale-con-arbitro-unico-artt-12-c-p-a-e-807-c-p-c.docx"]),
    ("Chiedere la fissazione dell'udienza d'appello in un contenzioso elettorale",
     ["amministrativo__istanza-di-fissazione-udienza-in-appello-nel-contenzioso-elettorale-art-129.docx"]),
    ("Ricorso contro gli atti del commissario ad acta nominato nel giudizio sul silenzio",
     ["amministrativo__ricorso-avverso-gli-atti-adottati-dal-commissario-ad-acta-nel-giudizio-avverso-il-silenzio.docx"]),
    ("Intervento di un creditore nell'espropriazione esattoriale",
     ["civile__ricorso-per-intervento-nell-espropriazione-esattoriale-artt-499-c-p-c-e-54-d-p-r-n-602-1973.docx"]),
    ("Opposizione alla rimozione dei sigilli sui beni del defunto",
     ["civile__opposizione-alla-rimozione-di-sigilli.docx"]),
    # every 72-bis opposition with a suspension request fits: the request names neither
    # the kind of credit nor the defect
    ("Opposizione al pignoramento dell'agente della riscossione ex 72-bis con richiesta di sospensione",
     ["civile__opposizione-a-pignoramento-ex-art-72-bis-d-p-r-n-602-1973-per-recupero-di-crediti-tributari-in-caso-di-vizi-derivati-del-pignoramento-con-contestuale-istanza-di-sospensione.docx",
      "civile__opposizione-a-pignoramento-ex-art-72-bis-d-p-r-n-602-1973-per-recupero-di-crediti-tributari-in-caso-di-vizi-propri-del-pignoramento-con-contestuale-istanza-di-sospensione.docx",
      "civile__opposizione-a-pignoramento-ex-art-72-bis-d-p-r-n-602-1973-per-recupero-di-crediti-extratributari-in-caso-di-vizi-derivati-del-pignoramento-con-contestuale-istanza-di-sospension.docx",
      "civile__opposizione-a-pignoramento-ex-art-72-bis-d-p-r-n-602-1973-per-recupero-di-crediti-extratributari-in-caso-di-vizi-propri-del-pignoramento-con-contestuale-istanza-di-sospensione.docx",
      "civile__opposizione-a-pignoramento-ex-art-72-bis-d-p-r-n-602-1973-per-violazione-dei-limiti-di-pignorabilita-ex-art-72-ter-d-p-r-n-602-1973-con-contestuale-istanza-di-sospensione-pa.docx"]),
    # "per saltum" is art. 569; the generic penal cassazione appeals are a usable fallback
    ("Ricorso per saltum in cassazione contro la sentenza penale di primo grado",
     ["penale__ricorso-immediato-per-cassazione-art-569.docx",
      "penale__ricorso-per-cassazione.docx", "penale__ricorso-per-cassazione---processo-penale.docx"]),
    ("Richiesta di restituzione dei documenti acquisiti durante le indagini difensive",
     ["penale__richiesta-di-restituzione-della-documentazione-acquisita-nell-espletamento-di-indagini-difensive-art-391-octies-comma-3.docx"]),
    ("Rinvio pregiudiziale in Cassazione sulla competenza per territorio",
     ["penale__richiesta-di-rinvio-pregiudiziale-alla-corte-di-cassazione-per-la-decisione-sulla-competenza-per-territorio-art-24-bis.docx"]),
    ("Lettera dell'inquilino al proprietario e all'amministratore perché il riscaldamento centralizzato non funziona",
     ["contratti__lettera-del-conduttore-al-locatore-e-all-amministratore-del-condominio-per-il-malfunzionamento-dell-impianto-centralizzato-di-riscaldamento.docx"]),
    ("Contratto di mutuo a rata fissa e durata variabile con clausola floor",
     ["contratti__contratto-di-mutuo-a-rata-fissa-ed-a-durata-variabile-con-clausola-c-d-floor.docx"]),
    ("Reclamo contro il decreto che ha rigettato l'accertamento dell'insolvenza prima della liquidazione coatta",
     ["concorsuale__reclamo-avverso-il-decreto-che-ha-respinto-la-richiesta-di-accertamento-dello-stato-di-insolvenza-anteriore-alla-liquidazione-coatta-amministrativa.docx"]),
    ("Il curatore deve chiedere al comitato dei creditori di poter sciogliersi dal contratto di locazione",
     ["concorsuale__istanza-del-curatore-al-comitato-dei-creditori-di-autorizzazione-a-recedere-dal-contratto-di-locazione-immobiliare-pendente-alla-data-di-apertura-della-liquidazione-giudiziale-ai-s.docx"]),
    ("Verbale dell'assemblea della spa che revoca i sindaci",
     ["societario__verbale-di-assemblea-di-s-p-a-per-revoca-dei-sindaci.docx"]),
    ("Azione di responsabilità contro il collegio sindacale di una srl",
     ["societario__azione-di-responsabilita-verso-l-organo-di-controllo-nelle-s-r-l.docx"]),
    ("Istanza di cessazione della materia del contendere nel processo tributario con spese compensate",
     ["tributario__istanza-di-declaratoria-di-cessazione-della-materia-del-contendere-con-compensazione-delle-spese.docx"]),
    ("Accordo di smart working per un neoassunto a tempo determinato",
     ["lavoro__accordo-di-lavoro-agile-a-tempo-determinato-ipotesi-di-nuova-assunzione.docx"]),
    ("Lettera del dipendente per chiedere le ferie",
     ["lavoro__richiesta-del-periodo-di-ferie-da-parte-del-lavoratore-art-2109-c-c.docx"]),
    ("Ricorso contro l'INPS per una prestazione previdenziale negata",
     ["lavoro__ricorso-ex-art-442-c-p-c.docx"]),
    ("Nota spese del commercialista che difende nel processo tributario",
     ["tributario__nota-spese-esperto-contabile-ragioniere-commercialista.docx",
      "tributario__nota-spese-del-dottore-commercialista.docx"]),
    # candidates for "absent" that the catalog turned out to have
    ("Ricorso alla Corte europea dei diritti dell'uomo",
     ["penale__modello-generale-di-ricorso-individuale-alla-corte-edu-artt-34-35-convenzione-e-47-regolamento.docx"]),
    ("Contratto di licenza di un marchio", ["contratti__contratto-di-licenza-di-marchio.docx"]),
    # a real match the model scores low ("badante" vs "collaboratore familiare")
    ("Contratto di lavoro domestico per una badante",
     ["civile__lettera-di-assunzione-di-collaboratore-familiare-non-convivente-artt-2240-c-c-ss.docx"]),
]
# Candidates for "the catalog does not have it" - checked against the catalog below.
# (verified 30 Sept 2026: no template for these; "reclamo alla compagnia aerea" was
# dropped - only a lawsuit for cancelled flights exists, too close to call either way)
ABSENT = [
    "Domanda di brevetto per un'invenzione industriale",
    "Contratto di compravendita di criptovalute",
    "Richiesta di cittadinanza italiana per matrimonio",
    "Ricorso alla Corte costituzionale in via principale",
    "Denuncia di smarrimento del passaporto",
    "Disdetta dell'abbonamento in palestra",
    "Scrivimi una poesia per il compleanno di mia madre",
    "Prepara un business plan per una startup",
    "Atto di matrimonio concordatario",
]


def main():
    catalog = json.load(open(C.CATALOG, encoding="utf-8"))
    fns = {e["filename"] for e in catalog}
    by_fn = {e["filename"]: i for i, e in enumerate(catalog)}
    by_label = {}
    for i, e in enumerate(catalog):
        by_label.setdefault(C.key(e.get("label", "")), i)
    old = json.load(open(C.OLD, encoding="utf-8"))
    pilot = sorted(f[:-5] for f in os.listdir(C.PILOT) if f.endswith(".docx") and not f.startswith("~$"))

    def old_fn(i):
        j = by_fn.get(old[i]["filename"], by_label.get(C.key(old[i]["label"])))
        return catalog[j]["filename"] if j is not None else None

    cases = []
    for q, acc in T.FOUND:
        cases.append({"q": q, "acc": [f for f in (old_fn(i) for i in acc) if f], "kind": "found", "origin": "v1"})
    for q, acc, n in T.NEW:
        a = [f for f in (old_fn(i) for i in acc) if f]
        j = by_label.get(C.key(pilot[n]))
        if j is not None:
            a.append(catalog[j]["filename"])
        cases.append({"q": q, "acc": a, "kind": "found", "origin": "v1-pilot"})
    for q in T.UNRELATED:
        if q in NOW_FOUND:
            cases.append({"q": q, "acc": list(NOW_FOUND[q]), "kind": "found", "origin": "v1-unrelated-now-found"})
        else:
            cases.append({"q": q, "acc": [], "kind": "absent", "origin": "v1"})
    for c in cases:
        for f in ADD.get(c["q"], []):
            if f not in c["acc"]:
                c["acc"].append(f)
    cases += [{"q": q, "acc": acc, "kind": "found", "origin": "new-template"} for q, acc in NEW_TEMPLATES]
    cases += [{"q": q, "acc": [], "kind": "absent", "origin": "new-absent"} for q in ABSENT]

    bad = sorted({f for c in cases for f in c["acc"] if f not in fns})
    if bad:
        sys.exit("labels naming files not in the catalog: %s" % bad)

    # embed any new request, then show the top matches of the new ones
    M = C.normalise(np.load(os.path.join(C.STORE, "template_vectors.npy")), 1024)
    qfile = os.path.join(C.STORE, "test_queries.json")
    cache = json.load(open(qfile, encoding="utf-8"))
    todo = [c["q"] for c in cases if c["q"] not in cache]
    if todo:
        import httpx
        r = httpx.post(C.BASE + "/embeddings", json={"model": C.MODEL, "input": [C.INSTRUCT + q for q in todo]},
                       headers={"Authorization": "Bearer " + C.KEY} if C.KEY else {}, timeout=300)
        r.raise_for_status()
        cache.update({q: d["embedding"] for q, d in zip(todo, r.json()["data"])})
        json.dump(cache, open(qfile, "w", encoding="utf-8"))
    for c in cases:
        if not c["origin"].startswith("new"):
            continue
        s = M @ C.normalise(np.array([cache[c["q"]]], dtype=np.float32), 1024)[0]
        order = np.argsort(-s)[:3]
        print("[%s] %s" % (c["kind"], c["q"]))
        for j in order:
            print("    %s %.3f  %s" % ("*" if catalog[j]["filename"] in c["acc"] else " ", s[j], catalog[j]["label"][:100]))
    json.dump(cases, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n%d requests -> %s  (found %d, absent %d)" % (
        len(cases), OUT, sum(c["kind"] == "found" for c in cases), sum(c["kind"] == "absent" for c in cases)))


if __name__ == "__main__":
    main()
