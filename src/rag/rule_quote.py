"""RULE_QUOTE (default false; acts with PLAN_THEN_WRITE_ENABLED): the rule of each plan article is the comma's own words.

Leonardo v52: rule sentences stated requisites no comma holds ("l'art. 1054 c.c. richiede che il fondo sia divenuto
intercluso per effetto di una vendita e che non vi sia altra via...") and quoted one comma as another. With the flag:

  1. operative(): per article of the writer's plan, the commi that work for the client, by keyword rules over each
     comma's text -- fondo servente / dominante, chi chiede / chi concede, proprietario / possessore / titolare,
     debitore / creditore, imputato / persona offesa -- matched to the client's roles (fact_signals.client_roles,
     client_side, the client's party role) and the other side's. A comma scores when it gives the client a right ("il
     proprietario ... ha diritto") or binds the other side ("il proprietario del fondo servente non può"); it is
     selected with the commi naming no role, a comma naming a role the other way is left out (1067 co. 1 for the
     dominant owner). No comma naming a role: every comma, numbered.
  2. requisites(): the conditions of a selected comma, split on quando / se / purché / salvo che / qualora and on its
     enumerations (";", ", e che", "a)"), each the comma's own words; offered as a numbered list only with 2 to 8 of
     them, else the comma whole.
  3. material_block(): sections 3 and 4 get REGOLE DELLE NORME DEL PIANO with the rule (RULE_TEXT): each rule quoted
     («…» with "co. N") or in words that stay within the comma; each requisite marked "provato" or "da provare" on
     the facts; no requisite added. An article the action gate or ROLE_FIT tags the other side's gets the commi that
     work for that side.
  4. check_sections(): after writing, a rule sentence -- a plan article cited, prevede / stabilisce / dispone / esclude
     / vieta / richiede / precisa / sancisce / consente / riconosce / impone ("ai sensi" is not one), no party named --
     must quote its comma or share >= 50% of its content words with
     it (the comma it names, else its best comma); otherwise it is replaced by the comma's quotation, the next
     sentence (the writer's application) kept. A quotation must belong to the comma number cited: a quotation of
     another comma gets that comma's number. Logged per case as case_analysis_rule_quote, with each article and the
     rejected sentence.

Off, nothing changes. No model call, no database read: the commi come from the rows already read.
"""

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import feature_flags

FLAG = "RULE_QUOTE"
TITLE = "REGOLE DELLE NORME DEL PIANO"
RULE_TEXT = ("Enuncia la regola di ciascuna norma o riportando il comma tra «…» con il suo numero («co. N»), o con parole "
             "tue che restano dentro quel comma: nessun requisito, eccezione o effetto che il comma non contenga. Per "
             "ciascun requisito numerato indica se, sui fatti del caso, è «provato» o «da provare»; non aggiungere "
             "requisiti. Una norma segnata «della controparte» si presenta solo come argomento della controparte.")
SHARE = 0.5                 # a rule sentence's content words that must be in its comma (v53 corrections: 0.7 -> 0.5)
REQUISITES_MIN, REQUISITES_MAX = 2, 8
ARTICLE_CHARS = 1800        # one article's commi in the block, beyond which the commi of the issue are kept
BLOCK_CHARS = 9000


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


# --- 1. the roles ---------------------------------------------------------------------------------------------------

# A comma's words naming a party's position, the specific ones first (their words are then not read again): each tag is
# one side of a pair, _OPPOSITE the other side.
_ROLE_WORDS: Tuple[Tuple[str, str], ...] = (
    ("servente", r"(?:proprietari\w*\s+del\s+)?fondo\s+(?:servente|gravato|asservito)"),
    ("dominante", r"(?:proprietari\w*\s+del\s+)?fondo\s+dominante|titolar\w*\s+della\s+servit\w*"),
    ("chiede", r"\bchi\s+(?:chiede|domanda|richiede)\b|\brichiedent\w*"),
    ("concede", r"\bchi\s+(?:concede|ha\s+concesso)\b|\bconcedent\w*"),
    ("persona_offesa", r"\bpersona\s+offesa\b|\bquerelant\w*"),
    ("imputato", r"\bimputat\w*|\bindagat\w*"),
    ("debitore", r"\bdebitor\w*"),
    ("creditore", r"\bcreditor\w*"),
    ("possessore", r"\bpossessor\w*|\bchi\s+(?:è\s+stato\s+)?(?:\w+\s+){0,3}(?:spogliat\w*|molestat\w*)\s+(?:del|nel)\s+possesso"),
    ("titolare", r"\btitolar\w*"),
    ("proprietario", r"\bproprietari\w*"),
)
_ROLE_RES = [(tag, re.compile(pattern, re.IGNORECASE)) for tag, pattern in _ROLE_WORDS]
_OPPOSITE = {"servente": "dominante", "dominante": "servente", "chiede": "concede", "concede": "chiede",
             "debitore": "creditore", "creditore": "debitore", "imputato": "persona_offesa",
             "persona_offesa": "imputato", "possessore": "proprietario", "proprietario": "possessore"}
# fact_signals.client_roles -> the tags of the client's position
_FROM_ROLES = {"claims_servitude": {"dominante", "titolare", "chiede"}, "resists_servitude": {"servente", "proprietario"},
               "claims_ownership": {"proprietario"}, "possessor": {"possessore"},
               "landlocked_owner": {"proprietario", "dominante", "chiede"}}
_FROM_SIDE = {"accused": {"imputato", "debitore"}, "victim": {"persona_offesa", "creditore"}}
# The client's or a party's own role words ("cliente, proprietario", "debitore ingiunto").
_PARTY_ROLE_RES = [(tag, re.compile(p, re.IGNORECASE)) for tag, p in (
    ("proprietario", r"propriet"), ("possessore", r"possessor"), ("titolare", r"titolar"), ("debitore", r"debitor"),
    ("creditore", r"creditor"), ("imputato", r"imputat|indagat"), ("persona_offesa", r"persona offesa|\boffes"),
    ("servente", r"fondo servente"), ("dominante", r"fondo dominante"))]

# The verb a role's mention governs: the strong ones first ("né può procurarsela ..., ha diritto di ottenere").
_STRONG_RIGHT_RE = re.compile(r"\b(?:ha|hanno)\s+(?:il\s+)?diritto\b|\b(?:può|possono)\s+(?:anche\s+)?(?:chiedere|agire|domandare|"
                              r"ottenere|far\w*|esigere|pretendere|esercitare|opporre)\b|\bè\s+concess\w*", re.IGNORECASE)
_STRONG_DUTY_RE = re.compile(r"\b(?:non|né)\s+(?:può|possono)\s+(?:\w+\s+)?(?:fare|compiere|esercitare|opporre|pretendere|"
                             r"chiedere|aggravare|diminuire|impedire)\b|\b(?:è|sono)\s+(?:tenut|obbligat)\w*|\b(?:deve|devono)\b|"
                             r"\brispond(?:e|ono)\b|\bnon\s+(?:ha|hanno)\s+diritto\b", re.IGNORECASE)
_MODAL_RE = re.compile(r"\b(non\s+|né\s+)?(?:può|possono|ha\s+facoltà|ha\s+diritto|hanno\s+diritto)\b", re.IGNORECASE)
_AGAINST_RE = re.compile(r"\b(?:contro|nei\s+confronti\s+d\w*|a\s+carico\s+d\w*)\s+(?:il\s+|lo\s+|la\s+|l['’]\s*|i\s+|gli\s+|chi\s+)?$",
                         re.IGNORECASE)


def _low(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _client_names(analysis: Dict[str, Any], plan: Optional[Dict[str, Any]]) -> List[str]:
    names = [str((plan or {}).get("parte_assistita") or "").strip(), str(analysis.get("parte_assistita") or "").strip()]
    return [n for n in dict.fromkeys(names) if n]


def _parties(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    facts = ((analysis or {}).get("fact_signals") or {}).get("facts") or {}
    return [p for p in facts.get("parties") or [] if isinstance(p, dict)]


def _is_client(party: Dict[str, Any], clients: Sequence[str]) -> bool:
    name, role = _low(party.get("name")), _low(party.get("role"))
    return "cliente" in role or "assistit" in role or any(name and (name in _low(c) or _low(c) in name) for c in clients)


def sides(analysis: Dict[str, Any], plan: Optional[Dict[str, Any]] = None) -> Tuple[Set[str], Set[str]]:
    """(the client's tags, the other side's): from fact_signals.client_roles, the client's side (client_side on the
    plan's party and position) and the parties' own roles; the other side's are the opposites of the client's and the
    roles of the other parties, none of the client's."""
    from . import fact_signals
    signals = (analysis or {}).get("fact_signals") or {}
    client: Set[str] = set()
    for role in fact_signals.client_roles(signals.get("facts"), signals.get("fired") or []):
        client |= _FROM_ROLES.get(role, set())
    position = f"{(plan or {}).get('parte_assistita') or ''} {(plan or {}).get('posizione') or ''}"
    client |= _FROM_SIDE.get(fact_signals.client_side(position) or str(signals.get("side") or ""), set())
    clients = _client_names(analysis, plan)
    others: Set[str] = set()
    for party in _parties(analysis):
        tags = {tag for tag, rx in _PARTY_ROLE_RES if rx.search(_low(party.get("role")))}
        (client if _is_client(party, clients) else others).update(tags)
    for tag, rx in _PARTY_ROLE_RES:
        if rx.search(_low((plan or {}).get("posizione"))):
            client.add(tag)
    counter = ({_OPPOSITE[t] for t in client if t in _OPPOSITE} | others) - client
    return client, counter


def _mentions(text: str) -> List[Tuple[str, int, int]]:
    """(tag, start, end) of each role a text names, a specific mention's words never read again as a generic one."""
    taken = [False] * len(text)
    out = []
    for tag, rx in _ROLE_RES:
        for m in rx.finditer(text):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            out.append((tag, m.start(), m.end()))
    return sorted(out, key=lambda x: x[1])


def _polarity(text: str, start: int, end: int) -> str:
    """"right", "duty" or "" for the role mentioned at text[start:end]: what its clause gives it or binds it to."""
    if _AGAINST_RE.search(text[max(0, start - 40):start]):
        return "duty"
    clause = re.split(r"(?<=[.;:])\s", text[end:end + 260], maxsplit=1)[0]
    strong = [(m.start(), "right") for m in _STRONG_RIGHT_RE.finditer(clause)] + \
             [(m.start(), "duty") for m in _STRONG_DUTY_RE.finditer(clause)]
    if strong:
        return min(strong)[1]
    m = _MODAL_RE.search(clause)
    if m:
        return "duty" if m.group(1) else "right"
    return ""


def comma_score(text: str, client: Set[str], counter: Set[str]) -> Tuple[int, bool]:
    """(score, names a party's role): +2 a right of the client or a duty of the other side, -1 the reverse."""
    score, named = 0, False
    for tag, start, end in _mentions(text):
        if tag not in client and tag not in counter:
            continue
        named = True
        how = _polarity(text, start, end)
        mine = tag in client
        if (mine and how == "right") or (not mine and how == "duty"):
            score += 2
        elif how:
            score -= 1
    return score, named


# --- the commi of an article ----------------------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^\W*(?:CAPO|SEZIONE|TITOLO|LIBRO|PARTE)\b")
_COMMA_HEAD_RE = re.compile(r"^\(?co\.\s*\d+\)\s*|^\d{1,3}(?:-[a-z]+)?\.\s+", re.IGNORECASE)
# the next chapter's heading the corpus glues to an article's last comma ("... fondi separati. CAPO V – DELL'ESERCIZIO")
_TRAILING_HEADING_RE = re.compile(r"\s+(?:CAPO|SEZIONE|TITOLO|LIBRO|PARTE)\s+[IVXLC\d]+\b.*$")


# v56 (v55-h10 21-nonies / 21-septies, v55-p1 43 / 56 c.p.): the corpus' footnote markers and editorial notes
_FOOTNOTE_RE = re.compile(r"\s*\[\d{1,4}\]")
_NOTE_RE = re.compile(r"(?:^|\s)Note\s*:.*$", re.DOTALL)
_RUBRICA_NOTE_RE = re.compile(r"\s*\(\d{1,3}\)")
REQUISITE_CHARS = 200           # a longer clause is no requisite (a definition, a penalty): its comma's line instead
# v57 (v56-p1 42/43 c.p., 589 c.p.; v56-h10 114 c.p.a.): a comma that defines or states a principle has no requisites;
# a clause on the punishment is none; a fragment cut on a conjunction or a comma is none
DEFINITION_RE = re.compile(r"^\s*(?:nessuno può essere punito|[^.:;]{0,40}:\s*è\b|[^.;]{0,80}\b(?:si intende|si "
                           r"intendono|si considera|si considerano)\b)", re.IGNORECASE)
PENALTY_RE = re.compile(r"\bè punit\w*|\bsono punit\w*|\bla pena\b|\breclusione\b|\bmulta\b|\bammenda\b|"
                        r"\barresto (?:da|fino)\b", re.IGNORECASE)
# v58: an aggravating circumstance or its exception is no element of the offence
AGGRAVANTE_RE = re.compile(r"^(?:il fatto è commess|se il colpevole|il colpevole|ricorr\w* (?:la|taluna|una) (?:delle )?"
                           r"circostanz|salvo che ricorr|la pena è aumentat|le pene sono aumentat)", re.IGNORECASE)
OFFENCE_BOOK = ("c.p.", 241, 649)        # c.p. Libro II: the offences, their requisites co. 1's elements
_PUNISHED_RE = re.compile(r",?\s*(?:(?:è|sono)\s+punit\w*|soggiac\w*|si\s+applica(?:no)?\b).*$", re.IGNORECASE | re.DOTALL)
# v59 (v58-h7 321/319-ter/322, v58-h10 114 c.p.a.): a clause on the penalty or on a consequence -- what follows from the
# requisites, never one of them -- in every code; a clause only pointing at other articles
CONSEQUENCE_RE = re.compile(r"\b(?:pena|pene|punit\w*|punibil\w*|reclusione|multa|ammenda|arresto|si\s+applica(?:no)?|"
                            r"ridott[aoie]|aumentat[aoie]|diminuit[aoie]|titolo\s+esecutivo|interessi\s+legali|"
                            r"in\s+misura\s+pari|penalit\w*|sanzion\w*)\b", re.IGNORECASE)
# a comma extending other articles' rules or penalties («Le pene stabilite … si applicano anche a chi …», «si applica la
# pena della reclusione»): its own line only, no requisites (v58-h7 320, 321, 319-ter)
EXTENSION_RE = re.compile(r"\bsi\s+applica(?:no)?\b(?:\s+(?:anche|altresì|la\s+pena|le\s+pene|le\s+disposizioni))?",
                          re.IGNORECASE)
_REFERENCE_RE = re.compile(r"\b(?:articol[oi]|artt?\.)\s*\d+(?:-[a-z]+)?(?:\s*(?:,|e)\s*\d+(?:-[a-z]+)?)*", re.IGNORECASE)
_REFERENCE_WORDS = {"e", "o", "nel", "nell", "nella", "negli", "nei", "del", "dell", "della", "degli", "dei", "al", "all",
                    "alla", "agli", "ai", "in", "di", "a", "da", "primo", "secondo", "terzo", "quarto", "quinto", "comma",
                    "commi", "relazione", "suddette", "suddetti", "suddetta", "suddetto", "ipotesi", "previste",
                    "previsti", "prevista", "previsto", "stabilite", "stabiliti", "stabilita", "indicate", "indicati",
                    "medesime", "stesse", "cui", "le", "la", "il", "lo", "i", "gli", "l", "alle", "allo", "dello",
                    "delle", "dai", "dagli", "dal", "dalla", "nelle", "nello", "sensi", "casi", "caso"}


def reference_only(clause: str) -> bool:
    """«nell'articolo 319», «e nell'articolo 320 in relazione alle suddette ipotesi degli articoli 318 e 319»: nothing but
    a pointer to other articles."""
    if not _REFERENCE_RE.search(clause or ""):
        return False
    rest = _REFERENCE_RE.sub(" ", str(clause).lower())
    return not [w for w in re.findall(r"[a-zà-ù]+", rest) if w not in _REFERENCE_WORDS]


def requisite_kept(clause: str) -> bool:
    """v59: may this clause be a requisite line? Not a penalty or a consequence (CONSEQUENCE_RE), not an aggravante, not
    only an article reference."""
    clause = str(clause or "")
    return bool(clause.strip()) and not CONSEQUENCE_RE.search(clause) and not AGGRAVANTE_RE.search(clause.strip()) \
        and not reference_only(clause)


def extends(text: str) -> bool:
    """Does this comma extend other articles' rules or penalties (EXTENSION_RE) rather than set its own?"""
    return bool(EXTENSION_RE.search(clean_text(text or "")))


_OWN_SUBJECT_RE = re.compile(r"\bchiunque\b|^\W*(?:chi|il pubblico ufficiale|l['’]incaricato)\b|"
                             # v76b: the subject after a leading clause («Quando …, coloro che partecipano …»,
                             # «Fuori dei casi …, il pubblico ufficiale … che …»)
                             r"\b(?:coloro|colui|colei)\s+che\b|,\s*(?:il\s+pubblico\s+ufficiale|l['’]incaricato)\b",
                             re.IGNORECASE)
_SANCTIONED_RE = re.compile(r"\bnei\s+confronti\s+d\w*\s+[^.;]{2,200}?\bche\b|\ba\s+chiunque\b|"
                            r"\ba\s+carico\s+d\w*\s+[^.;]{2,120}?\bche\b", re.IGNORECASE)
_RULES_SUBJECT_RE = re.compile(r"^\W*(?:le|la|il|i|gli)\s+(?:disposizion|pen[ae]\b|norm|sanzion\w*\s+previst)",
                               re.IGNORECASE)
_NP_VERB_RE = re.compile(r"\s(?:il|lo|la|l['’]|i|gli|le)\s*[a-zà-ü]+(?:\s+[a-zà-ü]+){0,3}?\s+(?:non\s+)?(?:ha|hanno|è|sono|"
                         r"può|possono|deve|devono|resta|restano|si\s+\w+)\b", re.IGNORECASE)
# v76b: a clausola di riserva or a default -- never an element («Fuori dei casi previsti …», «Salvo che il fatto
# costituisca reato», «se il contratto non dispone diversamente», «salvo patto contrario»)
_NOT_ELEMENT_RE = re.compile(r"^\W*(?:fuori\s+de[il]\s+casi|salvo\s+che\s+il\s+fatto|se\s+il\s+fatto\s+non\s+costituisce)|"
                             r"\bsalvo\s+quanto\b|\bfatt[oaie]\s+salv|\bferm[oaie]\s+rest|"
                             r"\b(?:diversamente|altrimenti|patto\s+contrario|diversa\s+(?:disposizione|pattuizione|"
                             r"volontà|previsione)|clausola\s+contraria|(?:non\s+)?(?:è\s+)?(?:stabilito|convenuto|"
                             r"disposto)\s+(?:altrimenti|diversamente))\b", re.IGNORECASE)


# v79.8 (C2): a decomposition fragment that states no condition -- never an element (dropped with its reason, logged)
_ROLE_NOUN = (r"(?:debitor\w*|creditor\w*|contraent\w*|parte|parti|vendit\w*|compratore|acquirente|locat\w*|"
              r"conduttore|comodant\w*|comodatario|appaltatore|committente|mandant\w*|mandatario|datore(?:\s+di\s+"
              r"lavoro)?|lavorator\w*|proprietari\w*|possessore|custode|danneggiat\w*|danneggiante|colpevole|agente|"
              r"terz[oi]|soggett\w*|person[ae]|interessat\w*|titolare|obbligat\w*)")
_BARE_SUBJECT_RE = re.compile(r"^(?:(?:il|lo|la|l['’]|i|gli|le|un|uno|una|ciascun[oa]|ognun[oa])\s*)?(?:(?:dei|degli|delle|"
                              r"di|del|della)\s+)?" + _ROLE_NOUN + r"(?:\s+(?:di|del|della|dei|delle)?\s*" + _ROLE_NOUN
                              + r")?$", re.IGNORECASE)
_CONNECTIVE_RE = re.compile(r"^(?:con|per|in|nel|nella|mediante|secondo)\s+(?:cui|il\s+quale|la\s+quale|i\s+quali|"
                            r"le\s+quali)\b(?:\s+(?:si|le\s+parti|esse|essi)?\s*\w+){0,2}(?:\s+che)?$|"
                            r"^(?:\w+\s+){0,3}(?:che|cui|quale|qualora|se|quando)$", re.IGNORECASE)
_SAVING_RE = re.compile(r"^\W*(?:salvo\b|in\s+ogni\s+caso\b|fatt[oaie]\s+salv|ferm[oaie]\s+rest|"
                        r"tranne\s+che|eccetto\s+che|a\s+meno\s+che)", re.IGNORECASE)
# v79.9 (D2): a modal permission («può chiedere la risoluzione»: the consequence, no condition), a bare object
# («il danno», «la cosa»: one noun), a scope clause («riguardo ai quali …», «nei limiti di …», «ai sensi di …»)
_PERMISSION_RE = re.compile(r"\b(?:può|possono|potrà|potranno|potrebbe|potrebbero)\s+(?:anche\s+|altresì\s+|sempre\s+)?"
                       r"(?:\w+(?:are|ere|ire|rre|arsi|ersi|irsi))\b", re.IGNORECASE)
_CONDITION_RE = re.compile(r"\b(?:se|quando|qualora|purch[ée]|ove|allorch[ée]|nel\s+caso|nei\s+casi|in\s+caso)\b",
                           re.IGNORECASE)
# a remedy's object, never a status: «il cittadino», «l'operatore» are qualified subjects, elements (v76b)
_BARE_OBJECT_RE = re.compile(r"^(?:il|lo|la|l['’]|i|gli|le|un|uno|una)\s*(?:dann[oi]|cos[ae]|prezzo|ben[ei]|somm[ae]|denaro|"
                             r"risarciment[oi]|indennit[àa]|interessi|spese|rimborso|compenso|corrispettivo|pagament[oi]|"
                             r"restituzion[ei]|risoluzione|riduzione|frutti|canon[ei]|pen[ae])$", re.IGNORECASE)
_SCOPE_RE = re.compile(r"^\W*(?:riguardo\s+(?:a|al|alla|ai|alle|agli|all['’])|relativamente\s+a|per\s+quanto\s+riguarda|"
                       r"in\s+ordine\s+a|con\s+riferimento\s+a|nei\s+limiti|entro\s+i\s+limiti|limitatamente\s+a|"
                       r"ai\s+sensi\s+d|a\s+norma\s+d|agli\s+effetti\s+d|per\s+gli\s+effetti\s+d)", re.IGNORECASE)
FRAGMENT_WHY = {"subject": "soggetto senza condizione", "connective": "connettivo senza condizione",
                "saving": "clausola di salvezza", "modal": "facoltà, non condizione",                  # v79.9 (D2)
                "object": "oggetto senza condizione", "scope": "clausola di ambito",
                "category": "categoria di soggetti o di attività, non condizione",                     # v79.10.4 (D4)
                "confiscation": "clausola di confisca, non condizione",
                "medicinal": "clausola sui medicinali, non condizione"}
# v79.10.4 (D4): statutory wording naming a category -- persons by their plural («i precettori», «i genitori e i
# tutori»), «coloro che» with an activity, a trade or an art («un mestiere o un'arte») -- with no condition in it
_DET = r"(?:(?:i|gli|le|il|lo|la|l['’]|un|uno|una|un['’])\s*)"
_TRADE = r"(?:mestier[ei]|art[ei]|professio\w+|industri[ae]|commerci[oa]?|attività|impres[ae]|uffici[oa]?)"
_PERSONS = r"(?:(?:i|gli|le)\s+[a-zà-ù]{4,}i\b)"
_COLORO = r"(?:coloro|quelli|quanti)\s+che\s+[a-zà-ù]+(?:\s+(?:a\s+)?" + _DET + r"?" + _TRADE + \
    r"(?:\s*(?:,|e|ed|o|od)\s*" + _DET + r"?" + _TRADE + r")*)?"
_TRADES = _DET + r"?" + _TRADE + r"(?:\s*(?:,|e|ed|o|od)\s*" + _DET + r"?" + _TRADE + r")*"
_CATEGORY_RE = re.compile(r"^(?:" + _TRADES + r"|(?:" + _PERSONS + r"|" + _COLORO + r")(?:\s*(?:,|e|ed|o|od)\s*(?:"
                          + _PERSONS + r"|" + _COLORO + r"))*)$", re.IGNORECASE)
# a confiscation clause (the consequence: «è sempre ordinata la confisca», «sono confiscati», «la confisca di …»); goods
# a conduct names as confiscated («sottrae cose sottoposte a sequestro o confiscate») are an element
_CONFISCATION_RE = re.compile(r"\b(?:è|sono)\s+(?:sempre\s+|altresì\s+)?(?:(?:ordinat|dispost)\w*\s+(?:la\s+)?confisc|"
                              r"confiscat[oaie]\b)|^\W*(?:(?:la|alla|della)\s+)?confisca\b|\bsi\s+procede\s+(?:alla\s+|a\s+)?"
                              r"confisca|\b(?:ordina|dispone)\s+(?:sempre\s+)?(?:la\s+)?confisca", re.IGNORECASE)
# a medicinal clause: the clause opens on the medicinals (or on their table or section) and names them by their table,
# section, prescription or therapeutic use («i medicinali compresi nella tabella …», «la vendita di medicinali dietro
# presentazione di ricetta»); a conduct on medicinals («vende medicinali senza ricetta», «detiene medicinali …
# previsti dalla tabella») is an element
_MED_HEAD = r"^\W*(?:(?:i|gli|le|l['’]|la|il|dei|delle|degli|per\s+i|per\s+le|ai|alle)\s*)?" \
    r"(?:(?:vendita|dispensazione|somministrazione|prescrizione|detenzione|impiego|uso|elenco|fornitura)\s+(?:di|dei|delle)\s+)?"
_MEDICINAL_RE = re.compile(_MED_HEAD + r"(?:medicinal\w*|farmac[io]\b)(?=[^.;]*\b(?:tabell\w*|sezion\w*|ricett\w*|"
                           r"prescrizion\w*\s+medic\w*|uso\s+terapeutic\w*))|"
                           r"^\W*(?:(?:la|le|nella|nelle|della|delle|alla|alle)\s+)?(?:tabell\w*|sezion\w*)\b[^.;]*\bmedicinal\w*",
                           re.IGNORECASE)
# a modal permission in its wider forms: «può essere disposta», «possono esserne», «è consentito», «è in facoltà di», «ha
# facoltà di»
# (an event that may follow -- «può derivarne un pericolo», «possono verificarsi» -- is a condition; what the law allows --
# «il fatto è consentito dalla legge» -- a scriminante's condition: neither is a permission)
_EVENT_VERB = r"(?!(?:deriv|consegu|risult|verific|sorg|nasc|produr|accad|aver|esser)\w*\b)"
_PERMISSION_MORE_RE = re.compile(r"\b(?:può|possono|potrà|potranno)\s+(?:anche\s+|altresì\s+|sempre\s+)?essere\s+\w+(?:at|ut|it|ss|st)"
                                 r"[oaie]\b|\b(?:può|possono|potrà|potranno)\s+(?:anche\s+|altresì\s+|sempre\s+)?" + _EVENT_VERB +
                                 r"\w+(?:ar|er|ir)(?:l[oaie]|ne|si|gli|ci|vi)\b|\b(?:è|sono)\s+(?:altresì\s+|sempre\s+)?"
                                 r"(?:consentit|permess)[oaie]\b(?!\s+(?:dalla|dal|dallo|dall['’]|dai|dalle|dagli|da|per\s+legge)\b)|"
                                 r"\b(?:è\s+in|ha|hanno)\s+facoltà\s+d", re.IGNORECASE)


def fragment_kind(text: str, wider: bool = True) -> str:
    """v79.10.4 (D4): fragment_why's kind -- a FRAGMENT_WHY key, "" for a condition. wider=False: v79.9's kinds only
    (no category, confiscation or medicinal clause, the modal permission in its v79.9 form) -- what a generated schema
    is checked on (element_schemas.check): D4's fragments are dropped one by one, never the schema."""
    t = " ".join(str(text or "").split()).strip(" ,.;:")
    if _SAVING_RE.match(t):
        return "saving"
    if _SCOPE_RE.match(t):                                  # v79.9 (D2)
        return "scope"
    if wider and _CONFISCATION_RE.search(t):                # v79.10.4 (D4)
        return "confiscation"
    if wider and _MEDICINAL_RE.search(t):
        return "medicinal"
    if not wider:
        if _PERMISSION_RE.search(t) and not _CONDITION_RE.search(t):
            return "modal"
    elif (_PERMISSION_RE.search(t) or _PERMISSION_MORE_RE.search(t)) and not _CONDITION_RE.search(t) \
            and not re.search(r"\bnon\s+(?:può|possono|potrà|potranno|è|sono|ha|hanno)\b", t, re.IGNORECASE):
        return "modal"
    if _BARE_OBJECT_RE.match(t):
        return "object"
    if _BARE_SUBJECT_RE.match(t):
        return "subject"
    if wider and _CATEGORY_RE.match(t):                     # v79.10.4 (D4)
        return "category"
    if _CONNECTIVE_RE.match(t):
        return "connective"
    return ""


def fragment_why(text: str) -> str:
    """v79.8 (C2): why a decomposed clause is no element -- a bare subject («il debitore», «uno dei contraenti»), a
    connective («con cui si conviene che»), a saving clause («salvo in ogni caso …»); v79.9 (D2): a modal permission
    («può chiedere …», with no condition in it), a bare object («il danno»), a scope clause («riguardo ai quali …»,
    «nei limiti di …»); v79.10.4 (D4): statutory wording naming a category («i precettori e coloro che insegnano un
    mestiere o un'arte», «un mestiere o un'arte»), a confiscation clause («è sempre ordinata la confisca»), a medicinal
    clause («i medicinali compresi nella tabella …»), a modal permission in its wider forms («può essere disposta», «è
    consentito», «ha facoltà di»; a negated «non può», «non è consentito» is a prohibition, never one); "" for a condition."""
    kind = fragment_kind(text)
    return FRAGMENT_WHY[kind] if kind else ""


def element_kept(text: str) -> bool:
    """v76b: a clause that is an element -- never a clausola di riserva nor a default clause."""
    return not _NOT_ELEMENT_RE.search(text or "")
# v76b: a qualified subject with its own relative clause («Il cittadino, o lo straniero …, il quale …», «L'imprenditore
# che …») -- never the penalties' or the rules' («Le pene stabilite … che», «Le disposizioni … che»)
_RELATIVE = r"(?:che|il\s+quale|la\s+quale|i\s+quali|le\s+quali)"
_QUALIFIED_RE = re.compile(r"^\W*((?:il|la|lo|i|gli|le)\s+|l['’]\s*)(?!pen[ae]\b|disposizion|norm|cas[io]\b|sanzion|"
                           r"limit|termin|stess)([^.;:]{2,100}?)\s*,?\s*\b" + _RELATIVE + r"\b\s*,?\s*", re.IGNORECASE)
_MARKER_RE = re.compile(r"\s+(?=(?:con|mediante|in\s+modo\s+da|al\s+fine\s+di|allo\s+scopo\s+di|a\s+fine\s+di|"
                        r"per\s+procurare|per\s+trarne|in\s+conseguenza\s+d\w*|per\s+effetto\s+d\w*|a\s+causa\s+d\w*|"
                        r"senza)\b)", re.IGNORECASE)


_RISERVA_HEAD_RE = re.compile(r"^\W*(?:salvo\s+che\s+il\s+fatto\s+costituisca[^,]{0,120},|fuori\s+de[il]\s+casi[^,]{0,160},)"
                              r"\s*", re.IGNORECASE)


def without_riserva(text: str) -> str:
    """v76b: the comma without its leading clausola di riserva («Salvo che il fatto costituisca reato, …»)."""
    return _RISERVA_HEAD_RE.sub("", text or "", count=1)


def qualified_subject(text: str) -> Tuple[str, str]:
    """v76b: (the subject's quality, the rest) of a clause opening on a qualified subject and its relative («Il
    cittadino che porta le armi …» -> («il cittadino», «porta le armi …»)); ("", text) otherwise."""
    m = _QUALIFIED_RE.match(text or "")
    if not m:
        return "", text
    subject = " ".join((m.group(1) + m.group(2)).split()).strip(" ,").lower()
    if len(subject.split()) > 12 or re.search(r"\b(?:cui|che|quale|quali)\b", subject) or \
            re.search(r"[\s,](?:e|o|ed|od)$", subject):       # a clause of its own («il cui fondo …, e»): no subject
        return "", text
    return subject, text[m.end():]


def split_markers(pieces: Sequence[str]) -> List[str]:
    """v76b: each piece cut before its means / purpose / cause markers (con, mediante, in modo da, al fine di, in
    conseguenza di, per effetto di, senza …), the parts of two words or more kept."""
    out = []
    for piece in pieces:
        for part in _MARKER_RE.split(piece):
            part = part.strip(" ,;.")
            if len(part.split()) >= 2:
                out.append(part)
    return out


def refers_penalties(text: str) -> bool:
    """v59 (v55-l2 585 c.p.: «Nei casi previsti dagli articoli 582, 583 …, la pena è aumentata …»): a comma extending
    other articles' rules, or setting a penalty for the cases other articles provide with no subject of its own
    (never «Chiunque, fuori dei casi indicati nell'articolo 571, maltratta …») -- its own line, no requisite."""
    text = " ".join(_COMMA_HEAD_RE.sub("", clean_text(text or "").strip(), count=1).split())
    text = re.split(r"(?<=[.;])\s+(?=[A-ZÀ-Ü])", text, maxsplit=1)[0]     # v76b: its first sentence decides
    own = bool(_OWN_SUBJECT_RE.search(text) or _QUALIFIED_RE.match(without_riserva(text))
               or _SANCTIONED_RE.search(text))
    if _RULES_SUBJECT_RE.match(text) and CONSEQUENCE_RE.search(text):     # v76b: «Le pene stabilite negli articoli
        return True                                                       # precedenti sono diminuite se …»
    if extends(text):           # v76b: «Le disposizioni … si applicano …» extends; «… nei confronti dei soggetti che
        return bool(_RULES_SUBJECT_RE.match(text)) or not own        # omettono … si applica la sanzione» is its own
    return bool(CONSEQUENCE_RE.search(text) and _REFERENCE_RE.search(text) and not own)


def offence(ref: Sequence[str]) -> bool:
    from .plan_then_write import _code_key
    m = re.match(r"\d+", str(ref[0]))
    return bool(m) and _code_key(str(ref[1])) == OFFENCE_BOOK[0] and OFFENCE_BOOK[1] <= int(m.group(0)) <= OFFENCE_BOOK[2]


_PENAL_PROVISION_RE = re.compile(r"\b(?:è|sono)\s+punit[oaie]\b|\bsi\s+applica\s+la\s+(?:pena|reclusione|multa|"
                                 r"sanzione)\b|\bè\s+soggett[oa]\s+alla\s+sanzione\b", re.IGNORECASE)


def _is_cp(ref: Sequence[str]) -> bool:
    from .plan_then_write import _code_key
    return _code_key(str(ref[1])) in ("c.p.", "c.c.", "c.p.p.", "c.p.c.", "c.p.a.")


def penal_provision(text: str) -> bool:
    """v76b: a comma that punishes a conduct («Chiunque … è punito», «… è soggetto alla sanzione …») -- an offence's
    first comma in any code (a special law's penal provision too)."""
    return bool(_PENAL_PROVISION_RE.search(text or "")) and not refers_penalties(text)


# --- v79.6 (item 4): alternative elements -- one element with branches, never several required ones ------------------------

_DISJ = r"(?:o|od|ovvero|oppure)"
_DISJ_HEAD_RE = re.compile(rf"^\s*{_DISJ}\b", re.IGNORECASE)
_DISJ_IN_RE = re.compile(rf"\s{_DISJ}\s", re.IGNORECASE)
_MARKER_HEAD_RE = re.compile(r"^\s*(?:con|mediante|in\s+modo\s+da|al\s+fine\s+di|allo\s+scopo\s+di|a\s+fine\s+di|per\s+"
                             r"procurare|per\s+trarne|senza|salvo|se|quando|qualora|purch[ée])\b", re.IGNORECASE)
BRANCH_LIST_WORDS = 6


def join_branches(pieces: Sequence[str]) -> List[str]:
    """v79.6: comma pieces with a branch list put back together -- a piece opening on «o / od / ovvero / oppure» (and «o,
    in alternativa,»), a short piece (BRANCH_LIST_WORDS words at most, no means or purpose marker) holding one, and the
    short items of an enumeration that ends in one («educazione, istruzione, cura, vigilanza o custodia») join the
    piece before them: the alternatives are one element, never several required elements."""
    items = [str(p or "").strip() for p in pieces]
    out: List[str] = []
    for i, p in enumerate(items):
        if not p:
            continue
        words = len(p.split())
        lead = bool(_DISJ_HEAD_RE.match(p)) or bool(re.match(r"^\s*in\s+alternativa\b", p, re.IGNORECASE))
        tail = words <= BRANCH_LIST_WORDS and bool(_DISJ_IN_RE.search(f" {p} ")) and not _MARKER_HEAD_RE.match(p)
        run = []                    # the short pieces after this one, up to the one that closes the list
        for q in items[i + 1:]:
            run.append(q)
            if len(q.split()) > BRANCH_LIST_WORDS or _MARKER_HEAD_RE.match(q):
                break
        item = words <= 2 and not _MARKER_HEAD_RE.match(p) and any(
            _DISJ_IN_RE.search(f" {q} ") or _DISJ_HEAD_RE.match(q) for q in run if len(q.split()) <= BRANCH_LIST_WORDS)
        joinable = out and not _MARKER_HEAD_RE.match(out[-1])      # a means or purpose clause is its own element
        if out and (lead or (joinable and (tail or item))):
            out[-1] = f"{out[-1]}, {p}"
        else:
            out.append(p)
    return out


_BRANCH_SPLIT_RE = re.compile(rf"\s*,\s*(?:{_DISJ}\s+)?(?:in\s+alternativa\s*,?\s*)?|\s+{_DISJ}\s+(?:,?\s*in\s+alternativa"
                              rf"\s*,?\s*)?", re.IGNORECASE)


def fits(piece: str) -> bool:
    """An element's length: REQUISITE_CHARS, twice that for one holding its alternatives (v79.6)."""
    return len(piece or "") <= (2 * REQUISITE_CHARS if branches(piece) else REQUISITE_CHARS)


def branches(text: str) -> List[str]:
    """v79.6: an element's alternatives, when it holds a disjunction -- «una persona della famiglia o comunque
    convivente» -> [«una persona della famiglia», «comunque convivente»]; [] for an element with none."""
    if not _DISJ_IN_RE.search(f" {text or ''} "):
        return []
    parts = [x.strip(" ,;.") for x in _BRANCH_SPLIT_RE.split(text or "") if x and x.strip(" ,;.")]
    return parts if len(parts) >= 2 else []


def offence_elements(text: str) -> List[str]:
    """v58 (v57-p2 640: «Chiunque, con artifizi o raggiri, …» is the requisite set): the elements of an offence's first
    comma -- its words before «è punito», «Chiunque» left out, split on its commas; an element of fewer than two words
    joins the next."""
    whole = " ".join(_COMMA_HEAD_RE.sub("", clean_text(text).strip(), count=1).split())
    if refers_penalties(whole):     # v59: «… si applicano anche a chi …», «Nei casi previsti …»: its own line only
        return []
    body = _PUNISHED_RE.sub("", whole)
    body = re.sub(r"^(?:chiunque|il pubblico ufficiale|l['’]incaricato di un pubblico servizio|chi)\b\s*,?\s*", "", body,
                  flags=re.IGNORECASE)       # v76: a whole word -- never the first letters of the clause
    out: List[str] = []
    carry = ""
    for piece in join_branches(re.split(r"\s*,\s*", body)):          # v79.6: a branch list is one element
        piece = f"{carry} {piece}".strip() if carry else piece.strip()
        if len(piece.split()) < 2:
            carry = piece
            continue
        carry = ""
        if requisite_kept(piece) and fits(piece) and element_kept(piece):
            out += _subjective_split(piece.strip(" .;:"))
    # v59 (v58-h7 322: «soggiace, qualora l'offerta o la promessa non sia accettata, alla pena …»): a condition set
    # inside the penalty clause is an element too
    penalty = _PUNISHED_RE.search(whole)
    for piece in re.split(r"\s*,\s*", whole[penalty.start():] if penalty else ""):
        piece = piece.strip(" .;:")
        if re.match(r"(?:qualora|se|purch[ée]|sempre che)\b", piece, re.IGNORECASE) and requisite_kept(piece) \
                and len(piece.split()) >= 3 and len(piece) <= REQUISITE_CHARS:
            out.append(piece)
    return [x for x in out if x]


# v74: the subjective element the offence's clause names («cagiona per colpa la morte di una persona») is an element of
# its own -- every record is split into elements the way the charge is, a one-clause offence too
_SUBJECTIVE_RE = re.compile(r"\s*\b(per colpa)\b\s*", re.IGNORECASE)


def _subjective_split(piece: str) -> List[str]:
    m = _SUBJECTIVE_RE.search(piece)
    if not m:
        return [piece]
    rest = " ".join((piece[:m.start()] + " " + piece[m.end():]).split())
    return [rest, m.group(1).lower()] if len(rest.split()) >= 2 else [piece]


# v76: the general penal splitter -- the clause after «chiunque» (wherever the penalty stands: «è punito … chiunque,
# con condotte reiterate, minaccia o molesta taluno in modo da cagionare …»), cut on its commas and before «con»,
# «mediante», «in modo da», «al fine di», «allo scopo di», «per procurare»
_PENAL_MARK_RE = re.compile(r"\s+(?=(?:con|mediante|in\s+modo\s+da|al\s+fine\s+di|allo\s+scopo\s+di|a\s+fine\s+di|"
                            r"per\s+procurare|per\s+trarne)\b)", re.IGNORECASE)


def penal_clauses(text: str) -> List[str]:
    """v76: an offence's elements when its commas alone do not give them (offence_elements): the clause the offence
    describes, from «chiunque» (else up to «è punito»), split on its commas and its means / purpose / effect markers."""
    whole = " ".join(_COMMA_HEAD_RE.sub("", clean_text(text).strip(), count=1).split())
    if refers_penalties(whole):         # «… si applicano anche a chi …»: its own line only, as offence_elements
        return []
    m = re.search(r"\bchiunque\b", whole, re.IGNORECASE)
    body = whole[m.end():] if m else _PUNISHED_RE.split(whole, maxsplit=1)[0]
    body = re.split(r"(?<=[.;])\s", body, maxsplit=1)[0]
    body = _PUNISHED_RE.split(body, maxsplit=1)[0]
    if not m and not element_kept(body.strip(" ,")):    # v76b: the subject after the penalty («Salvo che …, è punito
        first = re.split(r"(?<=[.;])\s", whole, maxsplit=1)[0]       # con … il pubblico ufficiale … che, al fine di …»)
        after = re.search(r"\b(?:il|la|lo|i|gli|le|l['’])\s*[a-zà-ü][^.;]{1,100}?\b" + _RELATIVE + r"\b", first,
                          re.IGNORECASE)
        punished = _PUNISHED_RE.search(first)
        if after and punished and after.start() > punished.start():
            body = first[after.start():]
    out: List[str] = []
    subject, rest = ("", body) if m else qualified_subject(without_riserva(body.strip()))     # v76b: a reato proprio
    # -- the subject's quality an element
    if subject:
        out.append(subject)
        body = rest
    for piece in join_branches(re.split(r"\s*,\s*", body)):          # v79.6: a branch list is one element
        for part in _PENAL_MARK_RE.split(piece):
            part = part.strip(" ,;.")
            if len(part.split()) >= 2 and requisite_kept(part) and fits(part) \
                    and not PENALTY_RE.search(part) and element_kept(part):
                out += _subjective_split(part)
    return list(dict.fromkeys(out))


# v75: a civil fattispecie told in one sentence -- «Qualunque fatto doloso o colposo, che cagiona ad altri un danno
# ingiusto, obbliga …», «Il debitore che non esegue esattamente la prestazione dovuta è tenuto al risarcimento …»: split
# at its consequence, the antecedent's clauses are its elements; a risarcimento whose antecedent names no danno has the
# danno as an element too
_CONSEQUENCE_RE = re.compile(r"\b(?:è|sono)\s+tenut[oaie]\b|\bobblig(?:a|ano)\b|\brispond(?:e|ono)\b|"
                             r"\b(?:è|sono)\s+responsabil[ei]\b|\bha(?:nno)?\s+diritto\b|\bdev(?:e|ono)\b", re.IGNORECASE)
_SUBJECT_HEAD_RE = re.compile(r"^(?:(?:qualunque|qualsiasi|chiunque|ogni|il|lo|la|i|gli|le)\s+|l['’]\s*)",
                              re.IGNORECASE)        # v76: a whole word only -- «imputato», «lesione» keep their letters


def _fitting(reqs: List[str]) -> List[str]:
    return reqs if REQUISITES_MIN <= len(reqs) <= REQUISITES_MAX else []


def _clauses(text: str) -> List[str]:
    """A fattispecie's clauses: split on commas and relative pronouns, each of two words and up, its pronoun dropped."""
    out = []
    for piece in join_branches(re.split(r"\s*,\s*|\s+(?=(?:che|il quale|la quale|i quali|le quali)\s)", text or "")):
        piece = re.sub(r"^(?:che|il quale|la quale|i quali|le quali)\s+", "", piece.strip(" ,;."), flags=re.IGNORECASE)
        words = [w for w in piece.split() if not re.fullmatch(r"(?:il|lo|la|l['’]\w*|i|gli|le|un|uno|una)", w, re.I)]
        if len(words) >= 2 and requisite_kept(piece) and len(piece) <= REQUISITE_CHARS:
            out.append(piece)
    return out


_CHI_RE = re.compile(r"^\W*(chi\s+[^,;.]{3,100}?)\s*,?\s+(?=(?:non\s+)?(?:è|sono|può|deve|ha\s+diritto|risponde|resta|"
                     r"rimane|perde|decade|acquista|è\s+tenut)\b)", re.IGNORECASE)


def subject_clause(text: str) -> str:
    """v76b: the clause naming who the rule is about -- «Chi ha ricevuto le cose …» (to its main verb), or a qualified
    subject with its relative (qualified_subject); "" otherwise."""
    flat = " ".join(_COMMA_HEAD_RE.sub("", clean_text(text).strip(), count=1).split())
    m = _CHI_RE.match(flat)
    if m:
        return m.group(1).strip()
    subject, rest = qualified_subject(flat)
    return subject


def civil_split(text: str, single: bool = False) -> List[str]:
    """v76b: a civil comma's elements, the first split that gives two or more -- its conditions (requisites), the
    single condition cut on its markers, the fattispecie told in one sentence (civil_elements); else, with single (the
    records' own decomposition, never the writer's material), its one condition of three words or more."""
    conds = [c for c in requisites(text) if element_kept(c)]       # a default clause is no element
    subject = subject_clause(text)          # «Chi ha ricevuto le cose …», «Il fideiussore per un'obbligazione futura»
    commas = [x.strip(" ,;.") for x in re.split(r"\s*,\s*", conds[0]) if len(x.split()) >= 3] if len(conds) == 1 else []
    for found in (conds, commas, [subject] + conds if subject and len(conds) == 1 else [], civil_elements(text)):
        if _fitting(found):
            return found
    return conds if single and len(conds) == 1 and len(conds[0].split()) >= 3 else []


def civil_elements(text: str) -> List[str]:
    """v75: the elements of a civil fattispecie in one sentence (above); [] when the sentence has no consequence verb
    or its antecedent no clause."""
    whole = " ".join(_COMMA_HEAD_RE.sub("", clean_text(text).strip(), count=1).split())
    first = re.split(r"(?<=[.;])\s", whole, maxsplit=1)[0]
    m = _CONSEQUENCE_RE.search(first)
    if not m:
        return []
    antecedent, consequence = first[:m.start()], first[m.end():]
    if re.match(r"\s*(?:se|quando|qualora|ove|nei\s+casi|nel\s+caso)\b", antecedent, re.IGNORECASE):
        return []               # a condition first: the requisites' own split (requisites) reads it
    subject, rest = qualified_subject(antecedent)       # v76b: «L'imprenditore che …»: the quality an element
    out = ([subject] if subject else []) + _clauses(rest if subject else _SUBJECT_HEAD_RE.sub("", antecedent))
    # (civil clauses are cut on commas and relatives only: «senza eccessivo dispendio» belongs to its verb)
    guarantee = re.match(r"\s*(?:a\s+)?(?:garantire|assicurare)\s+che\s+(.+)", consequence, re.IGNORECASE)
    if not out and guarantee:   # v76: «Il venditore è tenuto a garantire che la cosa venduta sia immune da vizi …»
        out = _clauses(guarantee.group(1))
    if not out and re.match(r"(?:è|sono)\s+responsabil|rispond", m.group(0), re.IGNORECASE):
        # «Ciascuno è responsabile del danno cagionato dalle cose che ha in custodia, salvo che …»: what one answers for
        body = re.split(r"\s*,?\s*\b(?:salvo\s+che|se\s+non|tranne\s+che|a\s+meno\s+che)\b", consequence)[0]
        out = _clauses(re.sub(r"^\s*(?:del|della|dello|dei|degli|delle|per\s+(?:il|la|i|gli|le))\s+", "", body))
    if out and re.search(r"\brisarc\w*\s+(?:del\s+)?dann", consequence, re.IGNORECASE) and \
            not any(re.search(r"\bdann", x, re.IGNORECASE) for x in out):
        out.append("il danno")
    return out


_DANGLING_RE = re.compile(r"\b(?:e|ed|o|od|ma|né|oppure|ovvero|nonché)\s*$", re.IGNORECASE)
_DISJ_END_RE = re.compile(r"[;,]?\s*\b(?:o|od|oppure|ovvero)\s*$", re.IGNORECASE)       # v79.6


# v58 (v57-p2 640, v57-h5 628): the c.p.'s footnote numbers «(3)» and the stray «]» of a removed passage
_PAREN_NOTE_RE = re.compile(r"\s*\(\d{1,2}\)(?=[\s.,;:»]|$)")
_STRAY_BRACKET_RE = re.compile(r"\s*\](?![^\[]*\])|\[(?![^\[\]]*\])\s*")


def clean_text(text: Any) -> str:
    """An article's text without its footnote markers ("[196]", "(3)") and cut at its editorial notes ("Note: ...")."""
    text = _FOOTNOTE_RE.sub("", _NOTE_RE.sub("", str(text or "")))
    text = _STRAY_BRACKET_RE.sub(" ", _PAREN_NOTE_RE.sub("", text))
    return "\n".join(" ".join(line.split()) for line in text.split("\n")).strip()       # the commi's lines kept


def clean_rubrica(rubrica: Any) -> str:
    """A rubrica without its footnote markers ("Truffa (1) (2) (3)", "Procedimento [202]")."""
    return " ".join(_RUBRICA_NOTE_RE.sub("", _FOOTNOTE_RE.sub("", str(rubrica or ""))).split())


def _heading_like(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(_HEADING_RE.match(text)) or (len(letters) > 10 and sum(c.isupper() for c in letters) > 0.7 * len(letters))


def commi_of(text: str = "", parts: Optional[Sequence[str]] = None, how: Optional[str] = None,
             suffixed: bool = False) -> List[Tuple[Optional[Any], str]]:
    """[(k, text)] of an article's real commi (comma_material.comma_numbers: a list item joins its comma; k None for a
    text with no commi to count, then one entry), headings ("CAPO VIII ...") left out. v67: suffixed, a later comma
    («1-bis. …») its own entry, k "1-bis" (comma_material.comma_labels) -- the procedural store's commi."""
    from . import comma_material
    if parts is None:
        parts, how = comma_material.article_parts(clean_text(text))
    else:                       # v56: each part cleaned; the parts after an editorial note dropped
        kept = []
        for part in parts:
            noted = bool(_NOTE_RE.search(str(part or "")))
            if clean_text(part):
                kept.append(clean_text(part))
            if noted:
                break
        parts = kept
    if not parts:
        return []
    numbers = (comma_material.comma_labels if suffixed else comma_material.comma_numbers)(parts, how)
    if all(n is None for n in numbers):
        whole = _TRAILING_HEADING_RE.sub("", " ".join(" ".join(parts).split()))
        return [] if not whole or _heading_like(whole) else [(None, whole)]
    grouped: Dict[Any, List[str]] = {}
    for n, part in zip(numbers, parts):
        grouped.setdefault(n or 1, []).append(_COMMA_HEAD_RE.sub("", part.strip(), count=1))
    out = [(k, _TRAILING_HEADING_RE.sub("", " ".join(" ".join(ps).split())))
           for k, ps in sorted(grouped.items(), key=lambda kv: comma_material.comma_order(kv[0]))]
    return [(k, t) for k, t in out if t and not _heading_like(t)]


def row_commi(row: Dict[str, Any], suffixed: bool = False) -> List[Tuple[Optional[Any], str]]:
    """An article row's commi: COMMA_MATERIAL's parts when it holds them, else its text split (v67: suffixed, see
    commi_of)."""
    from . import comma_material
    if row.get(comma_material.COMMI_KEY):
        return commi_of(parts=row[comma_material.COMMI_KEY], how=row.get(comma_material.FROM_KEY), suffixed=suffixed)
    s = row.get("s") or {}
    return commi_of(str(s.get("plain_text") or s.get("abstract") or ""), suffixed=suffixed)


def operative(commi: Sequence[Tuple[Optional[int], str]], client: Set[str], counter: Set[str]
              ) -> Tuple[List[Tuple[Optional[int], str]], str]:
    """(the commi selected, how). "role": a comma gives the client a right or binds the other side (score > 0) -- it is
    selected with the commi naming no party's role (1170 co. 2's requisites, 1054 co. 2's divisione), and those naming a
    role the other way (1067 co. 1 for the dominant owner) are left out; "named": no comma scores, the ones scoring
    below 0 left out; "all": every comma, when none names a party's role or none would be left."""
    if not commi:
        return [], "none"
    scored = [(comma_score(t, client, counter), (k, t)) for k, t in commi]
    if any(s > 0 for (s, _), _ in scored):
        chosen = [c for (s, n), c in scored if s > 0 or not n]
        return chosen, "role"
    chosen = [c for (s, _), c in scored if s >= 0]
    if chosen and len(chosen) < len(commi):
        return chosen, "named"
    return list(commi), "all"


# --- 2. the requisites ----------------------------------------------------------------------------------------------

_CONNECTOR_RE = re.compile(r"\b(salvo\s+che|salvo|sempre\s+che|purch[ée]|qualora|quando|in\s+caso\s+di|"
                           r"nel\s+caso\s+(?:di|in\s+cui)|nei\s+casi\s+in\s+cui|"
                           r"se\b(?!\s+(?:ne|stess\w*)\b))\b", re.IGNORECASE)   # v76: «quando la cosa» an article;
                           # v76b: «se la restituzione» too (the code's «se la» is an article; «se ne», «se stesso» not);
                           # «in caso di», «nel caso in cui» are conditions
# where the main clause starts after a leading condition ("Se il fondo ..., il proprietario ha diritto")
_MAIN_AFTER_RE = re.compile(r",\s+(?=(?:il|lo|la|l['’]|i|gli|le|questa|questo|questi|queste|esso|essa|essi|esse|egli|ella|si|"
                            r"non|è|sono|può|possono|deve|devono|ha|hanno|chi|tale|tali|ciascun\w*|ognuno|nessun\w*)\b)",
                            re.IGNORECASE)
_ENUM_SPLIT_RE = re.compile(r"\s*;\s*|,?\s+e\s+(?=che\b)|,\s+e\s+(?=(?:non|è|sia|siano|sono|abbia|abbiano|ha|hanno)\b)|"
                            r",?\s+nonché\s+|"
                            r"\s+(?=(?:[a-z]|\d{1,2})\)\s)", re.IGNORECASE)
_ITEM_MARK_RE = re.compile(r"^(?:[a-z]|\d{1,2})\)\s+")


def _clean_item(text: str) -> str:
    return _ITEM_MARK_RE.sub("", text.strip(" ,;:.")).strip(" ,;:.")


def requisites(text: str) -> List[str]:
    """The comma's conditions, each in its own words: the text after each quando / se / purché / sempre che / salvo
    (che) / qualora to the next one (a leading condition to where the main clause starts), split on its enumerations.
    "salvo" items keep the word (an exception). v56: a comma with none of those words is a definition or a penalty --
    no requisites (its comma's line); a clause longer than REQUISITE_CHARS neither."""
    text = " ".join(_COMMA_HEAD_RE.sub("", clean_text(text).strip(), count=1).split())
    if DEFINITION_RE.search(text):          # v57: a definition or a principle (42, 43 c.p.): the comma's line only
        return []
    marks = list(_CONNECTOR_RE.finditer(text))
    pieces: List[str] = []
    if not marks:
        return []
    for i, m in enumerate(marks):
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        segment = text[m.end():stop]
        dot = re.search(r"\.(?:\s|$)", segment)
        if dot:
            segment = segment[:dot.start()]
        cut = not (i + 1 < len(marks) and not dot)          # the segment ends the text, not at the next connector
        leading = m.start() == 0 or re.search(r"[.;:]\s*$", text[:m.start()])
        if leading:
            main = _MAIN_AFTER_RE.search(segment)
            if main and len(segment[:main.start()].split()) >= 3:
                segment = segment[:main.start()]
        word = " ".join(m.group(1).lower().split())
        if word.startswith(("in caso", "nel caso di")):          # v76b: a noun-phrase condition ends where the main
            main = _NP_VERB_RE.search(segment)                   # clause's subject and verb begin («In caso di recesso
            if main:                                             # il cliente ha diritto …»)
                segment = segment[:main.start()]
        items = _ENUM_SPLIT_RE.split(segment)
        pending = ""                # v79.6: an item ending on «o / ovvero / oppure» is a branch of the next one
        join_next = ""
        for j, raw in enumerate(items):
            item = _clean_item(raw)
            if pieces and j and _DISJ_HEAD_RE.match(raw.strip(" ,;:")):     # v79.6: «…; ovvero b) …» -- a branch
                head = re.match(rf"^\s*({_DISJ})\b\s*", raw.strip(" ,;:"), re.IGNORECASE)
                rest = _clean_item(raw.strip(" ,;:")[head.end():])
                if rest:
                    pieces[-1] = f"{pieces[-1]} {head.group(1).lower()} {rest}"
                else:
                    join_next = head.group(1).lower()        # a bare «ovvero»: the next item is the branch
                continue
            if join_next and pieces:
                pieces[-1], join_next = f"{pieces[-1]} {join_next} {item}", ""
                continue
            if pending:
                item, pending = f"{pending} {item}", ""
            if _DISJ_END_RE.search(raw.strip(" ,;:")) and j + 1 < len(items):
                pending = item
                continue
            if _DANGLING_RE.search(raw.strip(" ,;:")) or PENALTY_RE.search(item) or not requisite_kept(item) \
                    or (cut and j == len(items) - 1 and raw.rstrip().endswith(",")):
                continue                    # v57: cut on a conjunction, or on a comma where the text ends; the punishment
            if len(item.split()) >= 3 and len(item) <= REQUISITE_CHARS:
                pieces.append(f"{word} {item}" if word.startswith("salvo") else item)
            elif len(item) > REQUISITE_CHARS and not word.startswith("salvo") and "," in item:     # v76b: a long
                pieces += [x.strip(" ,;.") for x in re.split(r"\s*,\s*", item)       # condition, cut on its commas
                           if len(x.split()) >= 3 and len(x) <= REQUISITE_CHARS and requisite_kept(x)]
    return list(dict.fromkeys(pieces))


# --- 3. the block for the writer ------------------------------------------------------------------------------------

def _label(ref: Sequence[str], k: Optional[int]) -> str:
    from .plan_then_write import _code_key
    return f"art. {ref[0]}" + (f", co. {k}," if k else "") + f" {_code_key(ref[1])}"


def _rows_by_ref(article_rows: Iterable[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    out: Dict[tuple, Dict[str, Any]] = {}
    for row in article_rows or []:
        ref = row.get("_article") if isinstance(row, dict) else None
        if ref and tuple(ref) not in out and row_commi(row):
            out[(str(ref[0]), ref[1])] = row
    return out


def _find_row(rows: Dict[tuple, Dict[str, Any]], ref: Sequence[str]) -> Optional[Dict[str, Any]]:
    from .plan_then_write import _same_ref
    ref = (str(ref[0]), ref[1])
    return rows.get(ref) or next((r for k, r in rows.items() if _same_ref(k, ref)), None)


def adverse_refs(plan: Dict[str, Any]) -> List[tuple]:
    """The plan's articles of the other side: the action gate's and ROLE_FIT's (norme_contrarie), with RENDER_COMPUTED
    the objections' (obiezioni_norme)."""
    labels = list(plan.get("norme_contrarie") or []) + [o["article"] for o in plan.get("obiezioni_norme") or []]
    return [tuple(lab.split(" ", 1)) for lab in dict.fromkeys(labels) if " " in lab]


def article_rules(plan: Dict[str, Any], article_rows: List[Dict[str, Any]], analysis: Dict[str, Any],
                  refs: Optional[Sequence[tuple]] = None) -> List[Dict[str, Any]]:
    """Per article of the plan (or refs) with text: {ref, label, commi, selected, how, requisites: {k: [...]},
    adverse}; an adverse article's commi chosen for the other side."""
    from .plan_then_write import _plan_refs, _same_ref
    client, counter = sides(analysis, plan)
    adverse = adverse_refs(plan)
    wanted = list(refs) if refs is not None else list(dict.fromkeys(_plan_refs(plan) + adverse))
    rows = _rows_by_ref(article_rows)
    out = []
    for ref in wanted:
        row = _find_row(rows, ref)
        if row is None:
            continue
        commi = row_commi(row)
        against = any(_same_ref(tuple(ref), a) for a in adverse)
        chosen, how = operative(commi, counter, client) if against else operative(commi, client, counter)
        if commi and (offence(ref) or (not _is_cp(ref) and penal_provision(commi[0][1]))):   # v58: an offence -- co.
            # 1's elements, never its aggravanti's commi; v76b: a special law's penal provision too
            chosen, how = [commi[0]], "offence"
            found = _fitting(offence_elements(commi[0][1]))
            if not found:               # v76: the general penal splitter («chiunque …, con …, in modo da …»); a
                clauses = penal_clauses(commi[0][1])      # one-clause offence (612: «minaccia ad altri un ingiusto
                found = _fitting(clauses) or (            # danno») its conduct alone, of three words at least
                    clauses if len(clauses) == 1 and len(clauses[0].split()) >= 3 else [])
                how = "offence_clauses"
            reqs = {commi[0][0]: found} if found else {}
        else:
            reqs = {k: r for k, t in chosen for r in [civil_split(t)] if r}       # v75, v76b: civil too
        # v59: a chosen comma extending other articles' penalties (320, 321, 319-ter c.p.) has its own line, never none
        referring = [k for k, t in chosen if k not in reqs and refers_penalties(t)]
        out.append({"ref": (str(ref[0]), ref[1]), "label": f"{ref[0]} {ref[1]}", "commi": commi,
                    "selected": [k for k, _ in chosen], "how": how, "requisites": reqs, "adverse": against,
                    "referring": referring, "client": sorted(client), "counter": sorted(counter)})
    return out


def _entry(rule: Dict[str, Any], party: str, focus: str = "",
           status: Optional[Callable[[str], str]] = None) -> str:
    from . import comma_material
    ref = rule["ref"]
    chosen = [(k, t) for k, t in rule["commi"] if k in rule["selected"]] or rule["commi"]
    whole = sum(len(t) + 12 for _, t in chosen)
    left: List[int] = []
    if whole > ARTICLE_CHARS and len(chosen) > 1:
        keep = {chosen[i - 1][0] for i, _ in comma_material.select([t for _, t in chosen], ARTICLE_CHARS, focus)}
        left = [k for k, _ in chosen if k not in keep and k is not None]
        chosen = [(k, t) for k, t in chosen if k in keep]
    why = {"role": f"il comma che opera {'per la controparte' if rule['adverse'] else f'per {party}'}",
           "named": "i commi che nominano la posizione delle parti",
           "all": "nessun comma riguarda una sola parte: tutti i commi"}.get(rule["how"], "")
    head = f"- art. {ref[0]} {_code_of(ref)}" + (" (norma della controparte)" if rule["adverse"] else "") + \
           (f" — {why}" if why else "") + ":"
    lines = [head]
    for k, t in chosen:
        reqs = rule["requisites"].get(k)
        tag = f"co. {k}: " if k else ""
        if reqs:
            lines.append(f"  {tag}requisiti (dal testo del comma):")
            lines += [f"    {i}. «{r}»" + (f" — {status(r)}" if status else "") for i, r in enumerate(reqs, 1)]
        else:
            lines.append(f"  {tag}«{t}»")
    if left:
        lines.append(f"  [commi non riportati: {', '.join(map(str, left))}]")
    return "\n".join(lines)


def _code_of(ref: Sequence[str]) -> str:
    from .plan_then_write import _code_key
    return _code_key(ref[1])


def material_block(plan: Dict[str, Any], article_rows: List[Dict[str, Any]], analysis: Dict[str, Any],
                   refs: Optional[Sequence[tuple]] = None, status: Optional[Callable[[str], str]] = None,
                   title: str = TITLE, rule_text: str = RULE_TEXT, force: bool = False) -> str:
    """REGOLE DELLE NORME DEL PIANO for a section (its refs, else every plan article), with RULE_TEXT; "" with the flag
    off or no article with text. INTERACTIVE_FLOW: status(requisite) -> "provato" / "da provare", written after each
    numbered requisite; its own title and rule; force: whatever the flag."""
    if not (enabled() or force) or not plan:
        return ""
    from .comma_material import plan_focus
    rules = article_rules(plan, article_rows, analysis, refs)
    if not rules:
        return ""
    party = plan.get("parte_assistita") or "la parte assistita"
    focus = plan_focus(plan)
    entries, used = [], 0
    for rule in rules:
        entry = _entry(rule, party, focus.get(rule["ref"], ""), status)
        if used + len(entry) > BLOCK_CHARS and entries:
            break
        entries.append(entry)
        used += len(entry)
    return f"{title}:\n" + "\n".join(entries) + f"\n{rule_text}"


def material_log(plan: Dict[str, Any], article_rows: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
    rules = article_rules(plan, article_rows, analysis)
    return {"client": rules[0]["client"] if rules else [], "counter": rules[0]["counter"] if rules else [],
            "articles": [{"article": r["label"], "commi": [k for k, _ in r["commi"]], "selected": r["selected"],
                          "how": r["how"], "adverse": r["adverse"],
                          "requisites": {str(k): len(v) for k, v in r["requisites"].items()}} for r in rules]}


# --- 4. the check after writing -------------------------------------------------------------------------------------

# v53 corrections: "ai sensi" out (on v52, 106 of the 184 replacements were applications citing the article that way)
_RULE_VERB_RE = re.compile(r"\b(?:prevede|prevedono|stabilisce|stabiliscono|dispone|dispongono|esclude|escludono|vieta|"
                           r"vietano|richiede|richiedono|precisa|precisano|sancisce|sanciscono|consente|consentono|"
                           r"riconosce|riconoscono|impone|impongono)\b", re.IGNORECASE)
# INTERACTIVE_FLOW: the topic answer's rule sentences also by these verbs (the exemptions and presumptions)
EXTENDED_VERB_RE = re.compile(_RULE_VERB_RE.pattern[:-3] + r"|esenta|esentano|esonera|esonerano|dispensa|dispensano|"
                              r"presume|presumono|libera|liberano)\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r"«([^»]{12,})»|“([^”]{12,})”")
_STOP = {"della", "delle", "dello", "degli", "dell", "nella", "nelle", "nello", "negli", "dalla", "dalle", "dallo",
         "dagli", "sulla", "sulle", "sullo", "sugli", "alla", "alle", "allo", "agli", "questa", "questo", "quale",
         "quali", "sono", "essere", "viene", "vengono", "anche", "come", "ogni", "tutte", "tutti", "proprio", "propria",
         "infatti", "inoltre", "quindi", "pertanto", "norma", "articolo", "comma", "codice", "civile", "penale",
         "procedura", "prevede", "prevedono", "stabilisce", "stabiliscono", "dispone", "dispongono", "esclude",
         "escludono", "vieta", "vietano", "richiede", "richiedono", "sensi", "secondo", "previsto", "prevista",
         "stabilito", "stabilita", "disposto", "disciplina", "sancisce"}
_CITE_TOKENS_RE = re.compile(r"\bartt?\.\s*[\w-]+(?:\s*(?:,|e)\s*[\w-]+)*|\b(?:co\.|comma|commi)\s*\d+|"
                             r"\bc\.\s?p\.\s?c\.|\bc\.\s?p\.\s?p\.|\bc\.\s?c\.|\bc\.\s?p\.|\bd\.\s?lgs\.?\s*[\d/]+", re.IGNORECASE)


def content_stems(text: str) -> Set[str]:
    words = re.findall(r"[a-zà-ü]{4,}", _CITE_TOKENS_RE.sub(" ", (text or "").lower()))
    return {w[:6] for w in words if w not in _STOP}


def share(sentence: str, comma: str) -> float:
    """The share of the sentence's content words (stems of 4+ letters, its citation and the rule verbs left out) in the
    comma; 1.0 for a sentence with none."""
    mine = content_stems(sentence)
    return len(mine & content_stems(comma)) / len(mine) if mine else 1.0


def party_names(analysis: Dict[str, Any], plan: Optional[Dict[str, Any]] = None) -> List[str]:
    """The parties' names (the facts', the plan's client, the defendants'), each word of 3+ letters starting upper case."""
    signals = (analysis or {}).get("fact_signals") or {}
    raw = [str(p.get("name") or "") for p in _parties(analysis)] + _client_names(analysis or {}, plan)
    raw += [str(d.get("name") or "") for d in signals.get("defendants") or [] if isinstance(d, dict)]
    out = []
    for name in raw:
        for word in re.findall(r"\b[A-ZÀ-Ü][\w'’-]{2,}", name):
            if word.lower() not in {"comune", "società", "signor", "signora", "dott", "avv"} and word not in out:
                out.append(word)
    return out


def _normal(text: str) -> str:
    """Lower case, every mark of punctuation a space: "contratto, se" and "contratto se" are the same words."""
    return " ".join(re.sub(r"[^\w]+", " ", (text or "").lower()).split())


def _commi_named(clause: str, ref: tuple) -> Set[int]:
    from . import comma_material
    from .plan_then_write import _same_ref
    named = {k for key, ks in comma_material.named_commi(clause).items() if _same_ref(key, ref) for k in ks}
    m = re.search(r"\bart\.\s*" + re.escape(str(ref[0])) + r"\s*,?\s*(?:co\.|comma)\s*(\d{1,2})\b", clause, re.IGNORECASE)
    return named | ({int(m.group(1))} if m else set())


def _quotes(clause: str) -> List[str]:
    return [a or b for a, b in _QUOTED_RE.findall(clause)]


def _holds(quote: str, text: str) -> bool:
    """The quotation is the text's: its words in it, punctuation aside, or (a word changed: "s'intende" / "si intende")
    nine tenths of its content words."""
    q = _normal(quote)
    if q and q in _normal(text):
        return True
    words = content_stems(quote)
    return len(words) >= 4 and len(words & content_stems(text)) >= 0.9 * len(words)


def _whole(rule: Dict[str, Any]) -> str:
    return " ".join(t for _, t in rule["commi"])


def judge(clause: str, rule: Dict[str, Any]) -> Dict[str, Any]:
    """{"ok", "why", "comma", "share", "fix"} for the clause of a rule sentence stating rule's article. A quotation must
    be the text of the comma the clause names (any comma when it names none); without one, SHARE of the clause's content
    words must be in the comma it names, else in the article's text. fix: "quote" (the clause's sentence replaced by the
    comma's quotation) or "renumber" (a quotation of another comma: its own number)."""
    commi = rule["commi"]
    named = _commi_named(clause, rule["ref"])
    quotes = _quotes(clause)
    if quotes:
        for quote in quotes:
            holders = [k for k, t in commi if _holds(quote, t)]
            if not holders:
                if _holds(quote, _whole(rule)):     # across two commi
                    continue
                return {"ok": False, "why": "quotation not in the article's text", "comma": _best(clause, rule)[0],
                        "share": None, "fix": "quote"}
            if named and not (set(holders) & named) and all(h is not None for h in holders):
                return {"ok": False, "why": f"quotation of co. {holders[0]} cited as co. {min(named)}",
                        "comma": holders[0], "share": None, "fix": "renumber", "named": min(named)}
        return {"ok": True, "why": "quotes its comma", "comma": None, "share": None, "fix": ""}
    own = [(k, t) for k, t in commi if k in named]
    if own:
        k, value = own[0][0], share(clause, own[0][1])
        where = f"co. {k}"
    else:
        k, value = _best(clause, rule)[0], share(clause, _whole(rule))
        where = "the article's text"
    ok = value >= SHARE
    return {"ok": ok, "why": f"{value:.0%} of its words in {where}", "comma": k, "share": round(value, 2),
            "fix": "" if ok else "quote"}


def _best(clause: str, rule: Dict[str, Any]) -> Tuple[Optional[int], float]:
    """The comma sharing most words with the clause (ties: a selected one, then the first) and that share."""
    best = max(rule["commi"], key=lambda c: (share(clause, c[1]), c[0] in rule["selected"], -(c[0] or 0)))
    return best[0], share(clause, best[1])


def quotation(rule: Dict[str, Any], k: Optional[int]) -> str:
    """"L'art. 1067, co. 2, c.c. dispone: «...»" for comma k (a selected comma when k is None or unknown)."""
    if k is None or not any(c[0] == k for c in rule["commi"]):
        k = rule["selected"][0] if rule["selected"] else rule["commi"][0][0]
    text = next(t for c, t in rule["commi"] if c == k).rstrip()
    end = "" if text.endswith((".", "!", "?")) else "."
    return f"L'{_label(rule['ref'], k)} dispone: «{text}{end}»"


def _renumbered(sentence: str, ref: tuple, wrong: int, right: int) -> str:
    """The sentence with the article's comma number `wrong` written as `right` ("co. 1" / "comma 1" / "primo comma")."""
    from . import comma_material
    words = {v: k for k, v in comma_material._ORDINALS.items()}
    out = re.sub(rf"\b(co\.|comma)\s*{wrong}\b", lambda m: f"{m.group(1)} {right}", sentence, count=1)
    if out == sentence and wrong in words and right in words:
        out = re.sub(rf"\b{words[wrong]}\s+comma\b", f"{words[right]} comma", sentence, count=1, flags=re.IGNORECASE)
    return out


def rule_clauses(sentence: str, rules: List[Dict[str, Any]], documents: List[Dict[str, Any]]
                 ) -> List[Tuple[Dict[str, Any], str]]:
    """[(rule, clause)]: for each rule verb of the sentence, the plan article whose citation is nearest to it, with the
    sentence's text between the other articles' citations around them -- "L'art. 1223 c.c. prevede ...; l'art. 1226
    c.c. dispone ..." is two clauses; "... in forza dell'art. 114 c.p.a. e dell'art. 2931 c.c., che prevede ..." states
    art. 2931's rule, not art. 114's. A verb whose nearest citation is no plan article states no plan rule."""
    from .plan_then_write import _citations, _same_ref
    cites = [(s, e, refs) for s, e, refs in _citations(sentence, documents)]
    if not cites:
        return []
    out: List[Tuple[Dict[str, Any], str]] = []
    seen = set()
    for verb in _RULE_VERB_RE.finditer(sentence):
        def gap(c):
            return verb.start() - c[1] if c[1] <= verb.start() else c[0] - verb.end() if c[0] >= verb.end() else 0
        near = min(cites, key=lambda c: (gap(c), c[0] > verb.start()))
        rule = next((r for r in rules for ref in near[2] if _same_ref(r["ref"], ref)), None)
        if rule is None or (rule["label"], near[0]) in seen:
            continue
        seen.add((rule["label"], near[0]))
        own = [c for c in cites if any(_same_ref(rule["ref"], ref) for ref in c[2])]
        others = [c for c in cites if c not in own]
        low = min(near[0], verb.start())
        high = max(near[1], verb.end())
        lo = max([c[1] for c in others if c[1] <= low] + [0])
        hi = min([c[0] for c in others if c[0] >= high] + [len(sentence)])
        stops = [m.end() for m in re.finditer(r"[;:]\s", sentence[lo:low])]      # "...; l'art. 1054 c.c. esclude"
        lo += stops[-1] if stops else 0
        stop = re.search(r";\s", sentence[high:hi])
        hi = high + stop.start() if stop else hi
        clause = sentence[lo:hi]
        if lo:          # the clause opens past another article's citation: from its first word ("; l'art.", ", che")
            clause = re.sub(r"^[^\w«“]*(?:(?:c\.\s?[cp]\.(?:\s?[cp]\.)?|(?:e|ed)\b|,)\s*)*", "", clause)
        out.append((rule, clause))
    return out


_PARTY_WORDS_RE = re.compile(r"\bcontroparte\b|\bassistit[oaie]\b|\bparte assistita\b|\bcliente\b", re.IGNORECASE)


def _as_sentence(clause: str) -> str:
    """A clause that passed, standing alone: no leading "ma" / "e" / list mark, no trailing ", ma", a capital, a stop."""
    text = re.sub(r"^\s*(?:[-•*]\s+|\d+[.)]\s+)?(?:(?:ma|e|ed|mentre|inoltre)\s+)?", "", clause.strip())
    tail = re.compile(r"[\s,;:]*(?:\b(?:ma|e|ed|mentre|che)|\b(?:l|dell|all|nell|dall|sull)['’]|\b(?:il|lo|la|del|dello|della|"
                      r"dal|al|nel|ai\s+sensi\s+dell['’]|in\s+base\s+all['’]))\s*$", re.IGNORECASE)
    while tail.search(text) and tail.search(text).group(0).strip(" ,;:"):
        text = tail.sub("", text)
    text = text.rstrip(" ,;:")
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "»", "!", "?")) else text + "."


def rule_sentence(sentence: str, names: Sequence[str], verbs: Optional[re.Pattern] = None) -> bool:
    """A rule verb (verbs, else _RULE_VERB_RE) and no party named -- by its name, or as "la controparte", "l'assistito",
    "il cliente" (a sentence naming a party applies the rule or argues, it does not state it)."""
    if not (verbs or _RULE_VERB_RE).search(sentence) or _PARTY_WORDS_RE.search(sentence):
        return False
    return not any(re.search(rf"\b{re.escape(n)}\b", sentence) for n in names)


def check_text(title: str, text: str, rules: List[Dict[str, Any]], names: Sequence[str],
               documents: List[Dict[str, Any]], verbs: Optional[re.Pattern] = None,
               quote: bool = True) -> Tuple[str, List[Dict[str, Any]], int]:
    """(text, log, rule clauses judged) of one section: each rule clause of a plan article judged (judge); a sentence with
    a failing one replaced by the quotation of each failing article's comma (a renumbering when only the comma number
    is wrong); the sentence after it untouched. quote False (v60q, single shot: articles are references, never quoted
    text): no quotation -- the failing clauses go, the clauses that passed stay as their own sentences; nothing passed,
    the sentence goes ("dropped": with a deferring ledger a proposal, the record check's to confirm)."""
    from .plan_then_write import _sentence_spans
    replace: Dict[Tuple[int, int], str] = {}
    log: List[Dict[str, Any]] = []
    judged = 0
    for s, e in _sentence_spans(text):
        sentence = text[s:e]
        if not rule_sentence(sentence, names, verbs):
            continue
        quoted, renumbered, failed, passed = [], sentence, [], []
        for rule, clause in rule_clauses(sentence, rules, documents):
            judged += 1
            verdict = judge(clause, rule)
            if verdict["ok"]:
                passed.append(clause)
                continue
            failed.append((rule, verdict))
            if verdict["fix"] == "renumber":
                renumbered = _renumbered(renumbered, rule["ref"], verdict["named"], verdict["comma"])
            else:
                quoted.append(quotation(rule, verdict["comma"]) if quote else "")
        if not failed:
            continue
        lead = re.match(r"^\s*(?:[-•*]\s+|\d+[.)]\s+)", sentence)
        kept = [_as_sentence(c) for c in passed] if quoted else []     # a clause that passed stays, as its own sentence
        new_text = " ".join(x for x in kept + list(dict.fromkeys(quoted)) if x)
        now = ((lead.group(0) if lead and new_text else "") + new_text) if quoted else \
            (lead.group(0) if lead else "") + renumbered.strip()
        if now.strip() == sentence.strip():
            continue
        replace[(s, e)] = now
        fixed = "quoted" if quote else "kept_passed" if new_text else "dropped"
        for rule, verdict in failed:
            log.append({"section": title, "article": rule["label"], "comma": verdict["comma"], "why": verdict["why"],
                        "share": verdict["share"], "how": "renumbered" if verdict["fix"] == "renumber" else fixed,
                        "sentence": " ".join(sentence.split())[:400], "now": now[:400]})
    for (s, e) in sorted(replace, reverse=True):
        tail = text[e:e + 1]
        text = text[:s] + replace[(s, e)] + ("" if not tail or tail.isspace() else " ") + text[e:]
    return text, log, judged


def check_sections(sections: List[Tuple[str, str]], plan: Dict[str, Any], article_rows: List[Dict[str, Any]],
                   analysis: Dict[str, Any], documents: Optional[List[Dict[str, Any]]] = None,
                   skip: Iterable[str] = (), verbs: Optional[re.Pattern] = None,
                   refs: Optional[Sequence[tuple]] = None, quote: bool = True
                   ) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """(sections, log): check_text on every section but those in skip (a rendered one); verbs and refs as check_text and
    article_rules take them (INTERACTIVE_FLOW: EXTENDED_VERB_RE, the topic's articles); quote as check_text's."""
    rules = article_rules(plan, article_rows, analysis, refs)
    names = party_names(analysis, plan)
    skip = set(skip)
    out, entries, judged = [], [], 0
    for title, text in sections:
        if title in skip or not rules:
            out.append((title, text))
            continue
        new, log, n = check_text(title, text, rules, names, documents or [], verbs, quote)
        out.append((title, new))
        entries += log
        judged += n
    return out, {"judged": judged, "replaced": len(entries), "sentences": len({(e["section"], e["sentence"]) for e in entries}),
                 "entries": entries,
                 "by_article": {a: sum(1 for e in entries if e["article"] == a)
                                for a in dict.fromkeys(e["article"] for e in entries)}}
