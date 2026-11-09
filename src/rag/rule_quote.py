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


_OWN_SUBJECT_RE = re.compile(r"\bchiunque\b|^\W*(?:chi|il pubblico ufficiale|l['’]incaricato)\b", re.IGNORECASE)


def refers_penalties(text: str) -> bool:
    """v59 (v55-l2 585 c.p.: «Nei casi previsti dagli articoli 582, 583 …, la pena è aumentata …»): a comma extending
    other articles' rules, or setting a penalty for the cases other articles provide with no subject of its own
    (never «Chiunque, fuori dei casi indicati nell'articolo 571, maltratta …») -- its own line, no requisite."""
    text = " ".join(_COMMA_HEAD_RE.sub("", clean_text(text or "").strip(), count=1).split())
    return extends(text) or bool(CONSEQUENCE_RE.search(text) and _REFERENCE_RE.search(text)
                                 and not _OWN_SUBJECT_RE.search(text))


def offence(ref: Sequence[str]) -> bool:
    from .plan_then_write import _code_key
    m = re.match(r"\d+", str(ref[0]))
    return bool(m) and _code_key(str(ref[1])) == OFFENCE_BOOK[0] and OFFENCE_BOOK[1] <= int(m.group(0)) <= OFFENCE_BOOK[2]


def offence_elements(text: str) -> List[str]:
    """v58 (v57-p2 640: «Chiunque, con artifizi o raggiri, …» is the requisite set): the elements of an offence's first
    comma -- its words before «è punito», «Chiunque» left out, split on its commas; an element of fewer than two words
    joins the next."""
    whole = " ".join(_COMMA_HEAD_RE.sub("", clean_text(text).strip(), count=1).split())
    if refers_penalties(whole):     # v59: «… si applicano anche a chi …», «Nei casi previsti …»: its own line only
        return []
    body = _PUNISHED_RE.sub("", whole)
    body = re.sub(r"^(?:chiunque|il pubblico ufficiale|l['’]incaricato di un pubblico servizio|chi)\s*,?\s*", "", body,
                  flags=re.IGNORECASE)
    out: List[str] = []
    carry = ""
    for piece in re.split(r"\s*,\s*", body):
        piece = f"{carry} {piece}".strip() if carry else piece.strip()
        if len(piece.split()) < 2:
            carry = piece
            continue
        carry = ""
        if requisite_kept(piece) and len(piece) <= REQUISITE_CHARS:
            out.append(piece.strip(" .;:"))
    # v59 (v58-h7 322: «soggiace, qualora l'offerta o la promessa non sia accettata, alla pena …»): a condition set
    # inside the penalty clause is an element too
    penalty = _PUNISHED_RE.search(whole)
    for piece in re.split(r"\s*,\s*", whole[penalty.start():] if penalty else ""):
        piece = piece.strip(" .;:")
        if re.match(r"(?:qualora|se|purch[ée]|sempre che)\b", piece, re.IGNORECASE) and requisite_kept(piece) \
                and len(piece.split()) >= 3 and len(piece) <= REQUISITE_CHARS:
            out.append(piece)
    return [x for x in out if x]


_DANGLING_RE = re.compile(r"\b(?:e|ed|o|od|ma|né|oppure|ovvero|nonché)\s*$", re.IGNORECASE)


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


def commi_of(text: str = "", parts: Optional[Sequence[str]] = None, how: Optional[str] = None
             ) -> List[Tuple[Optional[int], str]]:
    """[(k, text)] of an article's real commi (comma_material.comma_numbers: a list item joins its comma; k None for a
    text with no commi to count, then one entry), headings ("CAPO VIII ...") left out."""
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
    numbers = comma_material.comma_numbers(parts, how)
    if all(n is None for n in numbers):
        whole = _TRAILING_HEADING_RE.sub("", " ".join(" ".join(parts).split()))
        return [] if not whole or _heading_like(whole) else [(None, whole)]
    grouped: Dict[int, List[str]] = {}
    for n, part in zip(numbers, parts):
        grouped.setdefault(n or 1, []).append(_COMMA_HEAD_RE.sub("", part.strip(), count=1))
    out = [(k, _TRAILING_HEADING_RE.sub("", " ".join(" ".join(ps).split()))) for k, ps in sorted(grouped.items())]
    return [(k, t) for k, t in out if t and not _heading_like(t)]


def row_commi(row: Dict[str, Any]) -> List[Tuple[Optional[int], str]]:
    """An article row's commi: COMMA_MATERIAL's parts when it holds them, else its text split."""
    from . import comma_material
    if row.get(comma_material.COMMI_KEY):
        return commi_of(parts=row[comma_material.COMMI_KEY], how=row.get(comma_material.FROM_KEY))
    s = row.get("s") or {}
    return commi_of(str(s.get("plain_text") or s.get("abstract") or ""))


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

_CONNECTOR_RE = re.compile(r"\b(salvo\s+che|salvo|sempre\s+che|purch[ée]|qualora|quando|se)\b"
                           r"(?!\s+(?:ne|lo|la|li|le|stess\w*)\b)", re.IGNORECASE)
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
        items = _ENUM_SPLIT_RE.split(segment)
        for j, raw in enumerate(items):
            item = _clean_item(raw)
            if _DANGLING_RE.search(raw.strip(" ,;:")) or PENALTY_RE.search(item) or not requisite_kept(item) \
                    or (cut and j == len(items) - 1 and raw.rstrip().endswith(",")):
                continue                    # v57: cut on a conjunction, or on a comma where the text ends; the punishment
            if len(item.split()) >= 3 and len(item) <= REQUISITE_CHARS:
                pieces.append(f"{word} {item}" if word.startswith("salvo") else item)
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
        if offence(ref) and commi:         # v58: an offence -- co. 1's elements, never its aggravanti's commi
            chosen, how = [commi[0]], "offence"
            reqs = {commi[0][0]: r for r in [offence_elements(commi[0][1])] if REQUISITES_MIN <= len(r) <= REQUISITES_MAX}
        else:
            reqs = {k: r for k, t in chosen for r in [requisites(t)] if REQUISITES_MIN <= len(r) <= REQUISITES_MAX}
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
               documents: List[Dict[str, Any]], verbs: Optional[re.Pattern] = None) -> Tuple[str, List[Dict[str, Any]], int]:
    """(text, log, rule clauses judged) of one section: each rule clause of a plan article judged (judge); a sentence with
    a failing one replaced by the quotation of each failing article's comma (a renumbering when only the comma number
    is wrong); the sentence after it untouched."""
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
                quoted.append(quotation(rule, verdict["comma"]))
        if not failed:
            continue
        lead = re.match(r"^\s*(?:[-•*]\s+|\d+[.)]\s+)", sentence)
        kept = [_as_sentence(c) for c in passed] if quoted else []     # a clause that passed stays, as its own sentence
        now = (lead.group(0) if lead else "") + (" ".join(kept + list(dict.fromkeys(quoted))) if quoted
                                                 else renumbered.strip())
        if now.strip() == sentence.strip():
            continue
        replace[(s, e)] = now
        for rule, verdict in failed:
            log.append({"section": title, "article": rule["label"], "comma": verdict["comma"], "why": verdict["why"],
                        "share": verdict["share"], "how": "renumbered" if verdict["fix"] == "renumber" else "quoted",
                        "sentence": " ".join(sentence.split())[:400], "now": now[:400]})
    for (s, e) in sorted(replace, reverse=True):
        tail = text[e:e + 1]
        text = text[:s] + replace[(s, e)] + ("" if not tail or tail.isspace() else " ") + text[e:]
    return text, log, judged


def check_sections(sections: List[Tuple[str, str]], plan: Dict[str, Any], article_rows: List[Dict[str, Any]],
                   analysis: Dict[str, Any], documents: Optional[List[Dict[str, Any]]] = None,
                   skip: Iterable[str] = (), verbs: Optional[re.Pattern] = None,
                   refs: Optional[Sequence[tuple]] = None) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """(sections, log): check_text on every section but those in skip (a rendered one); verbs and refs as check_text and
    article_rules take them (INTERACTIVE_FLOW: EXTENDED_VERB_RE, the topic's articles)."""
    rules = article_rules(plan, article_rows, analysis, refs)
    names = party_names(analysis, plan)
    skip = set(skip)
    out, entries, judged = [], [], 0
    for title, text in sections:
        if title in skip or not rules:
            out.append((title, text))
            continue
        new, log, n = check_text(title, text, rules, names, documents or [], verbs)
        out.append((title, new))
        entries += log
        judged += n
    return out, {"judged": judged, "replaced": len(entries), "sentences": len({(e["section"], e["sentence"]) for e in entries}),
                 "entries": entries,
                 "by_article": {a: sum(1 for e in entries if e["article"] == a)
                                for a in dict.fromkeys(e["article"] for e in entries)}}
