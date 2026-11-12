"""v60h (9): roles are not names. A party the case names by role only («il lavoratore», «la società», «l'acquirente»)
is written by its role with the article, lower case, in every string the user reads -- round 1's opening and parties,
the restatement, «Chi assisti» options, the parere, the procedural section -- never capitalised as a name
(«Lavoratore ha firmato», «il parere per Società»).

is_role(name, case): the name's first content word is a role (ROLE_STEMS) -- or, the case given, a common noun the case
itself writes in lower case («un idraulico», «il produttore»; a proper name is never lower case there) -- and no other
word is capitalised (a proper name: «Società Alfa», «Ditta Rossi», «Mario» stay names). label(name): the role with its article; any other name as given.
contract(text): «di il» -> «del», «a l'» -> «all'», … after a slot filled with a label. lower_roles(text, names): in a
model's text, each capitalised role name lower case with its article (and the preposition before it contracted).

v60i (7): the articled preposition in every party reference -- «dell'inquilino», «all'acquirente», never «di
l'inquilino», «a l’acquirente» (either apostrophe) nor «dell'l'inquilino» (a capitalised role after a contracted
preposition, «dell'Inquilino», keeps the one article); contract_roles(text, names) contracts a role phrase the text
already writes in lower case, and lower_roles runs it.
"""

import re
from typing import Iterable, List, Sequence

ROLE_STEMS = (
    "lavorat", "dipendent", "operai", "datore", "datrice", "aziend", "societ", "ditt", "impres", "acquirent", "compratr",
    "comprator", "venditor", "venditric", "consumator", "client", "controparte", "committent", "appaltator", "conduttor",
    "locator", "locatric", "inquilin", "proprietari", "vicin", "condomin", "amministrator", "ente", "banca", "istitut",
    "assicurazion", "assicurat", "compagni", "medic", "ospedal", "struttur", "truffator", "ignot", "sconosciut", "autor",
    "utent", "professionist", "fornitor", "prestator", "prepost", "dirigent", "colleg", "responsabil", "titolar",
    "gestor", "persona", "vittim", "imputat", "indagat", "querelant", "denunciant", "attor", "attric", "convenut",
    "ricorrent", "resistent", "ered", "coniug", "mogli", "marit", "madre", "padre", "figli", "fratell", "sorell",
    "minor", "pazient", "passegger", "conducent", "pedon", "ciclist", "automobilist", "motociclist", "ospit", "tecnic",
    "istruttor", "insegnant", "student", "mittent", "destinatari", "beneficiari", "mandatari", "mandant", "agent",
    "rappresentant", "soci", "creditor", "debitor", "garant", "fideiussor", "mutuatari", "mutuant", "comodatari",
    "comodant", "usufruttuari", "convivent", "compagn", "parte", "offes", "danneggiat", "assistit", "comune",
    "amministrazion", "uffici", "negozi", "piattaform", "sito", "corrier", "spedizionier", "gestore")
_ARTICLES = {"il", "lo", "la", "l", "i", "gli", "le", "un", "uno", "una", "sig", "sigra", "signor", "signora", "dott",
             "dottssa"}
_LINKS = {"di", "del", "della", "dello", "dei", "degli", "delle", "da", "in", "con", "e", "ed", "per", "a", "al", "alla"}
_FEMININE_E = {"parte", "persona", "madre", "moglie", "gente", "fonte", "società", "comunità", "autorità", "università"}


def _words(name: str) -> List[str]:
    return re.findall(r"[^\W\d_]+", str(name or ""))


def _role_word(word: str) -> bool:
    low = word.lower()
    return any(low.startswith(stem) for stem in ROLE_STEMS)


def common_in(word: str, case: str) -> bool:
    """The case writes the word in lower case, and capitalised only where a sentence starts: a common noun."""
    if not word or not case:
        return False
    low = word.lower()
    if not re.search(rf"(?<!\w){re.escape(low)}(?!\w)", case):
        return False
    for m in re.finditer(rf"(?<!\w){re.escape(low[:1].upper() + low[1:])}(?!\w)", case):
        before = case[:m.start()].rstrip()
        if before and not before.endswith((".", "!", "?", ":", "\n", "«", '"')):
            return False
    return True


def is_role(name: str, case: str = "") -> bool:
    """A party named by its role only: its first content word a role (or a common noun of the case), no other word
    capitalised."""
    words = [w for w in _words(name) if w.lower() not in _ARTICLES]
    if not words or not (_role_word(words[0]) or common_in(words[0], case)):
        return False
    return not any(w[:1].isupper() for w in words[1:] if w.lower() not in _LINKS)


def _feminine(noun: str) -> bool:
    low = noun.lower()
    return (low.endswith(("a", "à", "ione", "trice", "ice", "tù")) and low not in ("collega", "pediatra", "psichiatra")) \
        or low in _FEMININE_E


def _article(phrase: str) -> str:
    """The definite article of a lower-case phrase, by its first word (a word in -i is a plural; -trici, -ici feminine)."""
    first = (phrase.split(" ") or [""])[0]
    vowel = first[:1] in "aeiouàèéìòù"
    impure = bool(re.match(r"s[^aeiou]|z|gn|ps|x|y", first))
    if first.endswith("i") and len(first) > 3:
        if first.endswith("ici"):
            return "le "
        return "gli " if vowel or impure else "i "
    if _feminine(first):
        return "l'" if vowel else "la "
    return "l'" if vowel else ("lo " if impure else "il ")


def role_phrase(name: str) -> str:
    """«Lavoratore» -> «il lavoratore»; «Datore di lavoro» -> «il datore di lavoro»; «la Società» -> «la società»."""
    words = str(name or "").split()
    while words and re.sub(r"[^\w]", "", words[0].lower()) in _ARTICLES:
        words = words[1:]
    phrase = " ".join(words).lower()
    return _article(phrase) + phrase if phrase else ""


def label(name: str, case: str = "") -> str:
    """The party as the user reads it: a role with its article, lower case; a proper name as given."""
    name = " ".join(str(name or "").split())
    return role_phrase(name) if is_role(name, case) else name


_PREP_FORMS = {"di": {"il": "del", "lo": "dello", "la": "della", "l'": "dell'", "i": "dei", "gli": "degli", "le": "delle"},
               "a": {"il": "al", "lo": "allo", "la": "alla", "l'": "all'", "i": "ai", "gli": "agli", "le": "alle"},
               "da": {"il": "dal", "lo": "dallo", "la": "dalla", "l'": "dall'", "i": "dai", "gli": "dagli", "le": "dalle"},
               "in": {"il": "nel", "lo": "nello", "la": "nella", "l'": "nell'", "i": "nei", "gli": "negli", "le": "nelle"},
               "su": {"il": "sul", "lo": "sullo", "la": "sulla", "l'": "sull'", "i": "sui", "gli": "sugli", "le": "sulle"}}
# an articled preposition already written («dell'», «della », «agli ») -> (its preposition, its article)
_CONTRACTED = {form.rstrip("'"): (prep, art) for prep, forms in _PREP_FORMS.items() for art, form in forms.items()}
_SPLIT_RE = re.compile(r"\b(di|a|da|in|su|Di|A|Da|In|Su)\s+(il|lo|la|l['’]|i|gli|le)(?:\s+|(?<=['’]))")


def _joined(prep: str, art: str, apostrophe: str = "'") -> str:
    """The articled preposition of prep + art, with prep's capital and the text's own apostrophe."""
    key = "l'" if art.lower() in ("l'", "l’") else art.lower()
    joined = _PREP_FORMS[prep.lower()][key]
    joined = joined.replace("'", apostrophe)
    return joined if prep[:1].islower() else joined[:1].upper() + joined[1:]


def contract(text: str) -> str:
    """The articled prepositions a filled slot leaves split: «di il» -> «del», «a l'» -> «all'», «in la» -> «nella»
    (v60i: «l’» too, its apostrophe kept)."""
    def repl(m):
        prep, art = m.group(1), m.group(2)
        apostrophe = art[-1] if art[-1] in "'’" else "'"
        joined = _joined(prep, art, apostrophe)
        return joined + ("" if joined.endswith(("'", "’")) else " ")
    return _SPLIT_RE.sub(repl, text or "")


def contract_roles(text: str, names: Sequence[str]) -> str:
    """v60i (7): a preposition and an article left split before a role name's noun, in lower case or capitalised
    («di l'inquilino», «a l’acquirente») -> the articled preposition («dell'inquilino», «all’acquirente»)."""
    nouns = set()
    for name in names or []:
        bare = re.sub(r"^(?:(?:il|lo|la|i|gli|le)\s+|l['’])", "", str(name or ""), flags=re.IGNORECASE).strip()
        if bare:
            nouns.add(re.escape(bare.split()[0].lower()))
    if not nouns or not text:
        return text
    pattern = re.compile(r"\b(di|a|da|in|su|Di|A|Da|In|Su)\s+(il|lo|la|l['’]|i|gli|le)(\s+|(?<=['’]))(?=(?i:"
                         + "|".join(sorted(nouns, key=len, reverse=True)) + r")\b)")

    def repl(m):
        art = m.group(2)
        apostrophe = art[-1] if art[-1] in "'’" else "'"
        joined = _joined(m.group(1), art, apostrophe)
        return joined + ("" if joined.endswith(("'", "’")) else " ")
    return pattern.sub(repl, text)


def role_names(parties: Iterable[dict], case: str = "") -> List[str]:
    """The parties' names that are roles (is_role), longest first."""
    names = {" ".join(str((p or {}).get("name") or "").split()) for p in parties or [] if isinstance(p, dict)}
    return sorted((n for n in names if n and is_role(n, case)), key=len, reverse=True)


_PHRASE_ARTICLE_RE = re.compile(r"^(?:il |lo |la |i |gli |le |l')")
_JOINED_ALT = "|".join(sorted(_CONTRACTED, key=len, reverse=True))


def lower_roles(text: str, names: Sequence[str]) -> str:
    """Each capitalised occurrence of a role name in text written by its role, lower case, with its article: after an
    article, the noun lowered; after a preposition, the articled preposition; at a sentence's start, «Il lavoratore».
    A bold heading line (a §2 article's, its rubrica) is left as it is."""
    if not names or not text:
        return text
    if "\n" in text:
        return "\n".join(line if line.lstrip().startswith("**") else contract_roles(_lower_line(line, names), names)
                         for line in text.split("\n"))
    return text if text.lstrip().startswith("**") else contract_roles(_lower_line(text, names), names)


def _lower_line(text: str, names: Sequence[str]) -> str:
    for name in names or []:
        bare = re.sub(r"^(?:(?:il|lo|la|i|gli|le)\s+|l['’])", "", name, flags=re.IGNORECASE)
        if not bare:
            continue
        capital = bare[:1].upper() + bare[1:]
        phrase = role_phrase(bare)
        noun = _PHRASE_ARTICLE_RE.sub("", phrase)
        pattern = re.compile(r"(?P<pre>\b(?:di|a|da|in|su|per|con|tra|fra|Di|A|Da|In|Su|Per|Con)\s+)?"
                             r"(?P<art>\b(?:il|lo|la|i|gli|le|Il|Lo|La|I|Gli|Le)\s+|\b[lL]['’]\s*)?"
                             r"(?P<joined>\b(?i:" + _JOINED_ALT + r")(?:['’]\s*|\s+))?"
                             + re.escape(capital) + r"\b(?!\s+[A-ZÀ-Ü])")          # «Società Alfa»: a name
        source = text or ""

        def repl(m, phrase=phrase, noun=noun, source=source):
            pre, art, joined = m.group("pre") or "", m.group("art") or "", m.group("joined") or ""
            if joined:          # v60i (7): «dell'Inquilino» -> «dell'inquilino», never «dell'l'inquilino»
                word = re.match(r"[^\W\d_]+", joined).group(0)
                prep = _CONTRACTED[word.lower()][0]
                prep = prep if word[:1].islower() else prep[:1].upper() + prep[1:]
                apostrophe = next((c for c in joined if c in "'’"), "'")
                own = re.match(r"(il |lo |la |i |gli |le |l')", phrase).group(1)
                out = _joined(prep, own.strip() if own != "l'" else "l'", apostrophe)
                return out + ("" if out.endswith(("'", "’")) else " ") + noun
            if art:
                return pre + art + noun
            if pre:
                return contract(pre + phrase)
            before = source[:m.start()]
            at_start = not before.strip() or re.search(r"(?:[.!?]\s+|\n\s*(?:[-*]\s+)?)$", before) is not None
            return phrase[:1].upper() + phrase[1:] if at_start else phrase
        text = pattern.sub(repl, source)
    return text
