"""v79.18.4 -- the persona offesa is a claimant (data «claimant.roles»): the verified frame's ruolo «persona_offesa» gives
the side attore, so the strategy never takes the defence's rules for the victim through 79.18.2's pretesa-tipo fallback."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.rag import analysis_step as A  # noqa: E402
from src.rag import strategy_step as S  # noqa: E402


def _frame(valore, intake=None):
    return {"ruolo": {"valore": valore, "intake": intake or valore}, "cliente": {"nome": "Elena Marchetti"},
            "parti": [{"nome": "Elena Marchetti", "lato": "cliente", "ruolo_nel_fatto": "ha pagato una bicicletta mai ricevuta"},
                      {"nome": "Fabio R.", "lato": "controparte", "ruolo_nel_fatto": "ha incassato il prezzo"}],
            "addebito_o_domanda": {"tipo": "accusa", "contenuto": "truffa", "articolo": "640 c.p.", "contro": "Fabio R."}}


def test_the_persona_offesa_is_a_claimant_by_the_verified_ruolo():
    assert A.client_side(_frame("persona_offesa"), "Elena Marchetti") == ("attore", "ruolo: persona offesa")
    assert A.client_side(_frame("persona_offesa", "parte_civile"), "Elena Marchetti")[0] == "attore"
    assert A.client_side(_frame("parte offesa"), "Elena Marchetti")[0] == "attore"
    assert A.claimant(_frame("persona_offesa"), "Elena Marchetti") is True
    # the accused stays the defence; an unknown role stays neutral
    assert A.client_side(_frame("imputato/indagato", "imputato"), "Elena Marchetti")[0] == "difesa"
    assert A.client_side(_frame("sconosciuto"), "Elena Marchetti") == ("neutro", "")
    for word in ("persona offesa", "parte offesa", "parte civile", "persone offese", "parti offese", "parti civili"):
        assert word in A.config()["claimant"]["roles"]
        assert word not in A.config()["claimant"]["defendant_roles"]


def test_the_strategys_side_is_the_frames_not_the_pretesa_tipos():
    side, why = A.client_side(_frame("persona_offesa"), "Elena Marchetti")
    assert S.side_of(side, _frame("persona_offesa")) == ("attore", "frame")
    # without the verified ruolo the 79.18.2 fallback still reads the tipo (an indagato's accusa -> difesa)
    assert S.side_of("neutro", _frame("sconosciuto")) == ("difesa", "pretesa_tipo:accusa")
