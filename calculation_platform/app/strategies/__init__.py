from .decision_table import DecisionTableStrategy
from .date_split_interest import DateSplitInterestStrategy
from .expression import ExpressionStrategy
from .penal_range_draft import PenalRangeDraftStrategy
from .percentage_of_base import PercentageOfBaseStrategy
from .procedural_deadline import ProceduralDeadlineStrategy
from .ravvedimento import RavvedimentoStrategy
from .progressive_brackets import ProgressiveBracketsStrategy
from .table_lookup import TableLookupStrategy

STRATEGIES = {
    "expression": ExpressionStrategy,
    "progressive_brackets": ProgressiveBracketsStrategy,
    "percentage_of_base": PercentageOfBaseStrategy,
    "date_split_interest": DateSplitInterestStrategy,
    "decision_table": DecisionTableStrategy,
    "penal_range_draft": PenalRangeDraftStrategy,
    "table_lookup": TableLookupStrategy,
    "procedural_deadline": ProceduralDeadlineStrategy,
    "ravvedimento": RavvedimentoStrategy,
}
