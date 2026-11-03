"""SAM 3 for aerial scene recognition: diagnosis and training-free / few-shot modules.

Modules
-------
scoring              presence / localization scores taken *before* SAM 3 multiplies them
data                 AID split, download, image loading
concept_translation  (1) class name -> visible noun phrases, and choosing among them
ensemble             (2) average the presence of several phrases per class
balance              (3) label-free class-skew correction
composition          (4) few-shot scene classifier over phrase presences ("parts")
metrics              accuracy, fixed/broken counts, error types, trigger signals
"""

__all__ = ["scoring", "data", "concept_translation", "ensemble", "balance", "composition", "metrics"]
