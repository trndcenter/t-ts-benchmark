import nip
from xgboost import XGBClassifier

@nip.nip
class GBCXGBModel(XGBClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)