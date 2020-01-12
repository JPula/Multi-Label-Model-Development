import pandas as pd
import re

class DefaultPreprocessor():
    def __init__(self):
        pass
    
    def _clean(self, text: str) -> str:
        text = re.sub(r'[^0-9a-zA-Z ]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = text.lower()

        return text if text else '_'

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        data.comment_text = data.comment_text.apply(self._clean)
        
        return data