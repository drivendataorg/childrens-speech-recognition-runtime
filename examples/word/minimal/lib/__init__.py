from pathlib import Path


class WordModel:
    def __init__(self, model_val):
        self.model_val = model_val

    @classmethod
    def load(cls, model_path):
        with Path(model_path).open("r") as f:
            model_val = f.read().strip()
        return cls(model_val)

    def predict(self, audio_path):
        with Path(audio_path).open("rb") as f:
            _ = f.read()
        return self.model_val
