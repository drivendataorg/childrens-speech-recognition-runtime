from pathlib import Path

from loguru import logger
import nemo.collections.asr as nemo_asr


class ParakeetModel:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, model_path):
        logger.info(f"Loading model from: {model_path}")
        model = nemo_asr.models.ASRModel.restore_from(model_path)
        return cls(model)

    def predict(self, audio_path: Path):
        hypotheses = self.model.transcribe([str(audio_path)], verbose=False)
        pred = hypotheses[0].text
        return pred

    def predict_batch(self, audio_paths: list[Path], batch_size: int = 4):
        hypotheses = self.model.transcribe(
            [str(p) for p in audio_paths], batch_size=batch_size, verbose=False
        )
        preds = [hyp.text for hyp in hypotheses]
        return preds
