import json
import os
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from lib import WordModel


def main():
    # Load model
    src_root = Path(__file__).parent.resolve()
    model_path = src_root / "model" / "word_model.txt"
    logger.info(f"Loading model from: {model_path}")
    model = WordModel.load(model_path)

    # Load manifest and process data
    data_dir = Path("data")
    manifest_path = data_dir / "utterance_metadata.jsonl"
    with manifest_path.open("r") as fr:
        lines = fr.readlines()
    logger.info(f"Processing {len(lines)} utterances from {manifest_path}")

    # Predict
    predictions = {}
    step = max(1, len(lines) // 100)  # Log every 1%
    with tqdm(total=len(lines), file=open(os.devnull, "w")) as pbar:
        for idx, line in enumerate(lines):
            item = json.loads(line)
            pred = model.predict(data_dir / item["audio_path"])
            predictions[item["utterance_id"]] = pred
            pbar.update(1)
            if idx % step == 0:
                logger.info(str(pbar))

    # Write submission file
    submission_format_path = data_dir / "submission_format.jsonl"
    submission_path = Path("submission") / "submission.jsonl"
    logger.info(f"Writing submission file to {submission_path}")
    with submission_format_path.open("r") as fr, submission_path.open("w") as fw:
        for line in fr:
            item = json.loads(line)
            item["orthographic_text"] = predictions[item["utterance_id"]]
            fw.write(json.dumps(item) + "\n")

    logger.success("Done.")


if __name__ == "__main__":
    main()
