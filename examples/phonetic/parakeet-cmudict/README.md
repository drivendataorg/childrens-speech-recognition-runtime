# Example Submission: Whisper + CMUdict [Phonetic Track]

This is an example solution for the [Phonetic Track](https://www.drivendata.org/competitions/308/childrens-word-asr/) of the On Top of Pasketti: Children’s Speech Recognition Challenge.

It provides a simple implementation of using NVIDIA's [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) model to perform automatic speech recognition to orthographic text. The [CMU Pronouncing Dictionary (CMUdict)](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) (via the [English-to-IPA package](https://github.com/mphilli/English-to-IPA)) is used to convert to the International Phonetic Alphabet (IPA) text.

> [!IMPORTANT]
> The model weights for parakeet-tdt-0.6b-v2 are not included in the repository. You will need to run the download script to download them for this to be a working example.

## How to use

1. From this directory, download the parakeet-tdt-0.6b-v2 model files from Hugging Face

    ```sh
    bash download.sh
    ```

2. Pack the code and model into a `submission.zip`. From the repository root, run

    ```sh
    just pack-example phonetic/whisper-cmudict
    ```

3. You can find the submission file at `submission/submission.zip` relative to the repository root. To validate the file, run

    ```sh
    just check-submission
    ```
