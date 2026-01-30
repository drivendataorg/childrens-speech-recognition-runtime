# Example Submission: Parakeet [Word Track]

This is an example solution for the [Word Track](https://www.drivendata.org/competitions/308/childrens-word-asr/) of the On Top of Pasketti: Children’s Speech Recognition Challenge.

It provides a simple implementation of using NVIDIA's [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) model to perform automatic speech recognition to orthographic text.

> [!IMPORTANT]
> The model weights for parakeet-tdt-0.6b-v2 are not included in the repository. You will need to run the download script to download them for this to be a working example.

## How to use

1. From this directory, download the parakeet-tdt-0.6b-v2 model files from Hugging Face

    ```sh
    bash download.sh
    ```

2. Pack the code and model into a `submission.zip`. From the repository root, run

    ```sh
    just pack-example word/parakeet
    ```

3. You can find the submission file at `submission/submission.zip` relative to the repository root. To validate the file, run

    ```sh
    just check-submission
    ```
