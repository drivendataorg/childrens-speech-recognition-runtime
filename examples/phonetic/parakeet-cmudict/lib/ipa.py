import eng_to_ipa as ipa


def text_to_ipa(text: str) -> str:
    """Convert English text to IPA transcript."""
    return ipa.convert(text)
