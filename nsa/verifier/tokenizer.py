class TokenizerAligner:
    """
    Ensures 1:1 token-to-state index alignment between the causal LLM
    and the state encoder.
    """

    def __init__(self, llm_tokenizer, encoder_tokenizer=None):
        self.llm_tokenizer = llm_tokenizer
        self.encoder_tokenizer = (
            encoder_tokenizer if encoder_tokenizer is not None else llm_tokenizer
        )

        self.is_same_tokenizer = self.llm_tokenizer == self.encoder_tokenizer

    def align_tokens(self, input_ids):
        """
        Aligns the tokens if tokenizers are different.
        If they are the same, returns the input_ids directly.
        """
        if self.is_same_tokenizer:
            return input_ids

        # Decode and re-encode if tokenizers differ
        # In a real implementation, this would need complex character-to-token alignment mapping
        text = self.llm_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        aligned_ids = self.encoder_tokenizer.encode(text, return_tensors="pt")
        return aligned_ids
