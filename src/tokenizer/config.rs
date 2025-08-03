//! Configuration types for tokenization

/// Configuration for tokenization
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub max_length: Option<usize>,
    pub padding: PaddingStrategy,
    pub truncation: bool,
    pub return_attention_mask: bool,
    pub return_offsets: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: None,
            padding: PaddingStrategy::None,
            truncation: false,
            return_attention_mask: true,
            return_offsets: false,
        }
    }
}

/// Padding strategy for tokenization
#[derive(Debug, Clone, PartialEq)]
pub enum PaddingStrategy {
    None,
    MaxLength,
    LongestFirst,
}

/// Type of tokenizer to use
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TokenizerType {
    #[default]
    BPE,
    HuggingFace,
}

/// Rich encoding result with all token information
#[derive(Debug, Clone)]
pub struct EncodingResult {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub token_type_ids: Option<Vec<u32>>,
    pub offsets: Option<Vec<(usize, usize)>>,
}

impl EncodingResult {
    pub fn simple(input_ids: Vec<u32>) -> Self {
        let attention_mask = vec![1; input_ids.len()];
        Self {
            input_ids,
            attention_mask,
            token_type_ids: None,
            offsets: None,
        }
    }
}
