//! Tokenizer trait definitions
use anyhow::Result;
use super::{
    config::TokenizerConfig,
    EncodingResult,
    ChatMessage,
};

/// Core tokenizer trait that all tokenizers must implement
pub trait Tokenizer: Send + Sync {
    // Core tokenization
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, token_ids: &[u32]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn token_to_id(&self, token: &str) -> Option<u32>;
    
    // Enhanced encoding with configuration
    fn encode_with_config(&self, text: &str, config: &TokenizerConfig) -> Result<EncodingResult>;
    
    // Batch processing
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>>;
    fn encode_batch_with_config(
        &self,
        texts: &[&str],
        config: &TokenizerConfig,
    ) -> Result<Vec<EncodingResult>>;
    
    fn decode_batch(&self, token_ids_batch: &[&[u32]]) -> Result<Vec<String>>;
    
    // Special token handling
    fn get_special_tokens(&self) -> std::collections::HashMap<String, u32>;
    fn add_special_tokens(&mut self, tokens: &std::collections::HashMap<String, u32>) -> Result<()>;
    
    // Text processing
    fn normalize(&self, text: &str) -> String;
    fn pre_tokenize(&self, text: &str) -> Vec<String>;
    
    // Advanced features
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String>;
    fn get_id_to_token(&self, id: u32) -> Option<String>;
}

/// Default implementations for optional methods
impl<T: Tokenizer> Tokenizer for Box<T> {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        (**self).encode(text)
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        (**self).decode(token_ids)
    }

    fn vocab_size(&self) -> usize {
        (**self).vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        (**self).token_to_id(token)
    }

    fn encode_with_config(&self, text: &str, config: &TokenizerConfig) -> Result<EncodingResult> {
        (**self).encode_with_config(text, config)
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        (**self).encode_batch(texts)
    }

    fn encode_batch_with_config(
        &self,
        texts: &[&str],
        config: &TokenizerConfig,
    ) -> Result<Vec<EncodingResult>> {
        (**self).encode_batch_with_config(texts, config)
    }

    fn decode_batch(&self, token_ids_batch: &[&[u32]]) -> Result<Vec<String>> {
        (**self).decode_batch(token_ids_batch)
    }

    fn get_special_tokens(&self) -> std::collections::HashMap<String, u32> {
        (**self).get_special_tokens()
    }

    fn add_special_tokens(&mut self, tokens: &std::collections::HashMap<String, u32>) -> Result<()> {
        (**self).add_special_tokens(tokens)
    }

    fn normalize(&self, text: &str) -> String {
        (**self).normalize(text)
    }

    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        (**self).pre_tokenize(text)
    }

    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        (**self).apply_chat_template(messages)
    }

    fn get_id_to_token(&self, id: u32) -> Option<String> {
        (**self).get_id_to_token(id)
    }
}
