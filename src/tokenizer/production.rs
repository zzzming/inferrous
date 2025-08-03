//! Production tokenizer using the `tokenizers` crate

use std::path::Path;
use anyhow::Result;
use super::{
    SpecialTokens, ChatMessage,
    config::TokenizerConfig,
    traits::Tokenizer,
    EncodingResult,
};

/// Production tokenizer using the Hugging Face `tokenizers` crate
#[cfg(feature = "tokenizers")]
pub struct ProductionTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "tokenizers")]
impl ProductionTokenizer {
    /// Load a tokenizer from a tokenizer.json file
    pub fn from_tokenizer_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        Ok(ProductionTokenizer { tokenizer })
    }
}

#[cfg(feature = "tokenizers")]
impl Tokenizer for ProductionTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
        
        Ok(encoding.get_ids().to_vec())
    }
    
    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(token_ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }
    
    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true) as usize
    }
    
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token).copied()
    }
    
    fn encode_with_config(&self, text: &str, config: &TokenizerConfig) -> Result<EncodingResult> {
        let mut encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
            
        let mut input_ids = encoding.get_ids().to_vec();
        
        // Apply truncation if needed
        if let Some(max_len) = config.max_length {
            if config.truncation && input_ids.len() > max_len {
                input_ids.truncate(max_len);
            }
        }
        
        // Generate attention mask
        let attention_mask = vec![1u32; input_ids.len()];
        
        // Get offsets if requested
        let offsets = if config.return_offsets {
            Some(encoding.get_offsets().iter()
                .map(|&(start, end)| (start as usize, end as usize))
                .collect())
        } else {
            None
        };
        
        Ok(EncodingResult {
            input_ids,
            attention_mask,
            token_type_ids: None, // Not supported in basic implementation
            offsets,
        })
    }
    
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        texts.iter()
            .map(|text| self.encode(text))
            .collect()
    }
    
    fn encode_batch_with_config(&self, texts: &[&str], config: &TokenizerConfig) -> Result<Vec<EncodingResult>> {
        let mut results: Result<Vec<_>> = texts.iter()
            .map(|text| self.encode_with_config(text, config))
            .collect();
            
        let mut results = results?;
        
        // Apply batch-level padding if needed
        if config.padding == super::config::PaddingStrategy::LongestFirst && texts.len() > 1 {
            let max_len = results.iter().map(|r| r.input_ids.len()).max().unwrap_or(0);
            
            for result in &mut results {
                let current_len = result.input_ids.len();
                if current_len < max_len {
                    // Use pad_token_id from tokenizer or default to 0
                    let pad_token_id = self.token_to_id("<pad>").unwrap_or(0);
                    result.input_ids.extend(vec![pad_token_id; max_len - current_len]);
                    result.attention_mask.extend(vec![0; max_len - current_len]);
                }
            }
        }
        
        Ok(results)
    }
    
    fn decode_batch(&self, token_ids_batch: &[&[u32]]) -> Result<Vec<String>> {
        token_ids_batch.iter()
            .map(|token_ids| self.decode(token_ids))
            .collect()
    }
    
    fn get_special_tokens(&self) -> std::collections::HashMap<String, u32> {
        let vocab = self.tokenizer.get_vocab(true);
        let special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>", "<cls>", "<sep>"];
        
        special_tokens.iter()
            .filter_map(|&token| {
                vocab.get(token).map(|&id| (token.to_string(), id))
            })
            .collect()
    }
    
    fn add_special_tokens(&mut self, _tokens: &std::collections::HashMap<String, u32>) -> Result<()> {
        // HuggingFace tokenizers are immutable after creation
        Err(anyhow::anyhow!(
            "Adding special tokens not supported for ProductionTokenizer - tokenizer is immutable. "
        ))
    }
    
    fn normalize(&self, text: &str) -> String {
        // Use the tokenizer's normalization if available
        text.to_string() // HuggingFace handles normalization internally
    }
    
    fn pre_tokenize(&self, _text: &str) -> Vec<String> {
        // Pre-tokenization is handled internally by the tokenizer
        vec![]
    }
    
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        // Simple chat template implementation
        let mut formatted = String::new();
        
        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    formatted.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", msg.content));
                }
                "user" => {
                    formatted.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content));
                }
                "assistant" => {
                    formatted.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", msg.content));
                }
                _ => {
                    formatted.push_str(&format!("<|im_start|>{}<|im_end|>\n", msg.role));
                }
            }
        }
        
        // Add assistant prompt if not already present
        if !formatted.ends_with("<|im_start|>assistant\n") {
            formatted.push_str("<|im_start|>assistant\n");
        }
        
        Ok(formatted)
    }
    
    fn get_id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}
