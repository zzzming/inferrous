use tokenizers::Tokenizer;
use std::path::Path;
use anyhow::Result;

/// Wrapper around the tokenizers library
/// This allows us to eventually replace the tokenizer implementation
pub struct InferrousTokenizer {
    inner: Tokenizer,
}

impl InferrousTokenizer {
    /// Load tokenizer from file (like Go's constructor pattern)
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
            
        Ok(Self { inner: tokenizer })
    }
    
    /// Encode text to token IDs
    /// Returns Vec<u32> instead of the tokenizers-specific type
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;
            
        Ok(encoding.get_ids().to_vec())
    }
    
    /// Decode token IDs back to text  
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let text = self.inner
            .decode(token_ids, false)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;
            
        Ok(text)
    }
    
    /// Decode a single token (useful for streaming)
    pub fn decode_token(&self, token_id: u32) -> Result<String> {
        self.decode(&[token_id])
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
    
    /// Get token for a word (if exists)
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
    
    /// Get word for a token ID (if exists)
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

// Debug trait for easier debugging (like Go's fmt.Stringer)
impl std::fmt::Debug for InferrousTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferrousTokenizer")
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
}