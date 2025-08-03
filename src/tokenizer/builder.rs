//! Tokenizer builder for creating tokenizer instances

use super::{
    TokenizerType,
    traits::Tokenizer,
    bpe::BPETokenizer,
    production::ProductionTokenizer,
};
use anyhow::Result;

/// Builder for creating tokenizer instances
pub struct TokenizerBuilder {
    tokenizer_type: TokenizerType,
    tokenizer_path: String,
}

impl TokenizerBuilder {
    /// Create a new tokenizer builder
    pub fn new(tokenizer_type: Option<TokenizerType>, tokenizer_path: String) -> Self {
        Self {
            tokenizer_type: tokenizer_type.unwrap_or_default(),
            tokenizer_path,
        }
    }

    /// Build the tokenizer based on the current configuration
    pub fn build(&self) -> Result<Box<dyn Tokenizer>> {
        match self.tokenizer_type {
            TokenizerType::BPE => {
                let tokenizer = BPETokenizer::from_tokenizer_json(&self.tokenizer_path)?;
                Ok(Box::new(tokenizer))
            }
            TokenizerType::HuggingFace => {
                #[cfg(not(feature = "tokenizers"))]
                return Err(anyhow::anyhow!(
                    "HuggingFace tokenizer requires 'tokenizers' feature to be enabled"
                ));
                
                #[cfg(feature = "tokenizers")]
                {
                    let tokenizer = ProductionTokenizer::from_tokenizer_json(&self.tokenizer_path)?;
                    Ok(Box::new(tokenizer))
                }
            }
        }
    }
}

/// Demo function to show tokenizer functionality
#[allow(dead_code)]
pub fn demo_tokenizers() -> Result<()> {
    use super::{TokenizerConfig, TokenizerType};
    
    println!("🚀 Tokenizer Demo\n");
    
    let test_texts = [
        "Hello world!",
        "The capital of France is",
        "Rust programming language",
    ];

    // Set tokenizer type based on features
    #[cfg(feature = "tokenizers")]
    let tokenizer_type = TokenizerType::HuggingFace;
    
    #[cfg(not(feature = "tokenizers"))]
    let tokenizer_type = TokenizerType::BPE;

    let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M/tokenizer.json";
    
    println!("=== USING {:?} TOKENIZER ===\n", tokenizer_type);
    
    // Create tokenizer using TokenizerBuilder
    let tokenizer = TokenizerBuilder::new(Some(tokenizer_type), model_path.to_string())
        .build()?;

    println!("📊 Vocab size: {}\n", tokenizer.vocab_size());
    
    for text in &test_texts {
        println!("Text: '{}'", text);
        let token_ids = tokenizer.encode(text)?;
        let decoded = tokenizer.decode(&token_ids)?;
        println!("   Tokens: {:?}", token_ids);
        println!("   Decoded: '{}\n", decoded);
    }

    // Show tokenizer type info
    match tokenizer_type {
        TokenizerType::BPE => {
            println!("ℹ️  Using educational BPETokenizer");
            println!("   To use production tokenizer, enable 'tokenizers' feature:");
            println!("   cargo run --features tokenizers\n");
        }
        TokenizerType::HuggingFace => {
            println!("ℹ️  Using production HuggingFace tokenizer\n");
        }
    }
    
    Ok(())
}
