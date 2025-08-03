//! Common types and structs for the tokenizer module

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a chat message with role and content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender (e.g., "user", "assistant", "system")
    pub role: String,
    /// Content of the message
    pub content: String,
}

/// Special tokens used by the tokenizer
#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    /// Unknown token
    pub unk: u32,
    /// Beginning of sentence token
    pub bos: u32,
    /// End of sentence token
    pub eos: u32,
    /// Padding token
    pub pad: u32,
}

impl SpecialTokens {
    /// Create a new SpecialTokens instance with default values
    pub fn new() -> Self {
        Self {
            unk: 0,
            bos: 1,
            eos: 2,
            pad: 3,
        }
    }
}