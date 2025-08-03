//! Tokenizer module for the Inferrous library.
//!
//! This module provides a unified interface for tokenization with multiple backends:
//! - `bpe`: A pure Rust implementation of Byte Pair Encoding (BPE)
//! - `production`: A production-grade tokenizer using the `tokenizers` crate (requires the `tokenizers` feature)
//!
//! # Examples
//!
//! ```no_run
//! use inferrous::tokenizer::{Tokenizer, TokenizerBuilder, TokenizerType};
//!
//! // Create a new tokenizer from a JSON file
//! let tokenizer = TokenizerBuilder::new(
//!     Some(TokenizerType::BPE),
//!     "path/to/tokenizer.json".to_string(),
//! ).build().unwrap();
//!
//! // Encode some text
//! let tokens = tokenizer.encode("Hello, world!").unwrap();
//! println!("Encoded tokens: {:?}", tokens);
//! ```

pub mod builder;
pub mod types;
pub mod traits;
pub mod bpe;
pub mod production;
pub mod config;

pub use self::builder::*;
pub use self::config::*;
pub use self::types::*;
pub use self::traits::*;
pub use self::bpe::*;
pub use self::production::*;

// Re-export the demo function if the tokenizers feature is enabled
// #[cfg(feature = "tokenizers")]
// pub use crate::tokenizer_impl::demo_tokenizers;
