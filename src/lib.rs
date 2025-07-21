// Inferrous Library - Rust Inference Engine
pub mod config;
pub mod tokenizer;
pub mod cpu;

// Re-export main types for easy importing
pub use config::Config;
pub use tokenizer::InferrousTokenizer;
pub use cpu::{SystemInfo, AcceleratorType};

// Common result type for the library
pub type Result<T> = anyhow::Result<T>;