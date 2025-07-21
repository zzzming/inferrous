use anyhow::Result;
use candle_core::{Device, Tensor, IndexOp, quantized::gguf_file};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use std::fs::File;
use std::io::{self, Write};

// Use our library components
use inferrous::{Config, InferrousTokenizer, SystemInfo};

fn main() -> Result<()> {
    println!("Inferrous - Rust Inference Engine");
    
    // Detect and show system information
    let system_info = SystemInfo::detect();
    println!("💻 System: {}", system_info);
    println!("🎯 Best accelerator: {}", system_info.best_accelerator());
    
    // Create auto-optimized configuration
    let config = Config::auto_optimized(
        "models/Qwen2.5-0.5B-Instruct/qwen2.5-0.5b-q4.gguf",
        "models/Qwen2.5-0.5B-Instruct/tokenizer.json"
    )
    .with_max_tokens(200)
    .with_metal(false);  // Override: CPU only for now (RMSNorm issue)
    
    // Validate configuration
    config.validate()?;
    
    // Apply environment settings
    config.apply_env_settings()?;
    
    println!("⚡ Auto-configured: {} threads for optimal performance", config.rayon_threads);
    println!("📁 Model: {:?}", config.model_path);
    println!("📄 Tokenizer: {:?}", config.tokenizer_path);
    
    // Use CPU device (from config in the future)
    let device = Device::Cpu;
    println!("Device: {:?}", device);
    
    // Load the GGUF model
    let mut file = File::open(&config.model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model_weights = ModelWeights::from_gguf(content, &mut file, &device)?;
    println!("✅ Model loaded successfully!");
    
    // Load tokenizer using our wrapper
    let tokenizer = InferrousTokenizer::from_file(&config.tokenizer_path)?;
    println!("✅ Tokenizer loaded! Vocab size: {}", tokenizer.vocab_size());
    
    // Interactive mode - this is what you expected!
    println!("\n🎯 Interactive Mode");
    println!("Type your text (or 'quit' to exit):");
    
    loop {
        print!("\n> ");
        io::stdout().flush()?;
        
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();
        
        if user_input == "quit" || user_input.is_empty() {
            println!("Goodbye!");
            break;
        }
        
        println!("🤔 Processing your text: '{}'", user_input);
        
        // Tokenize user input with our wrapper
        let mut token_ids = tokenizer.encode(user_input)?;
        println!("Tokens: {:?}", token_ids.iter().take(10).collect::<Vec<_>>());
        
        print!("🤖 AI Response:");
        io::stdout().flush()?;
        
        // Generate response tokens with KV cache simulation
        let mut start_pos = token_ids.len();
        
        for _step in 0..config.max_tokens {
            // Only process new tokens after the first pass
            let current_tokens = if _step == 0 {
                &token_ids[..]  // First pass: all tokens
            } else {
                &token_ids[start_pos..]  // Subsequent: only new tokens
            };
            
            let input_tensor = Tensor::new(current_tokens, &device)?.unsqueeze(0)?;
            let logits = model_weights.forward(&input_tensor, if _step == 0 { 0 } else { start_pos })?;
            
            // print!("{}", input_tensor.shape());
            let last_logits = logits.i(0)?;
            let next_token_id = last_logits.argmax(0)?.to_scalar::<u32>()?;
            
            // Stop on special tokens
            if next_token_id == 0 || next_token_id == 1 || next_token_id == 2 {
                break;
            }
            
            token_ids.push(next_token_id);
            start_pos = token_ids.len() - 1;  // Update start position
            
            // Decode the new token using our wrapper
            let token_text = tokenizer.decode_token(next_token_id)?;
            print!("{}", token_text);
            io::stdout().flush()?;
        }
        
        println!("...\n");
    }

    Ok(())
}
