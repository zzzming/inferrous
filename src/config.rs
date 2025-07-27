use crate::cpu::{AcceleratorType, SystemInfo};
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::cmp;

/// Configuration for the Inferrous engine
/// Similar to a Go struct but with Rust ownership semantics
#[derive(Debug, Clone)]
pub struct Config {
    // File paths
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,

    // Threading configuration
    pub rayon_threads: usize,
    pub omp_threads: usize,

    // Generation parameters
    pub max_tokens: usize,
    pub temperature: f32,

    // Device configuration
    pub use_metal: bool,
}

impl Config {
    /// Create a new Config with automatic hardware detection (like a Go constructor)
    pub fn new<P: AsRef<Path>>(model_path: P, tokenizer_path: P) -> Self {
        let system_info = SystemInfo::detect();
        let optimal_threads = system_info.optimal_cpu_threads();
        let best_accelerator = system_info.best_accelerator();

        Self {
            model_path: model_path.as_ref().to_path_buf(),
            tokenizer_path: tokenizer_path.as_ref().to_path_buf(),
            rayon_threads: optimal_threads,
            omp_threads: optimal_threads,
            max_tokens: 200,
            temperature: 1.0,
            use_metal: matches!(best_accelerator, AcceleratorType::Metal),
        }
    }

    /// Get system hardware information (useful for debugging)
    pub fn system_info() -> SystemInfo {
        SystemInfo::detect()
    }

    /// Get optimal configuration for detected hardware
    pub fn auto_optimized<P: AsRef<Path>>(model_path: P, tokenizer_path: P) -> Self {
        let system_info = SystemInfo::detect();
        let mut config = Self::new(model_path, tokenizer_path);

        // Optimize based on detected hardware
        match system_info.best_accelerator() {
            AcceleratorType::Metal => {
                config.use_metal = true;
                // Metal can handle more parallel work
                config.rayon_threads = system_info.logical_cpus;
            }
            AcceleratorType::Cuda => {
                config.use_metal = false;
                // CUDA setup (future)
            }
            AcceleratorType::Rocm => {
                config.use_metal = false;
                // ROCm setup (future)
            }
            AcceleratorType::Cpu => {
                config.use_metal = false;
                config.rayon_threads = system_info.optimal_cpu_threads();
            }
        }

        config
    }

    /// Builder pattern methods (very Rust-like)
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.rayon_threads = threads;
        self.omp_threads = threads;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_metal(mut self, use_metal: bool) -> Self {
        self.use_metal = use_metal;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if !self.model_path.exists() {
            return Err(anyhow::anyhow!(
                "Model file not found: {:?}",
                self.model_path
            ));
        }

        if !self.tokenizer_path.exists() {
            return Err(anyhow::anyhow!(
                "Tokenizer file not found: {:?}",
                self.tokenizer_path
            ));
        }

        if self.rayon_threads == 0 {
            return Err(anyhow::anyhow!("Thread count must be > 0"));
        }

        if self.max_tokens == 0 {
            return Err(anyhow::anyhow!("Max tokens must be > 0"));
        }

        Ok(())
    }

    /// Apply environment variable settings
    pub fn apply_env_settings(&self) -> Result<()> {
        unsafe {
            let rayon_threads = self.rayon_threads.to_string();
            let omp_threads = self.omp_threads.to_string();
            let optimal_threads = cmp::min(self.rayon_threads, self.omp_threads).to_string();
            
            println!("Setting RAYON_NUM_THREADS = {}", rayon_threads);
            std::env::set_var("RAYON_NUM_THREADS", &optimal_threads);
            
            println!("Setting OMP_NUM_THREADS = {}", omp_threads);
            std::env::set_var("OMP_NUM_THREADS", &optimal_threads);
        }
        Ok(())
    }
}

// Default trait implementation (like Go's zero values)
impl Default for Config {
    fn default() -> Self {
        let system_info = SystemInfo::detect();
        let optimal_threads = system_info.optimal_cpu_threads();

        Self {
            model_path: PathBuf::from("models/model.gguf"),
            tokenizer_path: PathBuf::from("models/tokenizer.json"),
            rayon_threads: optimal_threads,
            omp_threads: optimal_threads,
            max_tokens: 50,
            temperature: 1.0,
            use_metal: false, // Conservative default
        }
    }
}
