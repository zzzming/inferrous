// Inferrous Library - Rust Inference Engine
pub mod config;
pub mod cpu;
pub mod tokenizer;

// Re-export main types for easy importing
pub use config::Config;
pub use cpu::{AcceleratorType, SystemInfo};
pub use tokenizer::InferrousTokenizer;

// Common result type for the library
pub type Result<T> = anyhow::Result<T>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_system_info_detection() {
        let system_info = SystemInfo::detect();

        // Basic sanity checks
        assert!(system_info.logical_cpus > 0);
        assert!(system_info.physical_cpus > 0);
        assert!(system_info.physical_cpus <= system_info.logical_cpus);

        // Should have at least CPU available
        let accelerator = system_info.best_accelerator();
        println!("Detected accelerator: {}", accelerator);
    }

    #[test]
    fn test_config_creation() {
        let config = Config::new("dummy_model.gguf", "dummy_tokenizer.json");

        assert!(config.rayon_threads > 0);
        assert!(config.omp_threads > 0);
        assert!(config.max_tokens > 0);
        assert!(config.temperature > 0.0);
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = Config::new("dummy_model.gguf", "dummy_tokenizer.json")
            .with_threads(8)
            .with_max_tokens(100)
            .with_temperature(0.8)
            .with_metal(false);

        assert_eq!(config.rayon_threads, 8);
        assert_eq!(config.omp_threads, 8);
        assert_eq!(config.max_tokens, 100);
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.use_metal, false);
    }

    #[test]
    fn test_config_default() {
        let config = Config::default();

        assert_eq!(config.model_path, PathBuf::from("models/model.gguf"));
        assert_eq!(
            config.tokenizer_path,
            PathBuf::from("models/tokenizer.json")
        );
        assert!(config.rayon_threads > 0);
    }

    #[test]
    fn test_optimal_cpu_threads() {
        let system_info = SystemInfo::detect();
        let optimal = system_info.optimal_cpu_threads();

        // Should be reasonable
        assert!(optimal > 0);
        assert!(optimal <= system_info.logical_cpus);
    }
}
