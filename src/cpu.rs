/// System hardware detection module
/// Detects CPU cores, GPU devices, and other accelerators
use std::fmt;

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub logical_cpus: usize,
    pub physical_cpus: usize,
    pub metal_gpus: usize,
    pub cuda_gpus: usize,
    pub rocm_gpus: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum AcceleratorType {
    Cpu,
    Metal,
    Cuda,
    Rocm,
}

impl SystemInfo {
    /// Detect all system hardware
    pub fn detect() -> Self {
        Self {
            logical_cpus: Self::detect_logical_cpus(),
            physical_cpus: Self::detect_physical_cpus(),
            metal_gpus: Self::detect_metal_gpus(),
            cuda_gpus: Self::detect_cuda_gpus(),
            rocm_gpus: Self::detect_rocm_gpus(),
        }
    }

    /// Get logical CPU count (includes hyperthreading)
    fn detect_logical_cpus() -> usize {
        // Use Rust standard library (available since 1.59)
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or_else(|_| Self::fallback_cpu_detection())
    }

    /// Get physical CPU core count
    fn detect_physical_cpus() -> usize {
        #[cfg(target_os = "macos")]
        {
            Self::macos_physical_cpus().unwrap_or_else(|| Self::detect_logical_cpus())
        }

        #[cfg(target_os = "linux")]
        {
            Self::linux_physical_cpus().unwrap_or_else(|| Self::detect_logical_cpus())
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            // Fallback: assume no hyperthreading
            Self::detect_logical_cpus()
        }
    }

    /// Detect Metal GPUs (Apple Silicon)
    fn detect_metal_gpus() -> usize {
        #[cfg(target_os = "macos")]
        {
            Self::macos_metal_gpus()
        }

        #[cfg(not(target_os = "macos"))]
        {
            0 // Metal only available on macOS
        }
    }

    /// Detect CUDA GPUs (NVIDIA)
    fn detect_cuda_gpus() -> usize {
        // Check for nvidia-smi or CUDA runtime
        Self::nvidia_gpu_count().unwrap_or(0)
    }

    /// Detect ROCm GPUs (AMD)
    fn detect_rocm_gpus() -> usize {
        // Check for rocm-smi or ROCm runtime
        Self::amd_gpu_count().unwrap_or(0)
    }

    /// Get total accelerator count
    pub fn total_accelerators(&self) -> usize {
        self.metal_gpus + self.cuda_gpus + self.rocm_gpus
    }

    /// Get optimal thread count for CPU inference
    pub fn optimal_cpu_threads(&self) -> usize {
        match self.logical_cpus {
            1..=4 => self.logical_cpus,       // Use all on low-core systems
            5..=8 => self.logical_cpus - 1,   // Leave 1 core for OS
            9..=16 => self.logical_cpus - 2,  // Leave 2 cores for OS
            17..=32 => self.logical_cpus - 4, // Leave 4 cores for OS on high-end
            _ => 28,                          // Cap at 28 for extreme systems
        }
    }

    /// Get best available accelerator
    pub fn best_accelerator(&self) -> AcceleratorType {
        if self.metal_gpus > 0 {
            AcceleratorType::Metal
        } else if self.cuda_gpus > 0 {
            AcceleratorType::Cuda
        } else if self.rocm_gpus > 0 {
            AcceleratorType::Rocm
        } else {
            AcceleratorType::Cpu
        }
    }
}

// Platform-specific implementations
impl SystemInfo {
    /// Fallback CPU detection using /proc/cpuinfo
    fn fallback_cpu_detection() -> usize {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
                let count = contents
                    .lines()
                    .filter(|line| line.starts_with("processor"))
                    .count();
                if count > 0 {
                    return count;
                }
            }
        }

        // Ultimate fallback
        1
    }

    #[cfg(target_os = "macos")]
    fn macos_physical_cpus() -> Option<usize> {
        use std::process::Command;

        let output = Command::new("sysctl")
            .args(&["-n", "hw.physicalcpu"])
            .output()
            .ok()?;

        let count_str = String::from_utf8_lossy(&output.stdout);
        count_str.trim().parse().ok()
    }

    #[cfg(target_os = "linux")]
    fn linux_physical_cpus() -> Option<usize> {
        let contents = std::fs::read_to_string("/proc/cpuinfo").ok()?;

        // Count unique physical IDs
        let mut physical_ids = std::collections::HashSet::new();
        for line in contents.lines() {
            if line.starts_with("physical id") {
                if let Some(id) = line.split(':').nth(1) {
                    physical_ids.insert(id.trim());
                }
            }
        }

        Some(physical_ids.len().max(1))
    }

    #[cfg(target_os = "macos")]
    fn macos_metal_gpus() -> usize {
        use std::process::Command;

        // Use system_profiler to detect GPUs
        if let Ok(output) = Command::new("system_profiler")
            .args(&["SPDisplaysDataType"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);

            // Look for Apple Silicon GPU indicators
            if output_str.contains("Apple M") || output_str.contains("Metal") {
                return 1; // M-series chips have 1 integrated GPU
            }
        }

        // Fallback: assume M-series has GPU
        if std::env::consts::ARCH == "aarch64" {
            1
        } else {
            0
        }
    }

    /// Detect NVIDIA GPUs via nvidia-smi
    fn nvidia_gpu_count() -> Option<usize> {
        use std::process::Command;

        let output = Command::new("nvidia-smi")
            .args(&["-L"]) // List GPUs
            .output()
            .ok()?;

        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let count = output_str
                .lines()
                .filter(|line| line.contains("GPU"))
                .count();
            Some(count)
        } else {
            None
        }
    }

    /// Detect AMD GPUs via rocm-smi
    fn amd_gpu_count() -> Option<usize> {
        use std::process::Command;

        let output = Command::new("rocm-smi")
            .args(&["-i"]) // Show GPU info
            .output()
            .ok()?;

        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let count = output_str
                .lines()
                .filter(|line| line.contains("GPU"))
                .count();
            Some(count)
        } else {
            // Alternative: check /sys/class/drm
            Self::linux_drm_gpu_count()
        }
    }

    #[cfg(target_os = "linux")]
    fn linux_drm_gpu_count() -> Option<usize> {
        let drm_path = "/sys/class/drm";
        let entries = std::fs::read_dir(drm_path).ok()?;

        let count = entries
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("card"))
            .count();

        if count > 0 { Some(count) } else { None }
    }

    #[cfg(not(target_os = "linux"))]
    fn linux_drm_gpu_count() -> Option<usize> {
        None
    }
}

// Display formatting
impl fmt::Display for SystemInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CPUs: {} logical, {} physical",
            self.logical_cpus, self.physical_cpus
        )?;

        if self.total_accelerators() > 0 {
            write!(f, ", GPUs: ")?;
            let mut gpu_parts = Vec::new();

            if self.metal_gpus > 0 {
                gpu_parts.push(format!("{} Metal", self.metal_gpus));
            }
            if self.cuda_gpus > 0 {
                gpu_parts.push(format!("{} CUDA", self.cuda_gpus));
            }
            if self.rocm_gpus > 0 {
                gpu_parts.push(format!("{} ROCm", self.rocm_gpus));
            }

            write!(f, "{}", gpu_parts.join(", "))?;
        }

        Ok(())
    }
}

impl fmt::Display for AcceleratorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            AcceleratorType::Cpu => "CPU",
            AcceleratorType::Metal => "Metal",
            AcceleratorType::Cuda => "CUDA",
            AcceleratorType::Rocm => "ROCm",
        };
        write!(f, "{}", name)
    }
}
