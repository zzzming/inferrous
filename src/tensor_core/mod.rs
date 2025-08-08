//! From-scratch tensor system for neural network inference
//!
//! This module provides a complete tensor implementation built from the ground up:
//! - **Core tensor** with Vec<f32> storage and row-major layout
//! - **Essential operations** for neural networks (matmul, activations, etc.)
//! - **Memory management** with views, slicing, and efficient layouts
//! - **Demonstrations** showing real neural network computations

use anyhow::{Result, anyhow};
use std::fmt;

// Include mathematical operations and demonstrations
pub mod ops;
pub mod demo;

// Re-export main functions for convenience
pub use demo::{run_all_demos, demo_basic_tensors, demo_neural_network_ops, demo_attention_mechanism};

/// Core tensor structure for neural network computations
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Flat storage of all tensor data in row-major order
    data: Vec<f32>,
    /// Shape of the tensor (dimensions)
    shape: Vec<usize>,
    /// Strides for indexing (how many elements to skip for each dimension)
    strides: Vec<usize>,
    /// Offset into the data array (for views/slices)
    offset: usize,
}

impl Tensor {
    /// Create a new tensor with given shape, initialized with zeros
    pub fn zeros(shape: &[usize]) -> Self {
        let total_elements = shape.iter().product();
        let strides = Self::compute_strides(shape);
        
        Self {
            data: vec![0.0; total_elements],
            shape: shape.to_vec(),
            strides,
            offset: 0,
        }
    }
    
    /// Create a new tensor with given shape, initialized with ones
    pub fn ones(shape: &[usize]) -> Self {
        let total_elements = shape.iter().product();
        let strides = Self::compute_strides(shape);
        
        Self {
            data: vec![1.0; total_elements],
            shape: shape.to_vec(),
            strides,
            offset: 0,
        }
    }
    
    /// Create a tensor from raw data and shape
    pub fn from_data(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(anyhow!(
                "Data length {} doesn't match expected size {} for shape {:?}",
                data.len(), expected_size, shape
            ));
        }
        
        let strides = Self::compute_strides(shape);
        
        Ok(Self {
            data,
            shape: shape.to_vec(),
            strides,
            offset: 0,
        })
    }
    
    /// Create a 1D tensor from a slice
    pub fn from_slice(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
            shape: vec![data.len()],
            strides: vec![1],
            offset: 0,
        }
    }
    
    /// Create a 2D tensor (matrix) from nested slices
    pub fn from_2d(data: &[&[f32]]) -> Result<Self> {
        if data.is_empty() {
            return Err(anyhow!("Cannot create tensor from empty data"));
        }
        
        let rows = data.len();
        let cols = data[0].len();
        
        // Verify all rows have same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(anyhow!(
                    "Row {} has length {}, expected {}",
                    i, row.len(), cols
                ));
            }
        }
        
        // Flatten data in row-major order
        let flat_data: Vec<f32> = data.iter().flat_map(|row| row.iter().copied()).collect();
        
        Self::from_data(flat_data, &[rows, cols])
    }
    
    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    
    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Check if the tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = Self::compute_strides(&self.shape);
        self.strides == expected_strides && self.offset == 0
    }
    
    /// Get a reference to the underlying data (only for contiguous tensors)
    pub fn as_slice(&self) -> Result<&[f32]> {
        if !self.is_contiguous() {
            return Err(anyhow!("Tensor is not contiguous"));
        }
        Ok(&self.data)
    }
    
    /// Get element at given indices
    pub fn get(&self, indices: &[usize]) -> Result<f32> {
        if indices.len() != self.shape.len() {
            return Err(anyhow!(
                "Expected {} indices, got {}",
                self.shape.len(), indices.len()
            ));
        }
        
        let flat_index = self.compute_flat_index(indices)?;
        Ok(self.data[flat_index])
    }
    
    /// Set element at given indices
    pub fn set(&mut self, indices: &[usize], value: f32) -> Result<()> {
        if indices.len() != self.shape.len() {
            return Err(anyhow!(
                "Expected {} indices, got {}",
                self.shape.len(), indices.len()
            ));
        }
        
        let flat_index = self.compute_flat_index(indices)?;
        self.data[flat_index] = value;
        Ok(())
    }
    
    /// Compute strides for given shape (row-major order)
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
    
    /// Compute flat index from multi-dimensional indices
    fn compute_flat_index(&self, indices: &[usize]) -> Result<usize> {
        // Bounds checking
        for (i, (&index, &dim_size)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if index >= dim_size {
                return Err(anyhow!(
                    "Index {} is out of bounds for dimension {} (size {})",
                    index, i, dim_size
                ));
            }
        }
        
        let mut flat_index = self.offset;
        for (&index, &stride) in indices.iter().zip(self.strides.iter()) {
            flat_index += index * stride;
        }
        
        if flat_index >= self.data.len() {
            return Err(anyhow!(
                "Computed index {} is out of bounds for data length {}",
                flat_index, self.data.len()
            ));
        }
        
        Ok(flat_index)
    }
    
    /// Reshape tensor to new shape (metadata-only operation)
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(anyhow!(
                "Cannot reshape tensor with {} elements to shape {:?} ({} elements)",
                self.numel(), new_shape, new_numel
            ));
        }
        
        if !self.is_contiguous() {
            return Err(anyhow!(
                "Cannot reshape non-contiguous tensor. Use .contiguous() first."
            ));
        }
        
        let new_strides = Self::compute_strides(new_shape);
        
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            strides: new_strides,
            offset: self.offset,
        })
    }
    
    /// Transpose the tensor (swap last two dimensions)
    pub fn transpose(&self) -> Result<Tensor> {
        match self.ndim() {
            0 | 1 => Ok(self.clone()), // No-op for 0D and 1D tensors
            2 => self.transpose_2d(),
            _ => Err(anyhow!("Transpose only implemented for 2D tensors"))
        }
    }
    
    /// Transpose 2D tensor (swap rows and columns)
    fn transpose_2d(&self) -> Result<Tensor> {
        let (_rows, _cols) = (self.shape[0], self.shape[1]);
        
        // Create new tensor with swapped dimensions
        let mut new_shape = self.shape.clone();
        new_shape.swap(0, 1);
        
        let mut new_strides = self.strides.clone();
        new_strides.swap(0, 1);
        
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }
    
    /// Create a contiguous copy of the tensor
    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        
        // Create new contiguous tensor by copying data in the right order
        let mut new_data = Vec::with_capacity(self.numel());
        let indices = self.all_indices();
        
        for idx in indices {
            new_data.push(self.get(&idx)?);
        }
        
        Tensor::from_data(new_data, &self.shape)
    }
    
    /// Create a view (slice) of the tensor
    /// 
    /// For now, only support slicing along the first dimension
    pub fn slice(&self, start: usize, end: usize) -> Result<Tensor> {
        if self.ndim() == 0 {
            return Err(anyhow!("Cannot slice 0-dimensional tensor"));
        }
        
        let dim_size = self.shape[0];
        if start >= dim_size || end > dim_size || start >= end {
            return Err(anyhow!(
                "Invalid slice [{}, {}) for dimension of size {}",
                start, end, dim_size
            ));
        }
        
        let mut new_shape = self.shape.clone();
        new_shape[0] = end - start;
        
        let new_offset = self.offset + start * self.strides[0];
        
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
        })
    }
    
    /// Generate all valid indices for this tensor (used internally)
    pub fn all_indices(&self) -> Vec<Vec<usize>> {
        let mut indices = Vec::new();
        self.generate_indices(&mut vec![], 0, &mut indices);
        indices
    }
    
    /// Recursively generate all indices (used internally)
    pub fn generate_indices(&self, current: &mut Vec<usize>, dim: usize, all: &mut Vec<Vec<usize>>) {
        if dim == self.shape().len() {
            all.push(current.clone());
            return;
        }
        
        for i in 0..self.shape()[dim] {
            current.push(i);
            self.generate_indices(current, dim + 1, all);
            current.pop();
        }
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor> {
        if self.shape.len() != shape.len() {
            return Err(anyhow!("Cannot broadcast tensor with shape {:?} to shape {:?}", self.shape, shape));
        }
        
        let mut new_data = Vec::with_capacity(shape.iter().product());
        
        for i in 0..shape.iter().product() {
            new_data.push(self.data[i]);
        }
        
        Ok(Tensor {
            data: new_data,
            shape: shape.to_vec(),
            strides: self.strides.clone(),
            offset: self.offset,
        })
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ndim() {
            0 => {
                // Scalar
                write!(f, "{}", self.data[self.offset])
            }
            1 => {
                // Vector
                write!(f, "[")?;
                for i in 0..self.shape[0] {
                    if i > 0 { write!(f, ", ")?; }
                    let val = self.get(&[i]).unwrap();
                    write!(f, "{:.4}", val)?;
                }
                write!(f, "]")
            }
            2 => {
                // Matrix
                writeln!(f, "[")?;
                for i in 0..self.shape[0] {
                    write!(f, "  [")?;
                    for j in 0..self.shape[1] {
                        if j > 0 { write!(f, ", ")?; }
                        let val = self.get(&[i, j]).unwrap();
                        write!(f, "{:8.4}", val)?;
                    }
                    writeln!(f, "]")?;
                }
                write!(f, "]")
            }
            _ => {
                // Higher dimensions - just show shape and some stats
                let min_val = self.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                write!(f, "Tensor(shape={:?}, min={:.4}, max={:.4})", self.shape, min_val, max_val)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_creation() {
        // Test zeros
        let zeros = Tensor::zeros(&[2, 3]);
        assert_eq!(zeros.shape(), &[2, 3]);
        assert_eq!(zeros.numel(), 6);
        assert_eq!(zeros.get(&[0, 0]).unwrap(), 0.0);
        assert_eq!(zeros.get(&[1, 2]).unwrap(), 0.0);
        
        // Test ones
        let ones = Tensor::ones(&[2, 2]);
        assert_eq!(ones.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(ones.get(&[1, 1]).unwrap(), 1.0);
    }
    
    #[test]
    fn test_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_data(data, &[2, 3]).unwrap();
        
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(tensor.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(tensor.get(&[1, 0]).unwrap(), 4.0);
        assert_eq!(tensor.get(&[1, 2]).unwrap(), 6.0);
    }
    
    #[test]
    fn test_from_2d() {
        let data = &[
            &[1.0, 2.0, 3.0][..],
            &[4.0, 5.0, 6.0][..],
        ];
        let tensor = Tensor::from_2d(data).unwrap();
        
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(tensor.get(&[1, 2]).unwrap(), 6.0);
    }
    
    #[test]
    fn test_indexing() {
        let mut tensor = Tensor::zeros(&[3, 4]);
        
        // Test setting and getting values
        tensor.set(&[1, 2], 42.0).unwrap();
        assert_eq!(tensor.get(&[1, 2]).unwrap(), 42.0);
        
        // Test bounds checking
        assert!(tensor.get(&[3, 0]).is_err()); // Row out of bounds
        assert!(tensor.get(&[0, 4]).is_err()); // Column out of bounds
        assert!(tensor.set(&[3, 0], 1.0).is_err());
    }
    
    #[test]
    fn test_stride_computation() {
        let strides = Tensor::compute_strides(&[2, 3, 4]);
        assert_eq!(strides, vec![12, 4, 1]);
        
        let strides = Tensor::compute_strides(&[5]);
        assert_eq!(strides, vec![1]);
        
        let strides = Tensor::compute_strides(&[2, 3]);
        assert_eq!(strides, vec![3, 1]);
    }
    
    #[test]
    fn test_contiguity() {
        let tensor = Tensor::zeros(&[2, 3]);
        assert!(tensor.is_contiguous());
        
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, &[2, 2]).unwrap();
        assert!(tensor.is_contiguous());
    }
    
    #[test]
    fn test_tensor_properties() {
        let tensor = Tensor::zeros(&[3, 4, 5]);
        assert_eq!(tensor.ndim(), 3);
        assert_eq!(tensor.numel(), 60);
        assert_eq!(tensor.shape(), &[3, 4, 5]);
    }
    
    #[test]
    fn test_reshape() {
        let tensor = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        
        // Reshape to 1D
        let reshaped = tensor.reshape(&[6]).unwrap();
        assert_eq!(reshaped.shape(), &[6]);
        assert_eq!(reshaped.get(&[0]).unwrap(), 1.0);
        assert_eq!(reshaped.get(&[5]).unwrap(), 6.0);
        
        // Reshape to 3x2
        let reshaped = tensor.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(reshaped.get(&[2, 1]).unwrap(), 6.0);
        
        // Error: wrong total elements
        assert!(tensor.reshape(&[2, 4]).is_err());
    }
    
    #[test]
    fn test_transpose() {
        let tensor = Tensor::from_2d(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
        ]).unwrap();
        
        let transposed = tensor.transpose().unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        
        // Check values are correctly transposed
        assert_eq!(transposed.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(transposed.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(transposed.get(&[1, 0]).unwrap(), 2.0);
        assert_eq!(transposed.get(&[1, 1]).unwrap(), 5.0);
        assert_eq!(transposed.get(&[2, 0]).unwrap(), 3.0);
        assert_eq!(transposed.get(&[2, 1]).unwrap(), 6.0);
    }
    
    #[test]
    fn test_slicing() {
        let tensor = Tensor::from_2d(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
        ]).unwrap();
        
        // Slice first two rows
        let slice = tensor.slice(0, 2).unwrap();
        assert_eq!(slice.shape(), &[2, 3]);
        assert_eq!(slice.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(slice.get(&[1, 2]).unwrap(), 6.0);
        
        // Slice single row
        let row = tensor.slice(1, 2).unwrap();
        assert_eq!(row.shape(), &[1, 3]);
        assert_eq!(row.get(&[0, 0]).unwrap(), 4.0);
        assert_eq!(row.get(&[0, 2]).unwrap(), 6.0);
        
        // Error: invalid range
        assert!(tensor.slice(2, 1).is_err()); // start >= end
        assert!(tensor.slice(3, 4).is_err()); // start >= dim_size
    }
    
    #[test]
    fn test_contiguous() {
        let tensor = Tensor::from_2d(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
        ]).unwrap();
        
        // Original tensor is contiguous
        assert!(tensor.is_contiguous());
        let cont = tensor.contiguous().unwrap();
        assert!(cont.is_contiguous());
        
        // After transpose, not contiguous
        let transposed = tensor.transpose().unwrap();
        assert!(!transposed.is_contiguous());
        
        // But we can make it contiguous
        let cont_transposed = transposed.contiguous().unwrap();
        assert!(cont_transposed.is_contiguous());
        assert_eq!(cont_transposed.shape(), &[3, 2]);
        assert_eq!(cont_transposed.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(cont_transposed.get(&[0, 1]).unwrap(), 4.0);
    }
    
    #[test]
    fn test_error_handling() {
        // Wrong data size
        let data = vec![1.0, 2.0, 3.0];
        assert!(Tensor::from_data(data, &[2, 3]).is_err());
        
        // Empty 2D data
        assert!(Tensor::from_2d(&[]).is_err());
        
        // Inconsistent row lengths
        let data = &[
            &[1.0, 2.0][..],
            &[3.0, 4.0, 5.0][..],
        ];
        assert!(Tensor::from_2d(data).is_err());
    }
}