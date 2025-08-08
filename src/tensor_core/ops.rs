//! Essential tensor operations for neural network inference
//! 
//! This module contains the core mathematical operations needed for
//! neural networks: matrix multiplication, element-wise operations,
//! reductions, and activations.

use super::Tensor;
use anyhow::{Result, anyhow};

impl Tensor {
    /// Matrix multiplication: self @ other
    /// 
    /// Supports:
    /// - 2D @ 2D: Standard matrix multiplication
    /// - 2D @ 1D: Matrix-vector multiplication  
    /// - 1D @ 2D: Vector-matrix multiplication
    /// - Batched operations with broadcasting
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        match (self.ndim(), other.ndim()) {
            (2, 2) => self.matmul_2d_2d(other),
            (2, 1) => self.matmul_2d_1d(other),
            (1, 2) => self.matmul_1d_2d(other),
            (1, 1) => self.dot(other), // Dot product
            _ => Err(anyhow!(
                "Unsupported matmul dimensions: {:?} @ {:?}",
                self.shape(), other.shape()
            ))
        }
    }
    
    /// 2D matrix multiplication (M, K) @ (K, N) -> (M, N)
    fn matmul_2d_2d(&self, other: &Tensor) -> Result<Tensor> {
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (k2, n) = (other.shape()[0], other.shape()[1]);
        
        if k != k2 {
            return Err(anyhow!(
                "Matrix dimensions don't match for multiplication: ({}, {}) @ ({}, {})",
                m, k, k2, n
            ));
        }
        
        let mut result = Tensor::zeros(&[m, n]);
        
        // Naive O(n³) implementation - optimize later
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k_idx in 0..k {
                    sum += self.get(&[i, k_idx])? * other.get(&[k_idx, j])?;
                }
                result.set(&[i, j], sum)?;
            }
        }
        
        Ok(result)
    }
    
    /// Matrix-vector multiplication (M, K) @ (K,) -> (M,)
    fn matmul_2d_1d(&self, other: &Tensor) -> Result<Tensor> {
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let k2 = other.shape()[0];
        
        if k != k2 {
            return Err(anyhow!(
                "Dimensions don't match: ({}, {}) @ ({})",
                m, k, k2
            ));
        }
        
        let mut result = Tensor::zeros(&[m]);
        
        for i in 0..m {
            let mut sum = 0.0;
            for k_idx in 0..k {
                sum += self.get(&[i, k_idx])? * other.get(&[k_idx])?;
            }
            result.set(&[i], sum)?;
        }
        
        Ok(result)
    }
    
    /// Vector-matrix multiplication (K,) @ (K, N) -> (N,)
    fn matmul_1d_2d(&self, other: &Tensor) -> Result<Tensor> {
        let k = self.shape()[0];
        let (k2, n) = (other.shape()[0], other.shape()[1]);
        
        if k != k2 {
            return Err(anyhow!(
                "Dimensions don't match: ({}) @ ({}, {})",
                k, k2, n
            ));
        }
        
        let mut result = Tensor::zeros(&[n]);
        
        for j in 0..n {
            let mut sum = 0.0;
            for k_idx in 0..k {
                sum += self.get(&[k_idx])? * other.get(&[k_idx, j])?;
            }
            result.set(&[j], sum)?;
        }
        
        Ok(result)
    }
    
    /// Dot product for 1D tensors
    fn dot(&self, other: &Tensor) -> Result<Tensor> {
        let n = self.shape()[0];
        let n2 = other.shape()[0];
        
        if n != n2 {
            return Err(anyhow!(
                "Vector lengths don't match: {} vs {}",
                n, n2
            ));
        }
        
        let mut sum = 0.0;
        for i in 0..n {
            sum += self.get(&[i])? * other.get(&[i])?;
        }
        
        // Return scalar as 0D tensor
        Tensor::from_data(vec![sum], &[])
    }
    
    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        self.elementwise_op(other, |a, b| a + b)
    }
    
    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        self.elementwise_op(other, |a, b| a - b)
    }
    
    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        self.elementwise_op(other, |a, b| a * b)
    }
    
    /// Element-wise division
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        self.elementwise_op(other, |a, b| a / b)
    }
    
    /// Add scalar to all elements
    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.scalar_op(|x| x + scalar)
    }
    
    /// Multiply all elements by scalar
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.scalar_op(|x| x * scalar)
    }
    
    /// Generic element-wise operation
    fn elementwise_op<F>(&self, other: &Tensor, op: F) -> Result<Tensor>
    where
        F: Fn(f32, f32) -> f32,
    {
        // For now, require exact shape match (no broadcasting)
        if self.shape() != other.shape() {
            return Err(anyhow!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(), other.shape()
            ));
        }
        
        let mut result = Tensor::zeros(self.shape());
        
        // Apply operation element by element
        let indices = self.all_indices();
        for idx in indices {
            let a = self.get(&idx)?;
            let b = other.get(&idx)?;
            result.set(&idx, op(a, b))?;
        }
        
        Ok(result)
    }
    
    /// Generic scalar operation
    fn scalar_op<F>(&self, op: F) -> Result<Tensor>
    where
        F: Fn(f32) -> f32,
    {
        let mut result = Tensor::zeros(self.shape());
        
        let indices = self.all_indices();
        for idx in indices {
            let val = self.get(&idx)?;
            result.set(&idx, op(val))?;
        }
        
        Ok(result)
    }
    
    
    /// Sum all elements
    pub fn sum(&self) -> Result<f32> {
        let mut total = 0.0;
        let indices = self.all_indices();
        for idx in indices {
            total += self.get(&idx)?;
        }
        Ok(total)
    }
    
    /// Sum along a specific axis
    pub fn sum_axis(&self, axis: usize) -> Result<Tensor> {
        if axis >= self.ndim() {
            return Err(anyhow!("Axis {} out of bounds for tensor with {} dimensions", axis, self.ndim()));
        }
        
        // Compute result shape (remove the summed axis)
        let mut result_shape = self.shape().to_vec();
        result_shape.remove(axis);
        
        // Handle scalar result case
        if result_shape.is_empty() {
            let total = self.sum()?;
            return Tensor::from_data(vec![total], &[]);
        }
        
        let mut result = Tensor::zeros(&result_shape);
        
        // Sum over the specified axis
        let indices = self.all_indices();
        for idx in indices {
            // Create result index by removing the axis dimension
            let mut result_idx = idx.clone();
            result_idx.remove(axis);
            
            let current_val = result.get(&result_idx)?;
            let add_val = self.get(&idx)?;
            result.set(&result_idx, current_val + add_val)?;
        }
        
        Ok(result)
    }
    
    /// Mean of all elements
    pub fn mean(&self) -> Result<f32> {
        let total = self.sum()?;
        Ok(total / self.numel() as f32)
    }
    
    /// Apply ReLU activation (max(0, x))
    pub fn relu(&self) -> Result<Tensor> {
        self.scalar_op(|x| x.max(0.0))
    }
    
    /// Apply SiLU/Swish activation (x * sigmoid(x))
    pub fn silu(&self) -> Result<Tensor> {
        self.scalar_op(|x| x * (1.0 / (1.0 + (-x).exp())))
    }
    
    /// Apply GELU activation (approximate)
    pub fn gelu(&self) -> Result<Tensor> {
        self.scalar_op(|x| {
            0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
        })
    }
    
    /// Apply softmax along the last dimension
    pub fn softmax(&self) -> Result<Tensor> {
        match self.ndim() {
            1 => self.softmax_1d(),
            2 => self.softmax_2d(),
            _ => Err(anyhow!("Softmax only supported for 1D and 2D tensors"))
        }
    }
    
    /// 1D softmax
    fn softmax_1d(&self) -> Result<Tensor> {
        let n = self.shape()[0];
        
        // Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..n {
            max_val = max_val.max(self.get(&[i])?);
        }
        
        // Compute exp(x - max)
        let mut exp_vals = Vec::new();
        let mut sum = 0.0;
        for i in 0..n {
            let exp_val = (self.get(&[i])? - max_val).exp();
            exp_vals.push(exp_val);
            sum += exp_val;
        }
        
        // Normalize
        let normalized: Vec<f32> = exp_vals.into_iter().map(|x| x / sum).collect();
        Tensor::from_data(normalized, &[n])
    }
    
    /// 2D softmax (along last dimension)
    fn softmax_2d(&self) -> Result<Tensor> {
        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let mut result = Tensor::zeros(&[rows, cols]);
        
        for i in 0..rows {
            // Extract row
            let mut row_data = Vec::new();
            for j in 0..cols {
                row_data.push(self.get(&[i, j])?);
            }
            
            let row_tensor = Tensor::from_slice(&row_data);
            let softmax_row = row_tensor.softmax_1d()?;
            
            // Copy back to result
            for j in 0..cols {
                result.set(&[i, j], softmax_row.get(&[j])?)?;
            }
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matrix_multiplication() {
        // Test 2D @ 2D
        let a = Tensor::from_2d(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
        ]).unwrap();
        
        let b = Tensor::from_2d(&[
            &[5.0, 6.0],
            &[7.0, 8.0],
        ]).unwrap();
        
        let c = a.matmul(&b).unwrap();
        
        // Expected: [[19, 22], [43, 50]]
        assert_eq!(c.get(&[0, 0]).unwrap(), 19.0);
        assert_eq!(c.get(&[0, 1]).unwrap(), 22.0);
        assert_eq!(c.get(&[1, 0]).unwrap(), 43.0);
        assert_eq!(c.get(&[1, 1]).unwrap(), 50.0);
    }
    
    #[test]
    fn test_matrix_vector_multiplication() {
        let matrix = Tensor::from_2d(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
        ]).unwrap();
        
        let vector = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        
        let result = matrix.matmul(&vector).unwrap();
        
        // Expected: [14, 32] (1*1 + 2*2 + 3*3 = 14, 4*1 + 5*2 + 6*3 = 32)
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.get(&[0]).unwrap(), 14.0);
        assert_eq!(result.get(&[1]).unwrap(), 32.0);
    }
    
    #[test]
    fn test_dot_product() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        
        let result = a.matmul(&b).unwrap();
        
        // Expected: 1*4 + 2*5 + 3*6 = 32
        assert_eq!(result.get(&[]).unwrap(), 32.0);
    }
    
    #[test]
    fn test_element_wise_operations() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        
        // Addition
        let sum = a.add(&b).unwrap();
        assert_eq!(sum.get(&[0]).unwrap(), 5.0);
        assert_eq!(sum.get(&[1]).unwrap(), 7.0);
        assert_eq!(sum.get(&[2]).unwrap(), 9.0);
        
        // Multiplication
        let prod = a.mul(&b).unwrap();
        assert_eq!(prod.get(&[0]).unwrap(), 4.0);
        assert_eq!(prod.get(&[1]).unwrap(), 10.0);
        assert_eq!(prod.get(&[2]).unwrap(), 18.0);
    }
    
    #[test]
    fn test_scalar_operations() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        
        let add_result = tensor.add_scalar(10.0).unwrap();
        assert_eq!(add_result.get(&[0]).unwrap(), 11.0);
        assert_eq!(add_result.get(&[1]).unwrap(), 12.0);
        assert_eq!(add_result.get(&[2]).unwrap(), 13.0);
        
        let mul_result = tensor.mul_scalar(2.0).unwrap();
        assert_eq!(mul_result.get(&[0]).unwrap(), 2.0);
        assert_eq!(mul_result.get(&[1]).unwrap(), 4.0);
        assert_eq!(mul_result.get(&[2]).unwrap(), 6.0);
    }
    
    #[test]
    fn test_reductions() {
        let tensor = Tensor::from_2d(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
        ]).unwrap();
        
        // Sum all elements
        assert_eq!(tensor.sum().unwrap(), 21.0);
        
        // Mean
        assert_eq!(tensor.mean().unwrap(), 3.5);
        
        // Sum along axis 0 (columns)
        let col_sum = tensor.sum_axis(0).unwrap();
        assert_eq!(col_sum.shape(), &[3]);
        assert_eq!(col_sum.get(&[0]).unwrap(), 5.0); // 1 + 4
        assert_eq!(col_sum.get(&[1]).unwrap(), 7.0); // 2 + 5
        assert_eq!(col_sum.get(&[2]).unwrap(), 9.0); // 3 + 6
        
        // Sum along axis 1 (rows)
        let row_sum = tensor.sum_axis(1).unwrap();
        assert_eq!(row_sum.shape(), &[2]);
        assert_eq!(row_sum.get(&[0]).unwrap(), 6.0);  // 1 + 2 + 3
        assert_eq!(row_sum.get(&[1]).unwrap(), 15.0); // 4 + 5 + 6
    }
    
    #[test]
    fn test_activations() {
        let tensor = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        
        // ReLU
        let relu_result = tensor.relu().unwrap();
        assert_eq!(relu_result.get(&[0]).unwrap(), 0.0);
        assert_eq!(relu_result.get(&[1]).unwrap(), 0.0);
        assert_eq!(relu_result.get(&[2]).unwrap(), 0.0);
        assert_eq!(relu_result.get(&[3]).unwrap(), 1.0);
        assert_eq!(relu_result.get(&[4]).unwrap(), 2.0);
        
        // SiLU should give reasonable values
        let silu_result = tensor.silu().unwrap();
        assert!(silu_result.get(&[0]).unwrap() < 0.0); // Negative input
        assert!(silu_result.get(&[4]).unwrap() > 1.0); // Positive input > 1
    }
    
    #[test]
    fn test_softmax() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let softmax_result = tensor.softmax().unwrap();
        
        // Check that probabilities sum to 1
        let sum = softmax_result.sum().unwrap();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that largest input gives largest probability
        let max_idx = (0..3)
            .max_by(|&i, &j| {
                softmax_result.get(&[i]).unwrap()
                    .partial_cmp(&softmax_result.get(&[j]).unwrap()).unwrap()
            })
            .unwrap();
        assert_eq!(max_idx, 2); // Index of largest input (3.0)
    }
    
    #[test]
    fn test_dimension_errors() {
        let a = Tensor::zeros(&[2, 3]);
        let b = Tensor::zeros(&[4, 3]); // Wrong dimensions
        
        // Should fail due to dimension mismatch
        assert!(a.matmul(&b).is_err());
        
        // Element-wise operations should fail with different shapes
        assert!(a.add(&b).is_err());
    }
}