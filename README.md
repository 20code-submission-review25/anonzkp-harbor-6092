

zkMaP: Zero-Knowledge Succinct Non-Interactive Matrix Multiplication Proofs


This repository contains an efficient implementation of zero-knowledge proofs for matrix multiplication using KZG polynomial commitments. Our implementation, ZK-MaP (Zero-Knowledge Succinct Non-Interactive Matrix Multiplication Proofs), provides high-performance verification of matrix operations with constant-sized proofs, designed for cryptographic applications.

OVERVIEW

ZK-MaP enables a prover to demonstrate that C = A × B for matrices A, B, and C without revealing their contents. Built on the KZG commitment scheme, the protocol offers the following advantages:

- Constant-Sized Proofs: The proof size remains fixed regardless of the matrix dimensions.
- Efficient Verification: Verification time is independent of the matrix size (constant time complexity).
- Batched Proofs: Supports efficient amortization when proving multiple operations.
- Non-Interactive Proofs: Achieved via a Fiat-Shamir transformation.

Cryptographic Foundations:
- KZG Commitments: Ensure binding and correct evaluation under the Polynomial Discrete Logarithm assumption.
- Fiat-Shamir Transform: Converts interactive proofs into non-interactive ones using a secure hash function.
- Trusted Setup: Although this demo uses a basic setup, production systems should adopt a secure multi-party computation (MPC) ceremony.

PERFORMANCE

Preliminary benchmarks on a typical CPU (single-threaded) are as follows:

Matrix Size | Proof Generation | Verification | Proof Size
-----------|------------------|--------------|---------------
128×128     | 243 ms           | 3.66 ms      | 320 bytes
256×256     | 859 ms           | 3.67 ms      | 320 bytes
512×512     | 3.28 s           | 3.70 ms      | 320 bytes
1024×1024   | 12.3 s           | 3.69 ms      | 320 bytes

Verification is orders of magnitude faster than recomputation, demonstrating the efficiency of our approach.

REQUIREMENTS

- Rust (2018 edition or later)
- For visualization: Python 3.6+ with matplotlib, pandas, and numpy

INSTALLATION

Clone the repository:

git clone https://github.com/20code-submission-review25/anonzkp-harbor-6092
cd zk-map

Build the project:

cargo build --release

REPOSITORY STRUCTURE

- src/
  - main.rs: Demonstrates core functionality and entry point for running benchmarks
  - kzg.rs: Implements KZG polynomial commitments
  - zk_matrix.rs: Encodes matrix polynomial proofs, including proof generation and verification
  - benchmark.rs: Contains the benchmarking suite for performance and scalability tests
  - utils.rs: Utility functions for polynomial operations
- plot.py: Script for visualizing benchmark results

USAGE

Basic Example:

use zk_matrix_proofs::{
    kzg::KZG,
    zk_matrix::ZKMatrixProof,
    ark_bls12_381::{Bls12_381, Fr, G1Projective as G1, G2Projective as G2},
    ark_std::UniformRand,
};

// Initialize KZG instance
let mut rng = ark_std::test_rng();
let degree = 64; // Must be >= max matrix dimension squared
let mut kzg_instance = KZG::<Bls12_381>::new(
    G1::rand(&mut rng),
    G2::rand(&mut rng),
    degree
);

// Trusted setup
let secret = Fr::rand(&mut rng);
kzg_instance.setup(secret);

// Create ZKMatrixProof instance
let zk_matrix = ZKMatrixProof::new(kzg_instance, degree);

// Generate random matrices
let a_matrix = generate_random_matrix::<Fr>(8, 8, &mut rng);
let b_matrix = generate_random_matrix::<Fr>(8, 8, &mut rng);

// Generate proof
let proof = zk_matrix.prove_matrix_mult(&a_matrix, &b_matrix);

// Verify proof
let result = zk_matrix.verify(&proof);
assert!(result, "Verification failed");

Batched Proofs:

use zk_matrix_proofs::zk_matrix::OptimizedBatchedZKMatrixProof;

// Create batched instance
let batched_zk = OptimizedBatchedZKMatrixProof::new(&zk_matrix);

// Generate multiple matrix pairs
let matrices_a = vec![
    generate_random_matrix::<Fr>(16, 16, &mut rng),
    generate_random_matrix::<Fr>(16, 16, &mut rng),
];
let matrices_b = vec![
    generate_random_matrix::<Fr>(16, 16, &mut rng),
    generate_random_matrix::<Fr>(16, 16, &mut rng),
];

// Generate batched proof
let batched_proof = batched_zk.prove_batched_matrix_mult(&matrices_a, &matrices_b);

// Verify batched proof
let result = batched_zk.verify_batched(&batched_proof);
assert!(result, "Batch verification failed");

BENCHMARKING

Run the benchmarks:

cargo run --release

This will generate CSV files with benchmark results:
- zk_matrix_benchmark.csv: Basic performance metrics
- zk_vs_nonzk_comparison.csv: Comparison with standard matrix multiplication
- batch_efficiency.csv: Efficiency of batched proofs
- parallelization_benchmark.csv: Effect of parallelization

Visualization:

To visualize benchmark results, run the included Python script:

python3 plot.py

This will generate publication-quality PDF plots:
- matrix_performance.pdf: Core performance metrics
- batch_efficiency.pdf: Batch processing efficiency
- parallel_scaling.pdf: Parallelization scaling
- zk_comparison.pdf: Comparison with non-ZK approach



ACKNOWLEDGMENTS

This research builds upon:
- KZG polynomial commitments (https://www.iacr.org/archive/asiacrypt2010/6477178/6477178.pdf)
- Arkworks libraries (https://github.com/arkworks-rs)
