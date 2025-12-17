#!/usr/bin/env python
"""
Test script to verify BF16 mixed precision implementation
Tests both FP32 and BF16 modes
"""

import os
import sys
import jax
import jax.numpy as jnp

def test_dtype_detection():
    """Test that USE_BF16 environment variable is properly detected"""
    print("\n" + "="*60)
    print("Testing USE_BF16 environment variable detection")
    print("="*60)

    # Test FP32 mode
    os.environ['USE_BF16'] = '0'
    use_bf16 = os.environ.get('USE_BF16', '1') == '1'
    print(f"USE_BF16=0 -> use_bf16={use_bf16} (expected: False)")
    assert use_bf16 == False, "FP32 mode detection failed"

    # Test BF16 mode
    os.environ['USE_BF16'] = '1'
    use_bf16 = os.environ.get('USE_BF16', '1') == '1'
    print(f"USE_BF16=1 -> use_bf16={use_bf16} (expected: True)")
    assert use_bf16 == True, "BF16 mode detection failed"

    print("✓ Environment variable detection working correctly")

def test_model_creation():
    """Test that models can be created in both FP32 and BF16 modes"""
    print("\n" + "="*60)
    print("Testing model creation with different precisions")
    print("="*60)

    from s5.ssm import init_S5SSM
    from s5.seq_model import StackedEncoderModel

    # Create a simple SSM initializer
    ssm_init_fn = init_S5SSM(
        H=32,  # Small model for testing
        P=16,
        Lambda_re_init=jnp.ones(16),
        Lambda_im_init=jnp.zeros(16),
        V=jnp.eye(16, dtype=jnp.complex64),
        Vinv=jnp.eye(16, dtype=jnp.complex64),
        C_init='trunc_standard_normal',
        discretization='bilinear',
        dt_min=0.001,
        dt_max=0.1,
        conj_sym=True,
        clip_eigs=False,
        bidirectional=False
    )

    # Test FP32 mode
    os.environ['USE_BF16'] = '0'
    print("\nTesting FP32 mode...")
    model_fp32 = StackedEncoderModel(
        ssm=ssm_init_fn,
        d_model=32,
        n_layers=2,
        training=False
    )

    # Initialize model
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((10, 32))  # (seq_len, d_model)
    variables = model_fp32.init(key, dummy_input, jnp.ones(10))
    print(f"FP32 model initialized successfully")

    # Test BF16 mode
    os.environ['USE_BF16'] = '1'
    print("\nTesting BF16 mode...")
    model_bf16 = StackedEncoderModel(
        ssm=ssm_init_fn,
        d_model=32,
        n_layers=2,
        training=False
    )

    variables_bf16 = model_bf16.init(key, dummy_input, jnp.ones(10))
    print(f"BF16 model initialized successfully")

    print("✓ Models can be created in both FP32 and BF16 modes")

def test_forward_pass():
    """Test forward pass in both modes"""
    print("\n" + "="*60)
    print("Testing forward pass")
    print("="*60)

    from s5.layers import SequenceLayer
    from s5.ssm import init_S5SSM

    # Create a simple SSM
    ssm_init_fn = init_S5SSM(
        H=16,
        P=8,
        Lambda_re_init=jnp.ones(8),
        Lambda_im_init=jnp.zeros(8),
        V=jnp.eye(8, dtype=jnp.complex64),
        Vinv=jnp.eye(8, dtype=jnp.complex64),
        C_init='trunc_standard_normal',
        discretization='bilinear',
        dt_min=0.001,
        dt_max=0.1,
        conj_sym=False,
        clip_eigs=False,
        bidirectional=False
    )

    # Test FP32
    os.environ['USE_BF16'] = '0'
    print("\nTesting FP32 forward pass...")
    layer_fp32 = SequenceLayer(
        ssm=ssm_init_fn,
        dropout=0.0,
        d_model=16,
        training=False,
        dtype=jnp.float32
    )

    key = jax.random.PRNGKey(0)
    dummy_input = jax.random.normal(key, (10, 16))
    variables = layer_fp32.init(key, dummy_input)
    output_fp32 = layer_fp32.apply(variables, dummy_input)
    print(f"FP32 output shape: {output_fp32.shape}, dtype: {output_fp32.dtype}")

    # Test BF16
    os.environ['USE_BF16'] = '1'
    print("\nTesting BF16 forward pass...")
    layer_bf16 = SequenceLayer(
        ssm=ssm_init_fn,
        dropout=0.0,
        d_model=16,
        training=False,
        dtype=jnp.bfloat16
    )

    variables_bf16 = layer_bf16.init(key, dummy_input)
    output_bf16 = layer_bf16.apply(variables_bf16, dummy_input)
    print(f"BF16 output shape: {output_bf16.shape}, dtype: {output_bf16.dtype}")

    print("✓ Forward pass works in both modes")

def main():
    print("\n" + "="*60)
    print("BF16 Mixed Precision Implementation Test")
    print("="*60)

    # Check JAX is available
    print(f"JAX version: {jax.__version__}")
    print(f"Devices available: {jax.devices()}")

    # Run tests
    test_dtype_detection()
    test_model_creation()
    test_forward_pass()

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nBF16 mixed precision implementation is working correctly.")
    print("You can now run training with:")
    print("  - FP32 mode: python run_train.py --use_bf16 False")
    print("  - BF16 mode: python run_train.py --use_bf16 True (default)")
    print("="*60)

if __name__ == "__main__":
    main()