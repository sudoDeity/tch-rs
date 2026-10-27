//! Minimal CPU ABI, linear algebra and autograd tests for PyTorch 2.15.
use tch::{Device, Kind, Tensor};

#[test]
fn autograd_finite_on_cpu() {
    let point = Tensor::from_slice(&[1.5_f32, -0.5]).set_requires_grad(true);
    let residual = Tensor::stack(
        &[
            point.select(0, 0).square() - 2.0,
            point.select(0, 1) * 2.0,
        ],
        0,
    );
    let loss = residual.square().sum(Kind::Float) * 0.5;
    let grad = Tensor::run_backward(&[loss], &[&point], false, false).remove(0);
    assert!(grad.isfinite().all().int64_value(&[]) != 0);
    assert!((grad.double_value(&[0]) - 0.75).abs() < 1e-5);
    assert!((grad.double_value(&[1]) + 2.0).abs() < 1e-5);
}

#[test]
fn cholesky_and_qr_legacy_wrappers_call_linalg() {
    let input = Tensor::from_slice(&[4.0_f32, 1.0, 1.0, 3.0])
        .view([2, 2])
        .to_device(Device::Cpu);
    let lower = input.cholesky(false);
    let chol_out = Tensor::zeros([2, 2], (Kind::Float, Device::Cpu));
    let lower_out = input.cholesky_out(&chol_out, false);
    for l in [&lower, &lower_out] {
        assert!(
            (l.matmul(&l.transpose(0, 1)) - &input)
                .abs()
                .max()
                .double_value(&[])
                < 1e-5
        );
    }
    let (q, r) = input.qr(true);
    assert!((q.matmul(&r) - &input).abs().max().double_value(&[]) < 1e-5);
    let q_out = Tensor::zeros([2, 2], (Kind::Float, Device::Cpu));
    let r_out = Tensor::zeros([2, 2], (Kind::Float, Device::Cpu));
    let (q, r) = input.qr_q(&q_out, &r_out, true);
    assert!((q.matmul(&r) - &input).abs().max().double_value(&[]) < 1e-5);
}
