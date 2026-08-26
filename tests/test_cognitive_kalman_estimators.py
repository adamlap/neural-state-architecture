from experiments.cognitive._kalman import ConstantVelocityKalman, ScalarKalman


def test_scalar_kalman_bootstraps_from_first_observation():
    # Regression guard: the filter used to start at x=0 with an outlier gate
    # tight enough to permanently reject the first real observation whenever the
    # true value was far from zero (e.g. ~97 on a 10-99 scale).
    f = ScalarKalman(measurement_noise=1.0, outlier_sigma=5.0)
    assert f.update(97.0)
    assert abs(f.x - 97.0) < 1e-9


def test_scalar_kalman_converges_to_a_constant_value():
    f = ScalarKalman(measurement_noise=0.5, process_noise=0.01, outlier_sigma=5.0)
    true_value = 42.0
    import random
    rng = random.Random(0)
    for _ in range(200):
        f.step(true_value + rng.uniform(-0.5, 0.5))
    assert abs(f.x - true_value) < 0.5


def test_scalar_kalman_rejects_gross_outliers_once_converged():
    f = ScalarKalman(measurement_noise=0.5, process_noise=0.01, outlier_sigma=5.0)
    for _ in range(50):
        f.step(10.0)
    accepted = f.update(500.0)
    assert not accepted
    assert abs(f.x - 10.0) < 1.0


def test_constant_velocity_kalman_bootstraps_from_first_observation():
    f = ConstantVelocityKalman(measurement_noise=1.0, outlier_sigma=5.0)
    assert f.update(97.0)
    assert abs(f.position - 97.0) < 1e-9


def test_constant_velocity_kalman_tracks_a_drifting_signal():
    f = ConstantVelocityKalman(measurement_noise=0.5, outlier_sigma=5.0)
    import random
    rng = random.Random(0)
    true_value = 0.0
    drift = 0.3
    for _ in range(100):
        true_value += drift
        f.step(true_value + rng.uniform(-0.5, 0.5))
    assert abs(f.position - true_value) < 2.0
    assert abs(f.velocity - drift) < 0.2
