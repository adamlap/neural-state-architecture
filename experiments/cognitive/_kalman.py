"""Small linear-Gaussian Kalman filters shared by the deterministic cognitive
benchmarks.

Earlier revisions of these benchmarks blended observations into an estimate with
hand-tuned fixed coefficients (e.g. ``0.7 * estimate + 0.3 * observation``). That
approach has two problems that this module fixes:

1. A fixed blend cannot express "how much should I trust this observation right
   now" — that trust should grow while unobserved (uncertainty increases) and
   shrink once well-corrected. A real Kalman gain, derived from a variance that
   is tracked explicitly, does this automatically instead of requiring a new
   magic constant per task.
2. A fixed *absolute* outlier-rejection threshold (e.g. "reject any residual over
   10.0") is brittle: it can wrongly reject a legitimate large correction after a
   long observation gap (variance is high, so a big residual is expected) while
   being too loose or too tight in other regimes. Gating on the residual's
   Mahalanobis distance (``residual^2 > sigma^2 * predicted_variance``) adapts
   with the filter's own uncertainty.

These are intentionally minimal (no numpy dependency) scalar/2-state filters —
enough to give the "predictive" conditions in the benchmarks a principled
estimator instead of an arbitrarily tuned one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScalarKalman:
    """Tracks a single scalar ``x`` evolving as ``x' = a*x + b*u + noise``.

    With the defaults (``a=1, b=0``) this is a *persistent* (non-predictive)
    belief: it does not move between observations. Supplying a known ``a``/``b``
    (e.g. a decay/gain from a known linear system) turns it into a predictive
    single-state filter.
    """

    a: float = 1.0
    b: float = 0.0
    process_noise: float = 0.05
    measurement_noise: float = 1.0
    outlier_sigma: float = 5.0
    x: float = 0.0
    p: float = 100.0  # initial uncertainty: unknown starting value

    def __post_init__(self) -> None:
        self._r = self.measurement_noise ** 2
        self._outlier_s2 = self.outlier_sigma ** 2
        self._initialized = False

    def predict(self, control: float = 0.0) -> None:
        self.x = self.a * self.x + self.b * control
        self.p = (self.a ** 2) * self.p + self.process_noise

    def update(self, observation: float) -> bool:
        """Fold in an observation. Returns False if rejected as a statistical outlier."""
        if not self._initialized:
            # Bootstrap directly from the first observation instead of gating it
            # against an arbitrary (e.g. zero) initial guess.
            self.x = observation
            self.p = self._r
            self._initialized = True
            return True
        residual = observation - self.x
        s = self.p + self._r
        if residual * residual > self._outlier_s2 * s:
            return False
        k = self.p / s
        self.x += k * residual
        self.p *= (1.0 - k)
        return True

    def step(self, observation: Optional[float], control: float = 0.0) -> float:
        self.predict(control)
        if observation is not None:
            self.update(observation)
        return self.x


@dataclass
class ConstantVelocityKalman:
    """Tracks a 2-state ``(position, velocity)`` system.

    Transition is ``position' = f11*position + f12*velocity + b1*u`` and
    ``velocity' = f21*position + f22*velocity + b2*u``. Defaults implement a
    constant-velocity model (``position' = position + velocity``, ``velocity'``
    unchanged), which is the right model for a latent quantity drifting at an
    unknown, roughly constant rate.
    """

    f11: float = 1.0
    f12: float = 1.0
    f21: float = 0.0
    f22: float = 1.0
    b1: float = 0.0
    b2: float = 0.0
    process_noise: float = 0.02
    measurement_noise: float = 1.0
    outlier_sigma: float = 5.0
    position: float = 0.0
    velocity: float = 0.0

    def __post_init__(self) -> None:
        self._r = self.measurement_noise ** 2
        self._outlier_s2 = self.outlier_sigma ** 2
        # Covariance of [position, velocity]; start uncertain (unknown initial state).
        self.p11, self.p12, self.p21, self.p22 = 100.0, 0.0, 0.0, 10.0
        self._initialized = False

    def predict(self, control: float = 0.0) -> None:
        x, v = self.position, self.velocity
        self.position = self.f11 * x + self.f12 * v + self.b1 * control
        self.velocity = self.f21 * x + self.f22 * v + self.b2 * control

        f11, f12, f21, f22 = self.f11, self.f12, self.f21, self.f22
        p11, p12, p21, p22 = self.p11, self.p12, self.p21, self.p22
        # P = F P F^T + Q (diagonal process noise on both components).
        a11 = f11 * p11 + f12 * p21
        a12 = f11 * p12 + f12 * p22
        a21 = f21 * p11 + f22 * p21
        a22 = f21 * p12 + f22 * p22
        n11 = a11 * f11 + a12 * f12
        n12 = a11 * f21 + a12 * f22
        n21 = a21 * f11 + a22 * f12
        n22 = a21 * f21 + a22 * f22
        q = self.process_noise
        self.p11, self.p12, self.p21, self.p22 = n11 + q, n12, n21, n22 + q

    def update(self, observation: float) -> bool:
        """Observation model H = [1, 0] (position is observed directly)."""
        return self.update_channel(observation, channel=0)

    def update_channel(self, observation: float, channel: int = 0) -> bool:
        """Fold in an observation of either state component (0=position, 1=velocity).

        Used when the two components of the state are not observed together, e.g.
        a system where only one dimension is sampled per step.
        """
        if not self._initialized:
            # Bootstrap directly from the first observation instead of gating it
            # against an arbitrary (e.g. zero) initial guess. The unobserved
            # component remains unknown (high uncertainty) until corrected.
            if channel == 0:
                self.position = observation
                self.p11, self.p12, self.p21 = self._r, 0.0, 0.0
            else:
                self.velocity = observation
                self.p22, self.p12, self.p21 = self._r, 0.0, 0.0
            self._initialized = True
            return True
        if channel == 0:
            estimate, p_ii, p_i_other = self.position, self.p11, self.p21
        else:
            estimate, p_ii, p_i_other = self.velocity, self.p22, self.p12
        residual = observation - estimate
        s = p_ii + self._r
        if residual * residual > self._outlier_s2 * s:
            return False
        k_self = p_ii / s
        k_other = p_i_other / s
        if channel == 0:
            self.position += k_self * residual
            self.velocity += k_other * residual
        else:
            self.velocity += k_self * residual
            self.position += k_other * residual
        p11, p12, p21, p22 = self.p11, self.p12, self.p21, self.p22
        if channel == 0:
            self.p11, self.p12 = p11 - k_self * p11, p12 - k_self * p12
            self.p21, self.p22 = p21 - k_other * p11, p22 - k_other * p12
        else:
            self.p22, self.p21 = p22 - k_self * p22, p21 - k_self * p21
            self.p12, self.p11 = p12 - k_other * p22, p11 - k_other * p21
        return True

    def step(self, observation: Optional[float], control: float = 0.0) -> float:
        self.predict(control)
        if observation is not None:
            self.update(observation)
        return self.position
