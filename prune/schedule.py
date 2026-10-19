"""Sparsity schedules."""


def cubic_sparsity(step, t0, t_end, s_final, s_init=0.0):
    """Zhu & Gupta (2017):  s_t = s_f + (s_i - s_f) * (1 - (t - t0)/(t_end - t0))^3.

    Prunes fastest early (when many weights are redundant and the network can recover) and
    slows down as the sparsity approaches the target (when each removal hurts more)."""
    if step < t0:
        return s_init
    if step >= t_end:
        return s_final
    frac = (step - t0) / (t_end - t0)
    return s_final + (s_init - s_final) * (1.0 - frac) ** 3


def oneshot_sparsity(step, t0, t_end, s_final, s_init=0.0):
    """Jump to the final sparsity at t0 (the 'weaker answer', kept as an ablation)."""
    return s_init if step < t0 else s_final
