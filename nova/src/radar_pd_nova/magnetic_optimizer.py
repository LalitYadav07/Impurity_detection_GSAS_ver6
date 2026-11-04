"""Presentation of the explicitly reviewed magnetic optimization budget."""

from gpx_recipe.symmetry.contracts import choices, number


def selected_optimizer_settings(kind, optimizer, iterations):
    """Do not add new policy keys to legacy or structural protocols."""
    if kind != 'magnetic':
        return {}
    selected = choices(optimizer, ['gsas_hessian', 'profiled_quadratic'])
    if selected == 'gsas_hessian':
        return {}
    if isinstance(iterations, str) and iterations.strip().isdecimal():
        iterations = int(iterations.strip())
    return dict(magnetic_optimizer=selected,
                optimizer_iterations=number(iterations, 1, 200, integer=True))


def optimizer_budget_label(policy):
    if policy.get('magnetic_optimizer') == 'profiled_quadratic':
        grouped=('One checked surrogate per model; all starts are screened together, then distinct calculated profiles are verified natively. ' if policy.get('grouped_surrogate') else '')
        return (grouped+f"Up to {policy['optimizer_iterations']} profiled optimizer iterations per start. "
                "GSAS-II evaluates the diffraction model; the profiled solver has its own convergence checks.")
    label = f"Up to {policy['cycles']} refinement cycles per start."
    if policy['cycles'] == 1:
        label += ' Preliminary one-cycle screening.'
    return label
