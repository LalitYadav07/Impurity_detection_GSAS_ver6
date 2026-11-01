"""Symmetry-constrained cell covariance for both native worker packages."""

CELL_KEYS = ("a", "b", "c", "alpha", "beta", "gamma", "volume")


def cell_fit_information(phase, covariance):
    """Propagate the final reciprocal-cell covariance through GSAS-II symmetry.

    Constraint-generated variables outside the direct cell vary-list are not
    approximated as independent observations. They remain unverified here.
    """
    import numpy as np
    from GSASII import GSASIIlattice as lattice, GSASIIstrIO as strio

    fresh = bool(covariance.get("freshCOV", False))
    prefix = f"{phase.id}::"
    names = [str(name) for name in covariance.get("varyList", [])]
    direct = [name for name in names if name in {prefix + f"A{i}" for i in range(6)}]
    states = {key: "unknown" for key in CELL_KEYS}
    output = {"metric_states": states, "covariance_fresh": fresh,
              "cell_covariance": None, "covariance_status": "unavailable"}
    if len(set(names)) != len(names) or len(covariance.get('variables', [])) != len(names):
        output['covariance_status'] = 'invalid_covariance'
        return output
    if not direct:
        # A flagged but absent cell may be constraint-derived or from an older
        # refinement. Neither case can be certified from stored flags alone.
        state = "unknown" if phase.data["General"]["Cell"][0] else "fixed"
        output["metric_states"] = {key: state for key in CELL_KEYS}
        return output
    sg = phase.data["General"]["SGData"]
    independent = {
        "-1": (0, 1, 2, 3, 4, 5), "mmm": (0, 1, 2),
        "4/m": (0, 2), "4/mmm": (0, 2),
        "3": (0, 2), "3m1": (0, 2), "31m": (0, 2),
        "6/m": (0, 2), "6/mmm": (0, 2),
        "3R": (0, 3), "3mR": (0, 3), "m3": (0,), "m3m": (0,),
    }.get(sg["SGLaue"])
    if sg["SGLaue"] == "2/m":
        independent = (0, 1, 2, {"a": 5, "b": 4, "c": 3}[sg["SGUniq"]])
    expected = {prefix + f"A{i}" for i in independent or ()}
    if set(direct) != expected:
        # Missing symmetry-independent variables can have external constraints
        # or frozen parameters. Their dependence is not encoded by cellFill.
        output["covariance_status"] = "cell_constraints_unqualified"
        return output
    base = lattice.cell2A(phase.data["General"]["Cell"][1:7])
    parameters = {prefix + f"A{i}": float(value) for i, value in enumerate(base)}
    parameters.update(
        {
            name: float(value)
            for name, value in zip(names, covariance.get("variables", []))
            if name in parameters
        }
    )

    def evaluate(values):
        reciprocal, _ = strio.cellFill(prefix, sg, values, {})
        return np.asarray([*lattice.A2cell(reciprocal), lattice.calc_V(reciprocal)], dtype=float)

    jacobian = np.zeros((7, len(direct)))
    for j, name in enumerate(direct):
        step = max(abs(parameters[name]) * 1e-6, 1e-10)
        plus, minus = dict(parameters), dict(parameters)
        plus[name] += step
        minus[name] -= step
        jacobian[:, j] = (evaluate(plus) - evaluate(minus)) / (2 * step)
    # Classify symmetry-fixed angles from the model, not a finite-difference
    # tolerance. Roundoff in A2cell can otherwise give a fixed hexagonal gamma
    # a tiny nonzero derivative and publish it as refined with a spurious ESD.
    # Conversely, a numerically small derivative does not make a free angle fixed.
    fixed_angles = set()
    if sg["SGLaue"] == "2/m":
        free_angle = {"a": "alpha", "b": "beta", "c": "gamma"}[sg["SGUniq"]]
        fixed_angles = {"alpha", "beta", "gamma"} - {free_angle}
    elif sg["SGLaue"] not in {"-1", "3R", "3mR"}:
        fixed_angles = {"alpha", "beta", "gamma"}
    for i, key in enumerate(CELL_KEYS):
        if key in fixed_angles:
            jacobian[i] = 0.0
        states[key] = "fixed" if key in fixed_angles else "refined" if fresh else "stale"
    output["cell_jacobian"] = jacobian.tolist()
    output['cell_variable_names'] = direct
    matrix = covariance.get("covMatrix")
    if not fresh or matrix is None:
        return output
    array = np.asarray(matrix, dtype=float)
    if array.shape != (len(names), len(names)) or not np.isfinite(array).all():
        return output
    indices = [names.index(name) for name in direct]
    selected = array[np.ix_(indices, indices)]
    scale = max(float(np.max(np.abs(selected))), np.finfo(float).tiny)
    if not np.allclose(selected, selected.T, rtol=1e-9, atol=scale*1e-12):
        output["covariance_status"] = "invalid_covariance"
        return output
    selected = (selected + selected.T) / 2
    if np.linalg.eigvalsh(selected).min() < -scale * 1e-9:
        output["covariance_status"] = "invalid_covariance"
        return output
    propagated = jacobian @ selected @ jacobian.T
    propagated = (propagated + propagated.T) / 2
    if np.isfinite(propagated).all():
        output["cell_covariance"] = propagated.tolist()
    return output
