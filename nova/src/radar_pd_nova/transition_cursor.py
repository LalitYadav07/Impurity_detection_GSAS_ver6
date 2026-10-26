"""Bounded native-data frames for a browser-local diffraction cursor preview."""

from .transition_tracking_plot import ribbon_trace, selection_presentation

MAX_PREVIEW_POINTS = 500_000


def selection_figure(figure, frame, region=None):
    """Replace selection details while retaining the already built map cells."""
    layout={**figure["layout"],"annotations":frame["annotations"],"shapes":frame["shapes"],
            "yaxis2":{**figure["layout"]["yaxis2"],**frame["axis"]}}
    for key in ("xaxis","xaxis2"):
        axis=dict(layout[key])
        if region is None:
            axis.pop("range",None)
        else:
            axis["range"]=list(region)
        layout[key]=axis
    return {"data":[figure["data"][0],frame["trace"]],"layout":layout}


def cursor_frames(patterns, backgrounds, run_ids, labels, *, selected, limits,
                  scale, contrast, base_shapes=(), overlay="off", regions=None,
                  point_budget=MAX_PREVIEW_POINTS):
    """Cache nearest native ribbons without resampling or rebuilding heatmaps.

    Missing frames deliberately use the server fallback. Preview selection
    never updates inference, the selected full fit, or the comparison policy.
    """
    frames = [None] * len(patterns)
    center = run_ids.index(selected) if selected in run_ids else len(patterns)-1
    backgrounds = backgrounds if backgrounds is not None else [None]*len(patterns)
    remaining = point_budget
    for i in sorted(range(len(patterns)), key=lambda j: abs(j-center)):
        pattern = patterns[i]
        if len(pattern.x) > remaining:
            continue
        remaining -= len(pattern.x)
        evidence = regions(run_ids[i]) if regions else {}
        presentation = selection_presentation(
            run_ids, labels, run_ids[i], ribbon=True,
            comparison_run=evidence.get("comparison_run"),
            change_regions=evidence.get("regions", ()), overlay=overlay)
        frames[i] = {
            "run_id": run_ids[i], "label": labels[i],
            "trace": {**ribbon_trace(pattern, backgrounds[i], limits, scale, contrast, labels[i]),
                      "xaxis": "x", "yaxis": "y"},
            "annotations": presentation["annotations"],
            "shapes": [*base_shapes, *presentation["shapes"]],
            "axis": presentation["axis"],
            "notice": evidence.get("notice", ""),
            "evidence": evidence.get("evidence", []),
        }
    return frames
