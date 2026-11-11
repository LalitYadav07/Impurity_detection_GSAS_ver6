"""Provider-neutral recipe response shape; native validation remains authoritative."""
import copy


def response_schema(model_signature=None):
    from .contracts import BACKGROUNDS
    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    def array(items): return {"type": "array", "items": items}
    def enum(*values): return {"type": "string", "enum": list(values)}
    def nullable(value): return {"anyOf": [value, {"type": "null"}]}
    string={"type":"string"};number={"type":"number"};integer={"type":"integer"}
    target={"anyOf":[
        obj({"kind":enum("instrument_profile"),"histogram":string,"parameter":enum("X","Y","Z")}),
        obj({"kind":enum("background","histogram_scale"),"histogram":string}),
        obj({"kind":enum("phase_scale","preferred_orientation","sample_size","sample_microstrain"),"phase":string,"histogram":string}),
        obj({"kind":enum("cell"),"phase":string}),
        obj({"kind":enum("atom"),"phase":string,"atom":string,"flags":enum("F","X","U","FX","FU","XU","FXU")})]}
    background=obj({"histogram":string,"function":enum(*BACKGROUNDS),"coefficients":integer,"fixed_points":enum("preserve","remove")})
    constraint={"anyOf":[obj({"kind":enum("occupancy_sum"),"terms":array(obj({"phase":string,"atom":string,"coefficient":number})),"total":number,"reason":string}),obj({"kind":enum("uiso_equal"),"atoms":array(obj({"phase":string,"atom":string})),"reason":string})]}
    stage=obj({"id":{"type":"string","description":"Unique safe stage identifier; case-insensitive. Do not use reserved names baseline or final."},"title":string,"cycles":integer,"free":array(target),"constraints":array(constraint),
               "max_function_evaluations":nullable({"type":"integer","description":"Native numerical/Jacobian residual-call limit, 1–100000. Required for numeric stages; use null for Hessian stages. The cycles field is not enforced by MINPACK."}),
               "relative_objective_tolerance":nullable({"type":"number","description":"Explicitly requested native relative-objective stopping tolerance (GSAS-II min dM/M), 1e-12 to 0.01. Use null to preserve the source control. It is not the separate shift/sigma acceptance criterion."}),
               "depends_on":{"type":"array","items":string,"description":"Earlier stage IDs that must have succeeded. Use [] for an independent stage. Sequential order alone is not a dependency."},
               "convergence":nullable(obj({"max_shift_over_sigma":number})),
               "background":nullable(background),"phase_normalization":array(string),
               "fit_ranges":array(obj({"histogram":string,"unit":enum("native","tof","d","Q","two_theta"),"lower":number,"upper":number})),
               "exclusions":array(obj({"histogram":string,"unit":enum("native","tof","d","Q","two_theta"),"mode":enum("add","replace","clear"),"regions":array(array(number))})),
               "bounds":array(obj({"phase":string,"atom":string,"parameter":enum("F","U"),"lower":number,"upper":number,"value":nullable(number)})),
               "texture":array(obj({"phase":string,"histogram":string,"model":enum("march-dollase","spherical-harmonics"),"direction":nullable(array(integer)),"order":nullable(integer),"value":nullable(number)}))})
    signature=enum(model_signature) if model_signature is not None else string
    failure_policy = enum("stop", "continue_independent")
    failure_policy["description"] = (
        "After a scientific stage rejection, stop or restore the complete previous checkpoint and "
        "continue only stages whose explicit dependencies succeeded. Use continue_independent only "
        "when requested or retained from the supplied previous plan; otherwise stop. Technical errors, "
        "invalid requests, and failed provenance/active-set checks always stop."
    )
    return obj({"schema":enum("radar-gpx-recipe/v1"),"model_signature":signature,"request":string,
                "failure_policy": failure_policy,
                "assumptions":array(string),"questions":array(string),"unsupported":array(string),"stages":array(stage)})


def normalize_optional_nulls(plan):
    """Strict structured output uses null for optional values; preserve all numbers."""
    plan=copy.deepcopy(plan)
    if not isinstance(plan,dict) or not isinstance(plan.get("stages"),list):return plan
    for stage in plan["stages"]:
        if not isinstance(stage,dict):continue
        if stage.get('max_function_evaluations', 1) is None:
            del stage['max_function_evaluations']
        if stage.get('relative_objective_tolerance', 1) is None:
            del stage['relative_objective_tolerance']
        for field,keys in (("bounds",("value",)),("texture",("direction","order","value"))):
            if not isinstance(stage.get(field),list):continue
            for item in stage[field]:
                if not isinstance(item,dict):continue
                for key in keys:
                    if key in item and item[key] is None:del item[key]
    return plan
