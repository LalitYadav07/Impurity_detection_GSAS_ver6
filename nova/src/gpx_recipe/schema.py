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
        obj({"kind":enum("background","histogram_scale"),"histogram":string}),
        obj({"kind":enum("phase_scale","preferred_orientation"),"phase":string,"histogram":string}),
        obj({"kind":enum("cell"),"phase":string}),
        obj({"kind":enum("atom"),"phase":string,"atom":string,"flags":enum("F","X","U","FX","FU","XU","FXU")})]}
    background=obj({"histogram":string,"function":enum(*BACKGROUNDS),"coefficients":integer,"fixed_points":enum("preserve","remove")})
    constraint={"anyOf":[obj({"kind":enum("occupancy_sum"),"terms":array(obj({"phase":string,"atom":string,"coefficient":number})),"total":number,"reason":string}),obj({"kind":enum("uiso_equal"),"atoms":array(obj({"phase":string,"atom":string})),"reason":string})]}
    stage=obj({"id":string,"title":string,"cycles":integer,"free":array(target),"constraints":array(constraint),
               "background":nullable(background),"phase_normalization":array(string),
               "fit_ranges":array(obj({"histogram":string,"unit":enum("native","tof","d","Q","two_theta"),"lower":number,"upper":number})),
               "exclusions":array(obj({"histogram":string,"unit":enum("native","tof","d","Q","two_theta"),"mode":enum("add","replace","clear"),"regions":array(array(number))})),
               "bounds":array(obj({"phase":string,"atom":string,"parameter":enum("F","U"),"lower":number,"upper":number,"value":nullable(number)})),
               "texture":array(obj({"phase":string,"histogram":string,"model":enum("march-dollase","spherical-harmonics"),"direction":nullable(array(integer)),"order":nullable(integer),"value":nullable(number)}))})
    signature=enum(model_signature) if model_signature is not None else string
    return obj({"schema":enum("radar-gpx-recipe/v1"),"model_signature":signature,"request":string,
                "assumptions":array(string),"questions":array(string),"unsupported":array(string),"stages":array(stage)})


def normalize_optional_nulls(plan):
    """Strict structured output uses null for optional values; preserve all numbers."""
    plan=copy.deepcopy(plan)
    if not isinstance(plan,dict) or not isinstance(plan.get("stages"),list):return plan
    for stage in plan["stages"]:
        if not isinstance(stage,dict):continue
        for field,keys in (("bounds",("value",)),("texture",("direction","order","value"))):
            if not isinstance(stage.get(field),list):continue
            for item in stage[field]:
                if not isinstance(item,dict):continue
                for key in keys:
                    if key in item and item[key] is None:del item[key]
    return plan
