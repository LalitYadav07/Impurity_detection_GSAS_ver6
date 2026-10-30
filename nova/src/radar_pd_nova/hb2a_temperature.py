"""Reviewed paired He-3 sensor selection; no title/setpoint substitution.

This is an explicit HB-2A acquisition-family choice, not a universal cryostat
thermometer. Low-range operation is accepted only while the high-range sensor
is on its observed 1.2 K floor. In the overlap/warm range the two readings must
agree. A scan crossing ranges or showing contradictory readings needs review.
"""
import math

HE3_AUTOMATIC_CHANNEL = 'He-3 paired thermometers (automatic range)'
HE3_CHANNELS = ('he3potl_temp', 'he3poth_temp')
HE3_POLICY = 'HB2A-paired-He3-thermometers/1'


def resolve_he3_temperature(channels, times):
    evidence={'policy':HE3_POLICY,'selected_channel':None,'status':'unresolved','sensors':{}}
    for name in HE3_CHANNELS:
        values=channels.get(name)
        if not values or len(values)!=len(times) or any(not math.isfinite(v) or v<=0 for v in values):
            evidence['reason']='Paired He-3 selection requires both complete positive measured thermometer channels.'
            return None,evidence
        evidence['sensors'][name]={'minimum':min(values),'maximum':max(values),
            'value':sum(t*v for t,v in zip(times,values))/sum(times),'unit':'K'}
    low,high=(channels[name] for name in HE3_CHANNELS)
    if max(low)<=1.2 and min(high)>=1.15 and max(high)<=1.25 and max(high)-min(high)<=.002:
        selected=HE3_CHANNELS[0]
        reason='Low-range sensor below 1.2 K; high-range sensor remains on its approximately 1.2 K floor.'
    elif min(low)>1.2 and min(high)>1.25 and all(abs(l-h)<=max(.2,.05*h) for l,h in zip(low,high)):
        selected=HE3_CHANNELS[1]
        reason='High-range sensor above 1.25 K and paired readings agree within max(0.2 K, 5%).'
    else:
        evidence['reason']='Paired He-3 readings cross the range boundary or disagree; select a reviewed channel explicitly. No nominal title or setpoint was substituted.'
        return None,evidence
    evidence.update(status='resolved',selected_channel=selected,reason=reason)
    return selected,evidence
