"""Descriptive peak-width diagnostics; not uncertainty estimates or fit gates."""
import numpy as np


def half_height_width(x,y,center,window=.065):
    x,y=np.asarray(x),np.asarray(y);mask=(x>center-window)&(x<center+window)&np.isfinite(x)&np.isfinite(y)
    x,y=x[mask],y[mask]
    if len(x)<12:return None
    bg=float(np.median(np.r_[y[:5],y[-5:]]));j=int(np.argmax(y));level=(y[j]+bg)/2
    if y[j]<=bg:return None
    left=right=j
    while left>0 and y[left]>level:left-=1
    while right<len(y)-1 and y[right]>level:right+=1
    if left==0 or right==len(y)-1:return None
    lo=np.interp(level,[y[left],y[left+1]],[x[left],x[left+1]])
    hi=np.interp(level,[y[right],y[right-1]],[x[right],x[right-1]])
    return dict(fwhm_A=float(hi-lo),center_A=float(x[j]),background=bg,height=float(y[j]))


def compare_widths(view,difc,centers):
    return [dict(d_A=c,final_observed=half_height_width(view['x']/difc,view['y'],c),radar_calculated=half_height_width(view['x']/difc,view['calc'],c),webmon_observed=half_height_width(view['live_x']/difc,view['live_scaled'],c)) for c in centers]
