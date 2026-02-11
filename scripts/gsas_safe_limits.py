#!/usr/bin/env python3
"""
GSAS-II Safe Limits & Instrument Guard

This module calculates safe operating limits for diffraction data to prevent
software instability. It ensures:
- D-spacing minimums (`d_min`) strictly respect the instrument's valid range.
- Protection against negative variance regions in Time-of-Flight data.
- Validation of instrument parameter files (.instprm).
"""

import numpy as np
import traceback

def calculate_safe_d_min(inst_params: dict) -> float:
    """
    Calculate the minimum safe d-spacing where variance sigma^2 > 0.
    Standard GSAS-II TOF Variance: sigma^2 = sig2*d^4 + sig1*d^2 + sigq*d + sig0
    """
    try:
        def _get_val(key, default=0.0):
            val = inst_params.get(key, [default])
            return float(val[0]) if isinstance(val, (list, tuple)) else float(val)

        s0 = _get_val('sig-0', 0.0)
        s1 = _get_val('sig-1', 0.0)
        s2 = _get_val('sig-2', 0.0)
        sq = _get_val('sig-q', 0.0)
        
        # Coefficients for np.roots (highest power first): [d^4, d^3, d^2, d^1, d^0]
        # P(d) = s2*d^4 + 0*d^3 + s1*d^2 + sq*d + s0
        coeffs = [s2, 0.0, s1, sq, s0]
        roots = np.roots(coeffs)
        
        # Filter for real, positive roots
        real_pos_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
        
        if not real_pos_roots:
            # Check sign at small d (e.g., 0.1)
            val = s2*(0.1**4) + s1*(0.1**2) + sq*0.1 + s0
            if val > 0:
                return 0.1
            return 0.1
            
        # The safe limit is the largest root + small buffer
        limit = max(real_pos_roots)
        # User reported that 5256.75 (approx 0.23A) was too low, but 5696.9 worked.
        # This suggests a need for a larger safety buffer.
        # Adding 0.05A buffer roughly corresponds to >10% increase at these low d-values.
        return limit + 0.05

    except Exception as e:
        print(f"[SafeLimits] Error calculating safe d_min: {e}")
        return 0.1

def apply_safe_limits(proj) -> bool:
    """
    Check instrument parameters and apply data limits to avoid negative variance regions.
    Returns True if limits were updated.
    """
    updated = False
    try:
        for hist in proj.histograms():
            # Get instrument params
            if 'Instrument Parameters' not in hist.data:
                continue
            
            inst = hist.data['Instrument Parameters'][0]
            
            # Calculate safe d-min
            safe_d = calculate_safe_d_min(inst)
            
            # Convert to TOF/2Theta if needed
            inst_type = inst.get('Type', [''])[0]
            
            safe_native = 0.0
            is_tof = 'T' in str(inst_type)
            
            if is_tof:
                # TOF = C*d + A*d^2 + B/d + Zero
                difC = float(inst.get('difC', [0])[0])
                difA = float(inst.get('difA', [0])[0])
                difB = float(inst.get('difB', [0])[0])
                Zero = float(inst.get('Zero', [0])[0])
                
                safe_native = difC * safe_d + difA * safe_d**2 + difB/safe_d + Zero
            else:
                # CW: 2Theta
                lam = float(inst.get('Lam', [1.54])[0])
                if safe_d > 0:
                    val = lam / (2*safe_d)
                    if val <= 1.0:
                        theta = np.arcsin(val)
                        safe_native = np.degrees(2*theta)
                        
            # Apply to Limits
            limits = hist.data.get('Limits', [(0, 10000), [0, 10000]]) 
            current_min = limits[1][0] 
            
            if safe_native > current_min:
                print(f"[SafeLimits] Histogram {hist.name}: Raising min limit {current_min:.2f} -> {safe_native:.2f} (d_min={safe_d:.2f}A)")
                limits[1][0] = safe_native
                hist.data['Limits'] = limits
                updated = True
                
    except Exception as e:
        print(f"[SafeLimits] Error checking limits: {e}")
        traceback.print_exc()
        
    return updated
