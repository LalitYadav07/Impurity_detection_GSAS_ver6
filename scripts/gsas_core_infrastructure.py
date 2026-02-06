#!/usr/bin/env python3
"""
Core Infrastructure for GSAS-II Integrated Impurity Detection Pipeline

This module provides the foundational classes for managing GSAS-II projects
and handling coordinate system conversions between native instrument coordinates
(2θ/TOF) and Q-space.
"""

import os
import tempfile
import traceback
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path

try:
    from GSASII import GSASIIscriptable as G2sc
    from GSASII.GSASIIobj import G2Exception
    GSAS_AVAILABLE = True
except ImportError:
    print("Warning: GSAS-II not available. Some functionality will be limited.")
    GSAS_AVAILABLE = False
    G2Exception = Exception


class GSASProjectManager:
    """
    Manages GSAS-II project lifecycle, including creation, histogram addition,
    and phase management for impurity detection pipeline.
    """
    
    def __init__(self, work_dir: str, project_name: str = "impurity_detection"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name
        self.project_path = self.work_dir / f"{project_name}.gpx"
        self.project: Optional[Any] = None
        self.main_histogram: Optional[Any] = None
        self.main_phase: Optional[Any] = None
        self.instrument_type: Optional[str] = None
        
    def create_project(self, overwrite: bool = True, template_gpx: Optional[str] = None) -> bool:
        """Create a new GSAS-II project, optionally from a template."""
        if not GSAS_AVAILABLE:
            raise RuntimeError("GSAS-II not available for project creation")
            
        try:
            if overwrite and self.project_path.exists():
                import time
                for i in range(5):
                    try:
                        self.project_path.unlink()
                        break
                    except PermissionError:
                        if i == 4: raise
                        time.sleep(0.2)
            
            if template_gpx and os.path.exists(template_gpx):
                import shutil
                shutil.copy2(template_gpx, str(self.project_path))
                self.project = G2sc.G2Project(gpxfile=str(self.project_path))
                # GSAS-II sometimes needs to know it's a new path
                self.project.save(str(self.project_path))
            else:
                self.project = G2sc.G2Project(newgpx=str(self.project_path))
                
            print(f"Created GSAS-II project: {self.project_path} (template={template_gpx})")
            return True
            
        except Exception as e:
            print(f"Failed to create GSAS-II project: {e}")
            traceback.print_exc()
            return False
    
    def add_histogram(self, data_file: str, instprm_file: str, 
                     fmthint: Optional[str] = None) -> bool:
        """Add powder histogram to the project."""
        if not self.project:
            raise RuntimeError("Project not initialized. Call create_project() first.")
            
        try:
            # Smart histogram addition with format hint fallback
            if fmthint:
                try:
                    self.main_histogram = self.project.add_powder_histogram(
                        data_file, instprm_file, fmthint=fmthint
                    )
                except TypeError:
                    # Fallback for older GSAS-II versions
                    self.main_histogram = self.project.add_powder_histogram(
                        data_file, instprm_file
                    )
            else:
                self.main_histogram = self.project.add_powder_histogram(
                    data_file, instprm_file
                )
                
            # Determine instrument type
            self.instrument_type = self._get_instrument_type()
            print(f"Added histogram: {data_file} (type: {self.instrument_type})")
            return True
            
        except Exception as e:
            print(f"Failed to add histogram: {e}")
            traceback.print_exc()
            return False
    
    def add_phase_from_cif(self, cif_file: str, phasename: str = "MainPhase",
                          link_to_histogram: bool = True) -> bool:
        """Add phase from CIF file and optionally link to histogram."""
        if not self.project:
            raise RuntimeError("Project not initialized.")
            
        try:
            histograms = [self.main_histogram] if (link_to_histogram and self.main_histogram) else []
            self.main_phase = self.project.add_phase(
                cif_file, phasename=phasename, histograms=histograms
            )
            print(f"Added phase '{phasename}' from {cif_file}")
            return True
            
        except Exception as e:
            print(f"Failed to add phase: {e}")
            traceback.print_exc()
            return False
    
    def add_phase_from_cif_text(self, cif_text: str, phasename: str = "Phase",
                               link_to_histogram: bool = True) -> bool:
        """Add phase from CIF text string."""
        # Create temporary CIF file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as tmp:
            tmp.write(cif_text)
            tmp.flush()
            temp_path = tmp.name
            
        try:
            result = self.add_phase_from_cif(temp_path, phasename, link_to_histogram)
            return result
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    
    def save_project(self) -> bool:
        """Save the current project state."""
        if not self.project:
            return False
            
        try:
            self.project.save(str(self.project_path))
            return True
        except Exception as e:
            print(f"Failed to save project: {e}")
            return False
    
    def get_instrument_type(self) -> Optional[str]:
        """Get instrument type (CW or TOF)."""
        return self.instrument_type
    
    def _get_instrument_type(self) -> str:
        """Determine if instrument is TOF or constant wavelength."""
        if not self.main_histogram:
            return "Unknown"
            
        try:
            inst_params = self.main_histogram.getHistEntryValue(['Instrument Parameters'])[0]
            inst_type = str(inst_params.get('Type', [''])[0])
            return "TOF" if 'T' in inst_type else "CW"
        except Exception:
            return "Unknown"
    
    def cleanup_project(self):
        """Clean up project resources."""
        self.project = None
        self.main_histogram = None
        self.main_phase = None
        self.instrument_type = None


class CoordinateHandler:
    """
    Handles coordinate system conversions between native instrument coordinates
    (2θ for CW, TOF for time-of-flight) and Q-space for impurity detection.
    """
    
    def __init__(self, instrument_type: str, instrument_params: Dict[str, Any]):
        self.instrument_type = instrument_type.upper()
        self.instrument_params = instrument_params
        
        # Extract relevant instrument parameters
        if self.instrument_type == "TOF":
            self.difC = float(instrument_params.get('difC', [0, 0, False])[1])
            self.difA = float(instrument_params.get('difA', [0, 0, False])[1])
            self.difB = float(instrument_params.get('difB', [0, 0, False])[1])
            self.zero_tof = float(instrument_params.get('Zero', [0, 0, False])[1])
        else:  # CW
            self.wavelength = float(instrument_params.get('Lam', [1.54])[0])
            self.zero_2theta = float(instrument_params.get('Zero', [0, 0, False])[1]) if \
                isinstance(instrument_params.get('Zero', [0, 0]), (list, tuple)) else 0.0
    
    @classmethod
    def from_gsas_histogram(cls, histogram):
        """Create CoordinateHandler from GSAS histogram object."""
        if not histogram:
            raise ValueError("Histogram object is None")
            
        try:
            inst_params = histogram.getHistEntryValue(['Instrument Parameters'])[0]
            inst_type = str(inst_params.get('Type', [''])[0])
            instrument_type = "TOF" if 'T' in inst_type else "CW"
            return cls(instrument_type, inst_params)
        except Exception as e:
            raise RuntimeError(f"Failed to extract instrument parameters: {e}")
    
    def q_to_native(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q values to native instrument coordinates."""
        q_values = np.asarray(q_values, dtype=float)
        
        if self.instrument_type == "TOF":
            return self._q_to_tof(q_values)
        else:
            return self._q_to_2theta(q_values)
    
    def native_to_q(self, native_values: np.ndarray) -> np.ndarray:
        """Convert native coordinates to Q values."""
        native_values = np.asarray(native_values, dtype=float)
        
        if self.instrument_type == "TOF":
            return self._tof_to_q(native_values)
        else:
            return self._2theta_to_q(native_values)
    
    def d_to_native(self, d_values: np.ndarray) -> np.ndarray:
        """Convert d-spacing values to native coordinates."""
        d_values = np.asarray(d_values, dtype=float)
        q_values = 2.0 * np.pi / np.maximum(d_values, 1e-10)
        return self.q_to_native(q_values)
    
    def _q_to_2theta(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q to 2θ for constant wavelength."""
        # Q = 4π sin(θ) / λ  =>  sin(θ) = Qλ/(4π)  =>  θ = arcsin(Qλ/(4π))
        sin_theta = (q_values * self.wavelength) / (4.0 * np.pi)
        sin_theta = np.clip(sin_theta, 0.0, 0.999)  # Avoid domain errors
        theta_rad = np.arcsin(sin_theta)
        two_theta_deg = 2.0 * np.degrees(theta_rad) + self.zero_2theta
        return two_theta_deg
    
    def _2theta_to_q(self, two_theta_deg: np.ndarray) -> np.ndarray:
        """Convert 2θ to Q for constant wavelength."""
        two_theta_corrected = two_theta_deg - self.zero_2theta
        theta_rad = np.radians(two_theta_corrected / 2.0)
        q_values = (4.0 * np.pi / self.wavelength) * np.sin(theta_rad)
        return q_values
    
    def _q_to_tof(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q to TOF using GSAS-II TOF equation."""
        # First convert Q to d-spacing: d = 2π/Q
        d_values = 2.0 * np.pi / np.maximum(q_values, 1e-10)
        
        # TOF equation: TOF = difC*d + difA*d² + difB/d + Zero
        tof_values = (self.difC * d_values + 
                     self.difA * d_values**2 + 
                     self.difB / np.maximum(d_values, 1e-10) + 
                     self.zero_tof)
        return tof_values
    
    def _tof_to_q(self, tof_values: np.ndarray) -> np.ndarray:
        """Convert TOF to Q (requires solving quadratic equation)."""
        # TOF - Zero = difC*d + difA*d² + difB/d
        # This is a cubic in d, but we can solve iteratively or use approximation
        
        tof_corrected = tof_values - self.zero_tof
        
        # Initial guess using linear approximation (difC dominates for most d-values)
        d_guess = tof_corrected / max(self.difC, 1e-10)
        
        # Newton-Raphson iteration to refine d
        for _ in range(5):  # Usually converges quickly
            f_d = (self.difC * d_guess + 
                   self.difA * d_guess**2 + 
                   self.difB / np.maximum(d_guess, 1e-10) - 
                   tof_corrected)
            
            df_d = (self.difC + 
                    2.0 * self.difA * d_guess - 
                    self.difB / np.maximum(d_guess**2, 1e-20))
            
            d_guess = d_guess - f_d / np.maximum(np.abs(df_d), 1e-10)
            d_guess = np.maximum(d_guess, 1e-10)  # Keep positive
        
        # Convert d to Q
        q_values = 2.0 * np.pi / d_guess
        return q_values
    
    def get_coverage_limits(self, data_x: np.ndarray) -> Tuple[float, float]:
        """Get coverage limits in Q-space from native data range."""
        if len(data_x) == 0:
            return 0.0, 10.0
            
        x_min, x_max = float(np.min(data_x)), float(np.max(data_x))
        q_limits = self.native_to_q(np.array([x_min, x_max]))
        return float(np.min(q_limits)), float(np.max(q_limits))
    
    def clip_to_coverage(self, x_vals: np.ndarray, y_vals: np.ndarray, 
                        x_limits: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Clip data arrays to specified x-range."""
        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)
        
        x_min, x_max = x_limits
        mask = (x_vals >= x_min) & (x_vals <= x_max)
        
        return x_vals[mask], y_vals[mask]


class IntensityNormalizer:
    """
    Manages intensity normalization to maintain consistency between
    experimental data, simulated patterns, and GSAS-II scale factors.
    """
    
    def __init__(self):
        self.scale_history: Dict[str, float] = {}
        self.normalization_method: str = "max_unity"
        self.reference_scale: float = 1.0
    
    def normalize_experimental(self, intensity: np.ndarray, 
                             method: str = 'max_unity') -> Tuple[np.ndarray, float]:
        """
        Normalize experimental intensity data.
        
        Args:
            intensity: Raw intensity array
            method: Normalization method ('max_unity', 'range_01', 'zscore')
            
        Returns:
            Tuple of (normalized_intensity, scale_factor_applied)
        """
        intensity = np.asarray(intensity, dtype=float)
        
        if method == 'max_unity':
            max_val = float(np.max(intensity))
            if max_val <= 0:
                return intensity, 1.0
            scale_factor = 1.0 / max_val
            normalized = intensity * scale_factor
            
        elif method == 'range_01':
            min_val = float(np.min(intensity))
            max_val = float(np.max(intensity))
            range_val = max_val - min_val
            if range_val <= 0:
                return intensity, 1.0
            scale_factor = 1.0 / range_val
            normalized = (intensity - min_val) * scale_factor
            
        elif method == 'zscore':
            mean_val = float(np.mean(intensity))
            std_val = float(np.std(intensity))
            if std_val <= 0:
                return intensity - mean_val, 1.0
            scale_factor = 1.0 / std_val
            normalized = (intensity - mean_val) * scale_factor
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        self.scale_history['experimental'] = scale_factor
        self.normalization_method = method
        return normalized, scale_factor
    
    def normalize_simulated(self, intensity: np.ndarray, 
                          ref_scale: float = 1.0) -> Tuple[np.ndarray, float]:
        """Normalize simulated pattern to match experimental scale."""
        intensity = np.asarray(intensity, dtype=float)
        
        if self.normalization_method == 'max_unity':
            max_val = float(np.max(intensity))
            if max_val <= 0:
                return intensity, 1.0
            scale_factor = ref_scale / max_val
            normalized = intensity * scale_factor
            
        else:
            # For other methods, use simple scaling
            scale_factor = ref_scale
            normalized = intensity * scale_factor
        
        return normalized, scale_factor
    
    def denormalize_for_gsas(self, normalized_intensity: np.ndarray, 
                           original_scale: float) -> np.ndarray:
        """Convert normalized intensity back to GSAS-compatible scale."""
        return normalized_intensity / original_scale
    
    def track_scale_factors(self) -> Dict[str, float]:
        """Return dictionary of all tracked scale factors."""
        return self.scale_history.copy()
    
    def get_experimental_scale(self) -> float:
        """Get the experimental data scale factor."""
        return self.scale_history.get('experimental', 1.0)


# Test and validation functions
def test_coordinate_conversion():
    """Test coordinate system conversions."""
    print("Testing coordinate conversions...")
    
    # Mock CW instrument parameters
    cw_params = {
        'Type': ['PXC'],
        'Lam': [1.54056],  # Cu Kα
        'Zero': [0.0, 0.0, False]
    }
    
    # Mock TOF instrument parameters  
    tof_params = {
        'Type': ['PXT'],
        'difC': [0.0, 15000.0, False],
        'difA': [0.0, 0.0, False],
        'difB': [0.0, 0.0, False],
        'Zero': [0.0, 0.0, False]
    }
    
    # Test CW conversions
    cw_handler = CoordinateHandler("CW", cw_params)
    test_q = np.array([2.0, 4.0, 6.0])
    two_theta = cw_handler.q_to_native(test_q)
    q_back = cw_handler.native_to_q(two_theta)
    print(f"CW: Q {test_q} -> 2θ {two_theta} -> Q {q_back}")
    
    # Test TOF conversions
    tof_handler = CoordinateHandler("TOF", tof_params)
    tof_vals = tof_handler.q_to_native(test_q)
    q_back_tof = tof_handler.native_to_q(tof_vals)
    print(f"TOF: Q {test_q} -> TOF {tof_vals} -> Q {q_back_tof}")


if __name__ == "__main__":
    # Run basic tests
    test_coordinate_conversion()
    print("Core infrastructure components ready for integration.")