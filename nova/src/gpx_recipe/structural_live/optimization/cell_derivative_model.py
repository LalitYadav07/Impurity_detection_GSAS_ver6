"""Short-lived lattice derivative proposals with native acceptance.

This stores derivatives of the *current observation's profiled rate*, not a
portable physical response or a fitted uncertainty. It never supplies an
objective value. Stalled or apparently stationary recycled proposals must be
retried with fresh native differences by the caller.
"""
import numpy as np


class CellDerivativeModel:
    def __init__(self, modes, dimensions, *, max_reuses=3, maximum_step=.2, mismatch_limit=.25,
                 maximum_bytes=64*1024*1024):
        if (type(modes) is not int or type(dimensions) is not int
                or not 0 <= modes <= dimensions
                or type(max_reuses) is not int or max_reuses < 1
                or type(maximum_bytes) is not int or maximum_bytes < 1
                or not np.isfinite([maximum_step, mismatch_limit]).all()
                or maximum_step <= 0 or not 0 < mismatch_limit < 1):
            raise ValueError('Bounded lattice derivative proposal settings required')
        self.modes=modes;self.dimensions=dimensions
        self.max_reuses=max_reuses;self.maximum_step=maximum_step;self.mismatch_limit=mismatch_limit
        self.maximum_bytes=maximum_bytes;self.memory_bypasses=0
        self.model=None;self.latest_modes=None;self.last_reused=False
        self.builds=0;self.reuses=0;self.same_point_hits=0;self.invalidations=0;self.rejected_predictions=0

    def invalidate(self):
        self.model=None;self.latest_modes=None;self.last_reused=False;self.invalidations+=1

    def _point(self, point, rate):
        point=np.asarray(point,float);rate=np.asarray(rate,float)
        if (point.shape != (self.dimensions,) or rate.ndim != 1 or not len(rate)
                or not np.isfinite(point).all() or not np.isfinite(rate).all()):
            raise ValueError('Complete finite structural point and native profiled rate required')
        return point,rate

    def _storage_allowed(self, rate):
        # Include both mode snapshots, lattice columns, copied return arrays
        # and native-rate anchors. This bounds this optional cache only; the
        # caller still accounts for its necessary native derivative workspace.
        return 8*((2*self.dimensions+8)*len(rate)+5*self.dimensions)<=self.maximum_bytes

    def observe_modes(self, point, rate, derivatives):
        point,rate=self._point(point,rate);derivatives=np.asarray(derivatives,float)
        if derivatives.shape != (self.modes,len(rate)) or not np.isfinite(derivatives).all():
            raise ValueError('All finite native displacement derivatives required')
        if not self._storage_allowed(rate):
            self.model=None;self.latest_modes=None;return
        self.latest_modes=dict(point=point.copy(),rate=rate.copy(),modes=derivatives.copy())
        # Refresh only the displacement columns. Retain the earlier point
        # and rate so the NEXT native rate change still tests this proposal;
        # re-anchoring here would erase the error we need to measure.
        if self.model is not None:
            if np.max(abs(point-self.model['native_point']),initial=0.) > self.maximum_step:
                self.model=None;self.invalidations+=1
            else:
                self.model['modes']=derivatives.copy()

    def columns(self, point, rate, weights, build):
        point,rate=self._point(point,rate);weights=np.asarray(weights,float)
        if weights.shape != rate.shape or not np.isfinite(weights).all() or np.any(weights<0) or not np.any(weights>0):
            raise ValueError('Finite observation weights with fitted support required')
        if not self._storage_allowed(rate):
            self.model=None;self.latest_modes=None;self.last_reused=False
            cells=np.asarray(build(),float)
            if cells.shape != (self.dimensions-self.modes,len(rate)) or not np.isfinite(cells).all():
                raise ValueError('All finite native lattice derivative columns required')
            self.builds+=1;self.memory_bypasses+=1
            return cells
        model=self.model;self.last_reused=False
        if model is not None and len(model['rate']) == len(rate):
            delta=point-model['point']
            if np.array_equal(point,model['point']) and np.array_equal(rate,model['rate']):
                self.same_point_hits+=1
                self.last_reused=bool(model['age'])
                return model['cells'].copy()
            reusable=(model['age']<self.max_reuses
                and np.max(abs(point-model['native_point']),initial=0.)<=self.maximum_step
                and (model['modes'] is not None or not np.any(delta[:self.modes])))
            if reusable:
                predicted=model['cells'].T@delta[self.modes:]
                if model['modes'] is not None:predicted+=model['modes'].T@delta[:self.modes]
                change=rate-model['rate'];error=change-predicted
                mismatch=float(np.linalg.norm(error*weights)/max(np.linalg.norm(change*weights),1e-12))
                if np.isfinite(mismatch) and mismatch<=self.mismatch_limit:
                    cells=model['cells'].copy();cell_delta=delta[self.modes:]
                    denominator=float(cell_delta@cell_delta)
                    # A predominantly displacement step must not amplify a
                    # tiny lattice displacement into a huge Broyden correction.
                    if denominator>1e-16 and denominator>=.01*float(delta@delta):
                        cells+=np.outer(cell_delta/denominator,error)
                    model.update(point=point.copy(),rate=rate.copy(),cells=cells,age=model['age']+1)
                    self.reuses+=1;self.last_reused=True
                    return cells.copy()
                self.rejected_predictions+=1
        cells=np.asarray(build(),float)
        if cells.shape != (self.dimensions-self.modes,len(rate)) or not np.isfinite(cells).all():
            raise ValueError('All finite native lattice derivative columns required')
        modes=np.empty((0,len(rate))) if self.modes==0 else None
        if (self.latest_modes is not None and len(rate)==len(self.latest_modes['rate'])
                and np.max(abs(point-self.latest_modes['point']),initial=0.)<=self.maximum_step):
            modes=self.latest_modes['modes'].copy()
        self.model=dict(point=point.copy(),native_point=point.copy(),rate=rate.copy(),
            modes=modes,cells=cells.copy(),age=0)
        self.builds+=1
        return cells.copy()

    def diagnostics(self):
        return dict(builds=self.builds,reuses=self.reuses,same_point_hits=self.same_point_hits,
            invalidations=self.invalidations,rejected_predictions=self.rejected_predictions,
            max_reuses=self.max_reuses,maximum_step=self.maximum_step,mismatch_limit=self.mismatch_limit,
            maximum_cache_bytes=self.maximum_bytes,memory_bypasses=self.memory_bypasses,
            scope='Current-observation lattice proposals only; full native objective acceptance',
            uncertainty_derivatives_qualified=False)
