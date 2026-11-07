/* Bounded scheduling adapter for the existing GSAS-II Fortran kernel.
 * There is no peak-shape formula, interpolation or saved prediction here.
 */
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

typedef void (*peak_function)(int *,float *,float *,float *,float *,float *,float *);

int native_peak_batch_abi(void) {
    return sizeof(int)==4 && sizeof(float)==4 && sizeof(double)==8 &&
           sizeof(int64_t)==8 && sizeof(ptrdiff_t)==8 ? 0x504e5401 : 0;
}

int native_peak_sum(peak_function peak,ptrdiff_t nr,ptrdiff_t nx,
                    const double *axis,const double *parameters,
                    const double *heights,const int64_t *bounds,
                    ptrdiff_t maximum_bins,float *offsets,float *values,
                    double *signal,ptrdiff_t *points) {
    *points=0;
    if (!peak||nr<0||nx<1||maximum_bins<2||maximum_bins>INT_MAX) return 1;
    for (ptrdiff_t row=0;row<nr;++row) {
        const int64_t low=bounds[2*row],high=bounds[2*row+1];
        const double *p=parameters+5*row;
        if (low<0||high>nx||high-low<2||high-low>maximum_bins) return 2;
        if (!isfinite(heights[row])) return 3;
        for (int j=0;j<5;++j) if (!isfinite(p[j])) return 3;
        float alpha=(float)p[1],beta=(float)p[2],sigma=(float)p[3],gamma=(float)p[4];
        if (!isfinite(alpha)||!isfinite(beta)||!isfinite(sigma)||!isfinite(gamma)) return 3;
        int n=(int)(high-low);
        for (int j=0;j<n;++j) offsets[j]=(float)(axis[low+j]-p[0]);
        peak(&n,offsets,&alpha,&beta,&sigma,&gamma,values);
        /* NumPy's native loop likewise materializes the double product before
         * adding it. Required compiler flags forbid fused multiply/add. */
        for (int j=0;j<n;++j) {
            const double contribution=heights[row]*(double)values[j];
            signal[low+j]=signal[low+j]+contribution;
        }
        *points+=n;
    }
    return 0;
}
