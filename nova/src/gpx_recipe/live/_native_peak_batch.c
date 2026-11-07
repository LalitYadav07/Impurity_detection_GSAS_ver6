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

/* Apply native unit-F2 columns without storing their redundant row indices
 * and double-precision weighted copies. Raw peaks remain original float32
 * outputs; the factor multiplication is still performed in double precision
 * before the structural coefficient, in the same reflection order as CSC. */
int native_window_multiply(ptrdiff_t nc,ptrdiff_t nr,ptrdiff_t nx,ptrdiff_t outputs,
                    const int64_t *mapping,const int64_t *bounds,const int64_t *pointers,
                    const double *factors,const float *raw,const double *weights,double *result) {
    if (nc<0||nr<0||nx<1||outputs<1||outputs>256) return 1;
    for (ptrdiff_t col=0;col<nc;++col) {
        const int64_t row=mapping[col];
        if (row==-1) continue;
        if (row<0||row>=nr) return 2;
        const int64_t low=bounds[2*row],high=bounds[2*row+1],begin=pointers[row];
        if (low<0||high>nx||high<=low||begin<0||high-low!=pointers[row+1]-begin) return 3;
        for (ptrdiff_t j=0;j<high-low;++j) {
            const double unit=factors[row]*(double)raw[begin+j];
            for (ptrdiff_t k=0;k<outputs;++k) {
                const double contribution=unit*weights[col*outputs+k];
                result[(low+j)*outputs+k]=result[(low+j)*outputs+k]+contribution;
            }
        }
    }
    return 0;
}

/* Proposal derivatives only. Every peak value and all five partials come
 * from the original installed GSAS-II Fortran derivative entry point. */
typedef void (*derivative_function)(int *,float *,float *,float *,float *,float *,
                                  float *,float *,float *,float *,float *,float *);
int native_peak_derivative_columns(derivative_function function,ptrdiff_t nr,ptrdiff_t nx,ptrdiff_t nc,
    const double *axis,const double *parameters,const double *height,const int64_t *bounds,
    const double *dp,const double *dh,ptrdiff_t maximum,float *scratch,double *signal,double *columns) {
    if (!function||nr<0||nx<1||nc<1||nc>24||maximum<2||maximum>INT_MAX) return 1;
    for (ptrdiff_t row=0;row<nr;row++) {
        const int64_t lo=bounds[2*row],hi=bounds[2*row+1];
        if (lo<0||hi>nx||hi-lo<2||hi-lo>maximum) return 2;
        const double *p=parameters+row*5;
        float a=(float)p[1],b=(float)p[2],s=(float)p[3],g=(float)p[4];
        int n=(int)(hi-lo);
        float *offset=scratch,*value=scratch+maximum,*dt=scratch+2*maximum;
        float *da=scratch+3*maximum,*db=scratch+4*maximum,*ds=scratch+5*maximum,*dg=scratch+6*maximum;
        for (int j=0;j<n;j++) offset[j]=(float)(axis[lo+j]-p[0]);
        function(&n,offset,&a,&b,&s,&g,value,dt,da,db,ds,dg);
        for (int j=0;j<n;j++) {
            signal[lo+j]+=height[row]*(double)value[j];
            for (ptrdiff_t c=0;c<nc;c++) {
                const double *v=dp+(row*nc+c)*5;
                double derivative=v[0]*(double)dt[j];
                derivative+=v[1]*(double)da[j];
                derivative+=v[2]*(double)db[j];
                derivative+=v[3]*(double)ds[j];
                derivative+=v[4]*(double)dg[j];
                columns[c*nx+lo+j]+=dh[row*nc+c]*(double)value[j]+height[row]*derivative;
            }
        }
    }
    return 0;
}
