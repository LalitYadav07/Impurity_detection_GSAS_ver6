import numpy as np

def compile_even_quadratic(forward, dimensions):
    """Central polarization removes odd artifacts from native zero-component guards.

    GSAS replaces exact zero magnetic components by 1e-8 internally. A one-sided
    basis construction turns those tiny offsets into spurious cross terms.
    Symmetric evaluations cancel the first-order offset contribution. Independent
    native checks remain mandatory; this is not a general surrogate guarantee.
    """
    eye=np.eye(dimensions);zero=np.asarray(forward(np.zeros(dimensions)),float)
    q=np.zeros((dimensions,dimensions,len(zero)))
    for i in range(dimensions):q[i,i]=(forward(eye[i])+forward(-eye[i]))/2-zero
    for i in range(dimensions):
        for j in range(i):
            q[i,j]=q[j,i]=(forward(eye[i]+eye[j])+forward(-eye[i]-eye[j])-forward(eye[i]-eye[j])-forward(-eye[i]+eye[j]))/8
    return q
