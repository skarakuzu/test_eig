# convert.py
import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

# ---- change the filename if needed
NPZ = "H_initial_Ni2O11.npz"
OUT = "matrix.dat"
# -----------------------------------

A = sp.load_npz(NPZ).tocsr()

# Handle complex matrices with real PETSc
if np.iscomplexobj(A.data) and np.dtype(PETSc.ScalarType).kind != "c":
    import warnings
    warnings.warn(
        "Matrix is complex but PETSc is built for real scalars. "
        "Imaginary part will be discarded."
    )
    A.data = A.data.real  

# enforce Hermitian by symmetrizing tiny round-off:
# A = (A + A.getH()) * 0.5

I = A.indptr.astype(PETSc.IntType, copy=False)
J = A.indices.astype(PETSc.IntType, copy=False)
V = A.data.astype(np.dtype(PETSc.ScalarType), copy=False)

M = PETSc.Mat().createAIJ(size=A.shape, csr=(I, J, V), comm=PETSc.COMM_SELF)
M.assemblyBegin(); M.assemblyEnd()

vw = PETSc.Viewer().createBinary(OUT, PETSc.Viewer.Mode.WRITE, comm=PETSc.COMM_SELF)
M.view(vw)

print(f"[ok] wrote PETSc binary: {OUT}  shape={A.shape}  nnz={A.nnz}  dtype={A.dtype}")

