# solve.py
from petsc4py import PETSc
from slepc4py import SLEPc

# ---- PETSc binary written in Step 1
BIN = "matrix.dat"
K   = 10      # number of lowest eigenpairs
TOL = 1e-8
# -----------------------------------

comm = PETSc.COMM_WORLD
rank = comm.getRank()

# Clear PETSc errors as Python exceptions:
PETSc.Sys.pushErrorHandler("python")

# Load matrix in parallel; PETSc distributes rows automatically
A = PETSc.Mat().create(comm=comm)
A.setType(PETSc.Mat.Type.AIJ)
vr = PETSc.Viewer().createBinary(BIN, PETSc.Viewer.Mode.READ, comm=comm)
A.load(vr)

#rstart, rend = A.getOwnershipRange()
#print(f"Rank {PETSc.COMM_WORLD.getRank()} owns rows [{rstart}, {rend})")

linfo = A.getInfo(PETSc.Mat.InfoType.LOCAL)
local_nnz = int(getattr(linfo, "nz_used", linfo["nz_used"]))

rank = A.getComm().getRank()
rstart, rend = A.getOwnershipRange()
print(f"[rank {rank}] rows {rstart}:{rend}  nnz={local_nnz}")


# If you prefer to force GPU from code (instead of CLI flags), uncomment:
# try:
#     A.setType("aijcusparse")  # NVIDIA CUDA
#     PETSc.Options()["vec_type"] = "cuda"
# except Exception:
#     if rank == 0:
#         print("[warn] GPU backend not available; using CPU types")

# SLEPc setup: Hermitian problem, robust default solver
eps = SLEPc.EPS().create(comm)
eps.setOperators(A)
eps.setProblemType(SLEPc.EPS.ProblemType.HEP)         # Hermitian
eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)               # Recommended
eps.setDimensions(K)
eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL) # "10 lowest"
eps.setTolerances(TOL)
eps.setFromOptions()  # allow runtime flags: -mat_type, -vec_type, -st_*, -eps_*, etc.

eps.solve()


# --- grab eigenvalues + eigenvectors (gathered to rank 0) ---
k_conv = min(eps.getConverged(), K)

vals = []
vecs = []  # NumPy arrays on rank 0

for i in range(k_conv):
    ev = eps.getEigenvalue(i)
    xr = A.createVecRight()
    eps.getEigenvector(i, xr)

    # Correct: toZero returns (scatter, seq_on_root)
    scatter, seq = PETSc.Scatter.toZero(xr)
    scatter.scatter(xr, seq,
                    addv=PETSc.InsertMode.INSERT_VALUES,
                    mode=PETSc.ScatterMode.FORWARD)

    if rank == 0:
        vals.append(ev)
        vecs.append(seq.getArray().copy())

if rank == 0:
    import numpy as np
    vals = np.asarray(vals)          # (k_conv,)
    X = np.vstack(vecs)              # (k_conv, N)
    print("[result] eigenvalues:", vals)
    np.save("eigenvalues.npy", vals)
    np.save("eigenvectors.npy", X)


#if rank == 0:
#    nconv = eps.getConverged()
#    print(f"[info] converged {nconv}/{K}")
#    for i in range(min(nconv, K)):
#        ev = eps.getEigenvalue(i)
#        print(f"lambda[{i}] = {ev}")

