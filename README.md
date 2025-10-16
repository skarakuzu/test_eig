First convert the `.npz` file to petsc format
```bash
pixi run convert
```

Diagonalize the problem in a parallel fashion
```bash
pixi run solve_cpu
```
