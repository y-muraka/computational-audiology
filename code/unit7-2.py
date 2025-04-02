import numpy as np
from scipy.linalg import lu_factor, lu_solve
import time

def solve_with_numpy(A, b):
    """
    NumPyのnp.linalg.solveを使用して連立方程式を解く
    """
    start_time = time.time()
    x = np.linalg.solve(A, b)
    end_time = time.time()
    return x, end_time - start_time

def solve_with_lu(A, b):
    """
    LU分解を使用して連立方程式を解く
    """
    start_time = time.time()
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)
    end_time = time.time()
    return x, end_time - start_time

# より大きな行列でパフォーマンス比較
sizes = [10, 100, 500]

for n in sizes:
    print(f"\n{n}x{n}行列:")
    
    # ランダムな正則行列を生成
    A = np.random.rand(n, n)
    b = np.random.rand(n)
    
    # NumPyのsolveを使用して解く
    numpy_solution, numpy_time = solve_with_numpy(A, b)
    lu_solution, lu_time = solve_with_lu(A, b)

    # 解の差を計算して精度を比較
    diff = np.linalg.norm(numpy_solution - lu_solution)
    print(f"\n両解法の差のノルム: {diff:.15e}")

