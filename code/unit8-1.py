import numpy as np
import matplotlib.pyplot as plt

# テスト関数とその導関数
def f(x):
    return np.sin(x)

def df(x):  # 実際の導関数
    return np.cos(x)

# 前方差分近似
def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h

# 中心差分近似
def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# エラー計算
def compute_errors(x_val, h_values):
    true_deriv = df(x_val)
    
    forward_errors = []
    central_errors = []
    
    for h in h_values:
        # 各差分法による近似
        forward_approx = forward_diff(f, x_val, h)
        central_approx = central_diff(f, x_val, h)
        
        # 絶対誤差の計算
        forward_error = abs(forward_approx - true_deriv)
        central_error = abs(central_approx - true_deriv)
        
        forward_errors.append(forward_error)
        central_errors.append(central_error)
    
    return forward_errors, central_errors

# 誤差の理論的なオーダーを示す参照線
def reference_lines(h_values):
    # O(h)参照線
    order_h = [h for h in h_values]
    # O(h^2)参照線
    order_h2 = [h**2 for h in h_values]
    # O(1/h)参照線（丸め誤差の影響）
    order_inv_h = [1.0/h for h in h_values]
    
    # スケーリング係数（グラフ上で見やすくするため）
    scale_h = 0.1
    scale_h2 = 0.1
    scale_inv_h = 1e-16  # マシンイプシロン程度のスケール
    
    return [scale_h * o for o in order_h], [scale_h2 * o for o in order_h2], [scale_inv_h * o for o in order_inv_h]

# メイン実行部分
def main():
    # テストする点
    x_val = np.pi / 4  # π/4 (45度)
    
    # 刻み幅hの範囲（対数スケール）
    h_values = np.logspace(-12, 0, 200)  # 10^(-16) から 10^0 までの範囲
    
    # 理論的な最適なhを計算
    # 前方差分: h_opt ≈ sqrt(ε)
    # 中心差分: h_opt ≈ ε^(1/3)
    epsilon = np.finfo(float).eps  # マシンイプシロン
    h_opt_forward = np.sqrt(epsilon)
    h_opt_central = np.power(epsilon, 1/3)
    
    # 誤差計算
    forward_errors, central_errors = compute_errors(x_val, h_values)
    
    # 理論的オーダーの参照線
    ref_h, ref_h2, ref_inv_h = reference_lines(h_values)
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, forward_errors, 'r-', label='Forward Difference Error')
    plt.loglog(h_values, central_errors, 'b-', label='Central Difference Error')
    plt.loglog(h_values, ref_h, 'k--', label='O(h) Reference')
    plt.loglog(h_values, ref_h2, 'g--', label='O(h²) Reference')
    plt.loglog(h_values, ref_inv_h, 'm--', label='O(1/h) Rounding Error')
    
    # Graph settings in English
    plt.xlabel('Step Size h')
    plt.ylabel('Absolute Error')
    plt.title('Error Comparison: Forward vs Central Difference (f(x) = sin(x), x = π/4)')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    
    # Annotate the total error model
    #plt.text(1e-5, 1e-12, "Total Error ≈ C₁h^p + C₂ε/h", fontsize=10, 
             #bbox=dict(facecolor='white', alpha=0.7))
             
    # Add minimum point annotation for both methods
    min_h_forward = h_values[np.argmin(forward_errors)]
    min_error_forward = min(forward_errors)
    min_h_central = h_values[np.argmin(central_errors)]
    min_error_central = min(central_errors)
    
    #plt.scatter(min_h_forward, min_error_forward, color='r', s=50, zorder=5)
    #plt.scatter(min_h_central, min_error_central, color='b', s=50, zorder=5)
    
    #plt.annotate(f"Optimal h ≈ {min_h_forward:.2e}", 
    #             xy=(min_h_forward, min_error_forward), 
    #             xytext=(min_h_forward*5, min_error_forward*5),
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
                 
    # Highlight the region where rounding errors dominate
    plt.axvspan(1e-10, 1e-7, alpha=0.2, color='yellow', label='Rounding Error Dominant')
    
    # 図の保存
    plt.savefig('diff_error_comparison.pdf')
    plt.show()

if __name__ == "__main__":
    main()