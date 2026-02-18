import numpy as np

from IsingSimulator import IsingSimulator


# L=16の格子サイズでシミュレーションを開始
# multi=1 は並行して走らせるシステムの数
sim = IsingSimulator(L=16, multi=1)

# 温度を設定
sim.set_temperatures(np.linspace(1, 5, 32))

# 乱数でスピンを初期化
sim.randomize_spins()

# シミュレーション実行 (burn_in=1000, calc_steps=1000)
# 戻り値としてエネルギーや磁化などの統計量を取得
results = sim.run(burn_in_steps=1000, calc_steps=1000)

print(f"Magnetization: {results['m']}")
