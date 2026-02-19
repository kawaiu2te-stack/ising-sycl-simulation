# 3D Ising Model Monte Carlo Simulation (SYCL + Python Interface)

SYCLを用いた3次元Ising模型のモンテカルロ・シミュレーション実装です。
C++による高速な計算コアと、Pythonによる柔軟な制御インターフェースを組み合わせることで、再コンパイルなしで動的なパラメータ変更やレプリカ交換法（Parallel Tempering）のような複雑なアルゴリズムの実装を可能にしています。

## 特徴 (Features)

  - **High Performance Core (C++ / SYCL)**

      - **Cross-Architecture**: NVIDIA GPU (CUDA) と Intel GPU (Level Zero/OpenCL) の両方で動作。
      - **Multi-spin coding**: ビット演算を活用し、1スレッドで複数のスピンを同時更新。
      - **MTGP32**: GPU向けの高速な乱数生成アルゴリズム MTGP32 をSYCL向けに移植・最適化。

  - **Flexible Control (Python)**

      - 標準入出力パイプラインを用いたC++コアとの通信。
      - Pythonスクリプトからシミュレーション中に「温度変更」「スピン取得」「レプリカ交換」などが可能。

## 要件 (Requirements)

  - **C++ Core**:
      - C++23 compatible compiler (icpx, clang++, etc.)
      - Intel oneAPI Base Toolkit (for SYCL)
      - Make
  - **Python Interface**:
      - Python 3.8+
      - NumPy
      - VisPy, PyQt5 (For Visualization)

## セットアップ (Setup)

### 1\. Pythonライブラリのインストール

```bash
pip install -r requirements.txt
```

### 2\. ビルド

`Makefile` が環境（NVIDIA GPU vs Intel GPU）を自動検出してビルドします。

```bash
make
# 実行ファイル 'main_sycl' が生成されます
```

## 使用方法 (Usage)

### 1\. 3D可視化デモ

シミュレーションの様子をリアルタイムで3D描画します。

```bash
python view_3d.py
```

  - **操作方法**:
      - `[` / `]`: 温度を下げる / 上げる
      - `r`: スピンをランダム化 (高温極限)
      - `0`: スピンを整列 (低温極限)
      - `Mouse Drag`: カメラ回転

### 2\. 数値実験デモ

温度を自動で変化させ、エネルギーと磁化の変化を出力します。

```bash
python demo.py
```

### 3\. Pythonスクリプトからの制御 (IsingSimulator)

`IsingSimulator` クラスを使用することで、複雑なプロセス管理を隠蔽してシミュレーションを操作できます。

```python
from IsingSimulator import IsingSimulator

# L=32の格子サイズでシミュレーションを開始
# multi=1 は並行して走らせるシステムの数
sim = IsingSimulator(L=32, multi=1)

# 温度を設定
sim.set_temp(4.51) # Critical temperature

# 乱数でスピンを初期化
sim.randomize_spins()

# シミュレーション実行 (burn_in=1000, calc_steps=1000)
# 戻り値としてエネルギーや磁化などの統計量を取得
results = sim.run(burn_in=1000, calc_steps=1000)

print(f"Magnetization: {results['m']}")
```

## ファイル構成

  - `main_sycl.cpp`: シミュレーションのコアロジック (SYCL)
  - `mtgp32dc_params_fast_11213.h`: MTGP32乱数生成器のパラメータ
  - `IsingSimulator.py`: C++プロセスを制御するPythonラッパー
  - `view_3d.py`: リアルタイム3D可視化ツール
  - `demo.py`: アニーリング実験デモ

## ライセンス (License)

乱数生成部 (`mtgp32dc_params_fast_11213.h` およびアルゴリズム) は、オリジナルの[MTGP](https://github.com/MersenneTwister-Lab/MTGP) (Hiroshima University) のライセンスに基づきます。
