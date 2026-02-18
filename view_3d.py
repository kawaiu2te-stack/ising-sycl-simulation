import numpy as np
from vispy import app, scene
from vispy.color import Color

from IsingSimulator import IsingSimulator

# 1. シミュレーター初期化
L = 10
multi = 1
sim = IsingSimulator(L, multi=multi)
sim.randomize_spins()
sim.reset_bonds()
current_temp = 4.5

# 2. 座標の生成 (x, y, z)
z, y, x = np.indices((L, L, L))
pos = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
# 中心を(0,0,0)に寄せる
pos -= L / 2
pos /= L / 2

# 3. キャンバスと表示の設定
canvas = scene.SceneCanvas(show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'  # マウスでカメラ操作

# 4. マーカー（スピン）の作成
markers = scene.visuals.Markers()
view.add(markers)

def set_temperature(T):
    global current_temp
    current_temp = max(0.0, T) # 負の温度は防ぐ
    sim.set_temperatures([current_temp] * 32 * multi)
    canvas.title = f'3D Ising Model T={current_temp:.2f}'
    print(f"Temperature set to: {current_temp:.2f}")

# キーボードイベントハンドラ
@canvas.events.key_press.connect
def on_key_press(event):
    global current_temp

    if event.key == 'Escape':
        canvas.close()
    elif event.text == ']':
        set_temperature(current_temp + 0.1)
    elif event.text == '[':
        set_temperature(current_temp - 0.1)
    elif event.text == 'r':
        print("Randomizing spins...")
        sim.randomize_spins()
    elif event.text == '0':
        print("Resetting to Ordered State...")
        sim.reset_spins()

def get_colors(data):
    """スピンの状態(0/1)を色(青/赤)に変換する"""
    colors = np.zeros((data.size, 4), dtype=np.float32)
    flat_data = data.ravel()
    colors[flat_data == 0] = [0.1, 0.2, 1.0, 0.3] # 0 -> 青
    colors[flat_data == 1] = [1.0, 0.1, 0.1, 0.6] # 1 -> 赤
    return colors

# 5. リアルタイム更新用の関数
def update(ev):
    sim.run(1, 0)
    grid = sim.get_spin(0)

    # 色データを更新
    markers.set_data(pos, face_color=get_colors(grid), size=300 / L, edge_width=0)

if __name__ == '__main__':
    print("Commands:")
    print("  '[' : Decrease Temperature")
    print("  ']' : Increase Temperature")
    print("  'r' : Randomize Spins")
    print("  '0' : Reset to Ordered State")
    print("  Esc : Quit")

    set_temperature(current_temp)

    # 60FPS程度で更新
    timer = app.Timer(interval=1/60.0, connect=update, start=True)

    app.run()
