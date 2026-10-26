#!pip install -q opencv-python-headless
import cv2, numpy as np, matplotlib.pyplot as plt

path = "leaf_video.mp4"    # upload the leaf video every run

cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
ok, f0 = cap.read()
prev = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)

# Show frame 0 with grid, then ask for the apex
plt.figure(figsize=(6, 10)); plt.imshow(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB))
plt.grid(alpha=.5); plt.title("Leaf apex: read (col, row)"); plt.show()
APEX = tuple(float(v) for v in input("Enter apex as col,row (e.g. 340,210): ").replace(',', ' ').split())

# Lucas-Kanade tracking (stops at track loss)
p = np.float32([[APEX]]); pts = [p[0, 0].copy()]
while True:
    ok, f = cap.read()
    if not ok: break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    q, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p, None, winSize=(31, 31), maxLevel=3)
    if not st[0, 0]: break
    p = q; pts.append(p[0, 0].copy()); prev = g
cap.release()

pts = np.array(pts); col, row = pts[:, 0], pts[:, 1]
n = len(pts); t = np.arange(n) / fps
rs, cs = row - row.mean(), col - col.mean()

# 1D FFTs and 2D FFT
fr = np.fft.rfftfreq(n, 1 / fps)
R, C = np.abs(np.fft.rfft(rs)), np.abs(np.fft.rfft(cs))
F2 = np.abs(np.fft.fftshift(np.fft.fft2(np.stack([rs, cs], 1))))

fig, ax = plt.subplots(2, 3, figsize=(16, 8))
ax[0, 0].imshow(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB))
ax[0, 0].plot(col, row, 'b-', lw=1); ax[0, 0].plot(*APEX, 'go'); ax[0, 0].set_title("Apex path on frame 0")
ax[0, 1].plot(t, row, label="row"); ax[0, 1].plot(t, col, label="col")
ax[0, 1].set(title="Coordinates vs time", xlabel="s"); ax[0, 1].legend()
ax[0, 2].imshow(F2, aspect="auto", cmap="viridis"); ax[0, 2].set_title("2D FFT magnitude (N x [row, col])")
ax[1, 0].plot(fr, R); ax[1, 0].set(title="1D FFT of row(t)", xlabel="Hz")
ax[1, 1].plot(fr, C); ax[1, 1].set(title="1D FFT of col(t)", xlabel="Hz")
ax[1, 2].axis("off")
plt.tight_layout(); plt.show()

print(f"fps={fps:.2f}, frames tracked={n} ({n/fps:.1f} s), apex={APEX}")
print(f"Dominant freq - row: {fr[1:][R[1:].argmax()]:.3f} Hz, col: {fr[1:][C[1:].argmax()]:.3f} Hz")