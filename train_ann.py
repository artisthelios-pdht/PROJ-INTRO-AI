"""
train_ann.py
=============
Huấn luyện ANN (Keras Sequential) để học bắt chước hàm heuristic tay,
dựa trên dữ liệu sinh ra từ data_generation.py.

Xử lý quan trọng: nhãn y có vài trăm mẫu là +-100000 (thế thắng/thua
tuyệt đối) trong khi đa số mẫu chỉ vài chục/trăm điểm. Nếu để nguyên,
MSE sẽ bị các outlier này chi phối hoàn toàn -> ANN học sai lệch.

=> Cách xử lý: GIỚI HẠN (clip) nhãn về khoảng [-CLIP, CLIP] rồi chia
   tỷ lệ tuyến tính về [-1, 1]. Lý do chọn clip tuyến tính thay vì
   log/exp: phép log nén quá mạnh khiến một sai số nhỏ trên thang log
   bị khuếch đại theo cấp số mũ khi giải nén (đã thử và thấy sai lệch
   lớn ngoài thực tế) -> không phù hợp khi minimax chỉ cần ANN ước
   lượng ĐÚNG THỨ TỰ (cái nào tốt hơn cái nào), không cần đúng tuyệt đối.
   Các trạng thái thắng/thua thật ANN không cần đoán chính xác con số
   100000, vì minimax đã tự phát hiện is_terminal() bằng luật cứng rồi.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # không cần hiển thị màn hình, chỉ lưu file ảnh
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

CLIP_VALUE = 300.0  # các thế cờ thường (chưa kết thúc) hiếm khi vượt mức này


def scale_label(y, clip_value: float = CLIP_VALUE):
    y_clipped = np.clip(y, -clip_value, clip_value)
    return y_clipped / clip_value  # về khoảng [-1, 1]


def inverse_scale_label(y_scaled, clip_value: float = CLIP_VALUE):
    return y_scaled * clip_value


def load_and_prepare_data(path="gomoku_dataset.npz", val_ratio=0.15, test_ratio=0.15, seed=42):
    data = np.load(path)
    X, y = data["X"], data["y"]
    y_scaled = scale_label(y)

    n = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    X, y_scaled = X[idx], y_scaled[idx]

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    X_train, y_train = X[:n_train], y_scaled[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y_scaled[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y_scaled[n_train + n_val:]

    print(f"Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)} "
          f"({len(X_train)/n:.0%}/{len(X_val)/n:.0%}/{len(X_test)/n:.0%})")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_model(hidden_layers=(64, 32), input_dim=36):
    """ANN dạng Sequential: Lego-style model.add(Dense(...)) như đề bài gợi ý."""
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(1, activation="tanh"))  # nhãn đã scale về [-1, 1]
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def plot_history(history, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Validation MAE")
    axes[1].set_title(f"{title} - MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  Đã lưu đồ thị: {filename}")


if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_prepare_data()

    print("\n=== Huấn luyện ANN heuristic cho Gomoku ===")
    model = build_model(hidden_layers=(64, 32))
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )
    plot_history(history, "Gomoku Heuristic ANN", "gomoku_ann_training.png")

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest MSE (thang log): {test_loss:.4f} | Test MAE (thang log): {test_mae:.4f}")

    # Kiểm tra trực quan trên thang điểm THẬT (sau khi giải nén)
    y_pred_scaled = model.predict(X_test[:5], verbose=0).flatten()
    y_pred_real = inverse_scale_label(y_pred_scaled)
    y_true_real = inverse_scale_label(y_test[:5])
    print("\nSo sánh 5 mẫu test (thang điểm thật):")
    for true_v, pred_v in zip(y_true_real, y_pred_real):
        print(f"  Thật: {true_v:8.1f}   |   ANN dự đoán: {pred_v:8.1f}")

    model.save("gomoku_ann_model.keras")
    print("\nĐã lưu model: gomoku_ann_model.keras")
