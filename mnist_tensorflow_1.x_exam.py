import tensorflow as tf
import pandas as pd

# 1. 하이퍼파라미터 설정
learning_rate = 0.001
num_epochs = 10
batch_size = 100

# 2. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# 원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3. 모델 구성 (Keras Sequential API 사용)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10) # 출력층 (logits)
])

# 4. 손실 함수, 옵티마이저 정의 및 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 5. 모델 학습 (반복문 없이 fit 메서드로 자동 처리)
print("학습 시작...")
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))

# 6. 최종 평가
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Final Test Accuracy: {accuracy * 100:.4f} %")
print("\nTensorflow Version:", tf.__version__)

# 7. Pandas 데이터프레임 출력
data = {
    '이름': ['황다현'],
    '학번': [2513260],
    '학과': ['인공지능공학부']
}

print()
df = pd.DataFrame(data)
print(df)
print()

    