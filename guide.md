# 텐서플로우 즉시 실행 (TensorFlow Eager Execution)

텐서플로우의 즉시 실행 (Eager execution)은 그래프 생성 없이 연산을 즉시 실행하는 명령형 프로그래밍 환경을 뜻합니다.
각 연산들은 나중에 실행할 계산 그래프를 만드는 것이 아니라, 실제 값이 반환됩니다.
이를 통해 텐서플로우를 좀더 쉽게 시작할 수 있고, 모델을 디버그 할 수 있습니다. 또한 불필요한 상용구도 줄여줍니다.
이 가이드를 따라하기 위해, 아래 코드를 대화형 `python` 인터프리터를 통해 실행해보세요.

즉시 실행 (eager execution)은 연구와 실험을 위해 다음을 제공하는 유연한 기계학습 플랫폼입니다:
* *직관적인 인터페이스*—사용자 코드를 자연스럽게 구조화 하고, 파이썬 데이터 구조를 사용합니다.
작은 모델과 작은 데이터에 대해서도 빠르게 반복수행 가능합니다.
* *쉬운 디버깅*—실행중인 모델을 검사하거나 변화사항을 평가할 때 연산들을 직접 호출할 수 있습니다.
* *자연스러운 흐름 제어*—동적 모델의 명세를 단순화 시켜, 그래프 흐름 제어 대신 파이썬 흐름 제어를 사용할 수 있습니다.

즉시 실행 (eager execution)은 텐서플로우의 대부분 연산 및 GPU 가속화를 지원합니다. 즉시 실행 (eager execution)이 동작하는 예제들은 다음 링크에서 확인해보세요: [tensorflow/contrib/eager/python/examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples).

주의: 몇몇 모델들은 즉시 실행 (eager execution)을 수행하기 위해 오버헤드가 증가할 수 있습니다. 성능 개선은 계속 진행중이며, 만약 문제점을 발견하거나 사용자 벤치마크 공유를 원한다면 [버그 보고하기](https://github.com/tensorflow/tensorflow/issues) 를 이용해주세요.


## 설정 및 기본 사용법 (Setup and Basic Usage)

가장 최신 버전의 텐서플로우로 업그레이드 하세요:

```
$ pip install --upgrade tensorflow
```

즉시 실행 (eager execution)을 시작하기 위해서, `tf.enable_eager_execution()` 구문을 프로그램이나 콘솔세션 제일 첫 부분에 추가하세요. 프로그램을 호출하는 다른 모듈에 이 연산을 추가하지 마세요.

```py
from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()
```

이제 사용자는 텐서플로우 연산을 실행할 수 있고, 결과를 즉시 확인할 수 있습니다:

```py
tf.executing_eagerly()        # => True

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))  # => "hello, [[4.]]"
```
즉시 실행 (eager execution)을 활성화하여 텐서플로우 연산이 즉시 실행되어 파이썬에게 그 값을 반환하여 줄수있도록 동작을 바꾸게 됩니다. `tf.Tensor` 객체는 계산 그래프의 노드에 대한 심볼릭 핸들 대신  실제 값을 참조합니다. 세션에서 생성하고 이후 실행할 계산 그래프가 없기 때문에, `print()` 구문이나 디버거를 이용하여 결과를 확인하는 것은 굉장히 쉬워졌습니다. 텐서 값을 평가, 출력 및 확인하는 것은 경사도 (gradients)를 계산하는 흐름을 방해하지 않습니다.

즉시 실행 (eager execution)은 [NumPy](http://www.numpy.org/)와 호환성이 매우 뛰어납니다. NumPy 연산은 `tf.Tensor`를 인자로 받습니다. 텐서플로우 [수학 연산](https://www.tensorflow.org/api_guides/python/math_ops)은 파이썬 객체와 NumPy 배열을 `tf.Tensor` 객체로 변환합니다. `tf.Tensor.numpy` 함수는 객체의 값을 NumPy `ndarray`형태로 반환합니다.

```py
a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
# => tf.Tensor([[1 2]
#               [3 4]], shape=(2, 2), dtype=int32)

# 브로드캐스팅을 지원합니다.
b = tf.add(a, 1)
print(b)
# => tf.Tensor([[2 3]
#               [4 5]], shape=(2, 2), dtype=int32)

# 연산자 오버로딩을 지원합니다.
print(a * b)
# => tf.Tensor([[ 2  6]
#               [12 20]], shape=(2, 2), dtype=int32)

# NumPy 값을 써봅시다.
import numpy as np

c = np.multiply(a, b)
print(c)
# => [[ 2  6]
#     [12 20]]

# 텐서로부터 numpy 형태의 값 받기:
print(a.numpy())
# => [[1 2]
#     [3 4]]
```

`tfe` 모듈은 즉시 실행 모드와 그래프 실행 모드 양쪽다 동작하는 다양한 기능들을 포함하고 있습니다. 이 방식은 [그래프로 동작하기](#그래프로 동작하기)에서 찾아 볼 수 있습니다:

```py
import tensorflow.contrib.eager as tfe
```

## 동적 흐름 제어 (Dynamic control flow)

즉시 실행 (eager execution)의 가장 큰 장점은 모델이 실행될 때, 호스트 언어의 모든 기능들을 활용할 수 있다는 점입니다. 예를 들어, [fizzbuzz](https://en.wikipedia.org/wiki/Fizz_buzz) 같은 코드를 쉽게 작성할 수 있습니다:

```py
def fizzbuzz(max_num):
  counter = tf.constant(0)
  for num in range(max_num):
    num = tf.constant(num)
    if num % 3 == 0 and num % 5 == 0:
      print('FizzBuzz')
    elif num % 3 == 0:
      print('Fizz')
    elif num % 5 == 0:
      print('Buzz')
    else:
      print(num)
    counter += 1
  return counter
```

위 예제는 텐서의 값에 따라 달라지게 되며, 런타임에서 각 값들을 출력 할 수 있습니다.


## 모델 작성 (Build a model)

많은 기계 학습 모델들은 레이어를 조합하는 것으로 표현되고 있습니다. 텐서플로우의 즉시 실행 기능과 함께 사용자는 본인만의 레이어를 작성하거나 `tf.keras.layers` 패키지에서 제공되는 레이어를 활용 할 수 있습니다.

사용자가 레이어를 표현하기 위해 파이썬 객체를 사용할 때, 텐서플로우는 간편하게 `tf.keras.layers.Layer`를  base 클래스로 제공합니다. 사용자만의 레이어를 만들기 위해 다음과 같이 상속받아서 사용할 수 있습니다:


```py
class MySimpleLayer(tf.keras.layers.Layer):
  def __init__(self, output_units):
    self.output_units = output_units

  def build(self, input):
    # build 함수는 레이어가 가장 처음 사용될 때 호출됩니다.
    # build()에서 변수를 만드는 것은 입력 shape에 따라 변수의 shape 결정할 수 있게 합니다.
    # 따라서, 사용자가 전체 shape을 명시할 필요 없도록 해줍니다.
    # 만약 사용자가 전체 shape을 이미 알고 있다면, __init__() 에서 변수를 만드는 것도 가능합니다.
    self.kernel = self.add_variable(
      "kernel", [input.shape[-1], self.output_units])

  def call(self, input):
    # __call__ 대신 call()을 오버라이드 가능.
    return tf.matmul(input, self.kernel)
```

위에서 작성한 `MySimpleLayer` 대신 `tf.keras.layers.Dense` 레이어를 사용해보면, 기능적인 면에서 더 상위 개념을 포함합니다 (bias를 추가할 수 있습니다).

레이어를 조합하여 모델을 만들고자 할 때, 사용자는 `tf.keras.Sequential` 사용하여 레이어의 선형적인 스택 구조를 표현 할 수 있습니다. 기본적인 모델을 위해 쉽게 사용할 수 있습니다:


```py
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, input_shape=(784,)),  # 입력 shape을 반드시 선언해줘야 한다.
  tf.keras.layers.Dense(10)
])
```

반면, `tf.keras.Model`로 부터 상속받은 클래스를 이용하여 모델을 구성할 수 있습니다. 이는 레이어 자체로 구성된 레이어들을 담는 컨테이너이며, `tf.keras.Model` 객체는 다른 `tf.keras.Model` 객체들을 포함할 수 있습니다.

```py
class MNISTModel(tf.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=10)
    self.dense2 = tf.keras.layers.Dense(units=10)

  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    result = self.dense2(result)  # reuse variables from dense2 layer
    return result

model = MNISTModel()
```
처음 입력이 레이어에 인자값들로 주어지면서 설정되기 때문에, `tf.keras.Model` 클래스를 위해 입력 shape을 따로 설정 할 필요가 없습니다.

`tf.keras.layers` 클래스는 레이어 객체와 함께 연결된 자체 모델 변수를 생성하고 포함합니다. 레이어 변수를 공유하기 위해서는 그 객체 자체를 공유합니다.


## 즉시 학습 (Eager training)

### 경사 계산하기 (Computing gradients)

[자동 미분법 (Automatic Differentiation)](https://en.wikipedia.org/wiki/Automatic_differentiation)은 다양한 기계 학습 알고리즘을 구현할 때 아주 유용합니다. (예, 신경망 학습을 위한 [역전파 (backpropagation)](https://en.wikipedia.org/wiki/Backpropagation)). 즉시 실행 (eager execution)이 수행되는 동안, `tfe.GradientTape`를 이용하여 나중에 경사도 계산을 수행할 연산을 추적할 수 있습니다.

`tfe.GradientTape`는 추적하지 않고 최상의 성능을 제공하기 위한 기본 기능입니다. 다른 연산들이 서로 다른 호출에 의해 발생 할 경우, 모든 전방향 연산들은 "테이프"에 기록됩니다. 경사를 계산하기 위해서 테이프를 뒤로 감아서 재생하고, 끝나면 폐기시킵니다. 하나의 `tfe.GradientTape`는 단 하나의 경사만 계산 가능하며, 그 다음 호출은 런타임 에러를 발생시킵니다.


```py
w = tfe.Variable([[1.0]])
with tfe.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, [w])
print(grad)  # => [tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)]
```
`tfe.GradientTape`를 이용하여 전방향 연산들을 기록하여, 간단한 모델을 학습하는 예제가 있습니다:

```py
#  3 * x + 2 주위의 여러 점들을 포함하는 간단한 데이터셋입니다.
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

def prediction(input, weight, bias):
  return input * weight + bias

# mean-squared error 를 이용한 loss 함수입니다.
def loss(weights, biases):
  error = prediction(training_inputs, weights, biases) - training_outputs
  return tf.reduce_mean(tf.square(error))

# loss를 weight와 bias에 대해 미분한 결과를 반환합니다.
def grad(weights, biases):
  with tfe.GradientTape() as tape:
    loss_value = loss(weights, biases)
  return tape.gradient(loss_value, [weights, biases])

train_steps = 200
learning_rate = 0.01
# W와 B를 임의의 값으로 시작합니다.
W = tfe.Variable(5.)
B = tfe.Variable(10.)

print("Initial loss: {:.3f}".format(loss(W, B)))

for i in range(train_steps):
  dW, dB = grad(W, B)
  W.assign_sub(dW * learning_rate)
  B.assign_sub(dB * learning_rate)
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))

print("Final loss: {:.3f}".format(loss(W, B)))
print("W = {}, B = {}".format(W.numpy(), B.numpy()))
```

결과 (수치는 달라질 수 있습니다):

```
Initial loss: 71.204
Loss at step 000: 68.333
Loss at step 020: 30.222
Loss at step 040: 13.691
Loss at step 060: 6.508
Loss at step 080: 3.382
Loss at step 100: 2.018
Loss at step 120: 1.422
Loss at step 140: 1.161
Loss at step 160: 1.046
Loss at step 180: 0.996
Final loss: 0.974
W = 3.01582956314, B = 2.1191945076
```

`tfe.GradientTape`를 재생하여 경사를 계산할 수 있으며, 학습 루프에서 적용 할 수 있습니다. 아래는 [mnist_eager.py](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_eager.py)에서 발췌한 예제입니다:

```py
dataset = tf.data.Dataset.from_tensor_slices((data.train.images,
                                              data.train.labels))
...
for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
  ...
  with tfe.GradientTape() as tape:
    logits = model(images, training=True)
    loss_value = loss(logits, labels)
  ...
  grads = tape.gradient(loss_value, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
```

다음 예제는 기본적인 [MNIST handwritten digits](https://www.tensorflow.org/tutorials/layers)을 분류하는 다층 모델을 만드는 것입니다. 이를 통해 즉시 실행 환경 (eager execution environment)에서 학습가능한 그래프를 만들기 위해 optimizer와 layer API를 활용해보도록 하겠습니다.


### 모델 학습하기 (Train a model)

즉시 실행 (eager execution) 모드에서는 학습과정 없이도, 모델을 호출하여 결과를 확인 할 수 있습니다:

```py
# 빈 이미지를 나타내는 텐서를 만듭니다.
batch = tf.zeros([1, 1, 784])
print(batch.shape)  # => (1, 1, 784)

result = model(batch)
# => tf.Tensor([[[ 0.  0., ..., 0.]]], shape=(1, 1, 10), dtype=float32)
```

이 예제는 [TensorFlow MNIST example](https://github.com/tensorflow/models/tree/master/official/mnist)의 [dataset.py 모듈](https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py)을 사용합니다. 이 파일을 사용자의 로컬 경로로 다운 받으세요. 아래 코드를 실행하여 MNIST 데이터 파일을 사용자 작업 경로로 다운로드 하고, 학습을 진행하기 위해 `tf.data.Dataset`로 준비하세요:

```py
import dataset  # dataset.py 파일을 다운로드 받아서 사용하세요.
dataset_train = dataset.train('./datasets').shuffle(60000).repeat(4).batch(32)
```

모델을 학습하기 위해, 최적화 시키기 위한 loss 함수를 정의하고 경사를 계산합니다. `optimizer`를 사용하여 변수들을 갱신하세요:

```py
def loss(model, x, y):
  prediction = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)

def grad(model, inputs, targets):
  with tfe.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

x, y = tfe.Iterator(dataset_train).next()
print("Initial loss: {:.3f}".format(loss(model, x, y)))

# 학습 루프
for (i, (x, y)) in enumerate(tfe.Iterator(dataset_train)):
  # 입력 함수와 인자들에 대해서 미분을 계산합니다.
  grads = grad(model, x, y)
  # 모델에 경사를 적용합니다.
  optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
  if i % 200 == 0:
    print("Loss at step {:04d}: {:.3f}".format(i, loss(model, x, y)))

print("Final loss: {:.3f}".format(loss(model, x, y)))
```

결과 (수치는 달라질 수 있습니다):

```
Initial loss: 2.674
Loss at step 0000: 2.593
Loss at step 0200: 2.143
Loss at step 0400: 2.009
Loss at step 0600: 2.103
Loss at step 0800: 1.621
Loss at step 1000: 1.695
...
Loss at step 6600: 0.602
Loss at step 6800: 0.557
Loss at step 7000: 0.499
Loss at step 7200: 0.744
Loss at step 7400: 0.681
Final loss: 0.670
```

빠른 학습을 위해, 계산을 GPU로 옮깁니다:

```py
with tf.device("/gpu:0"):
  for (i, (x, y)) in enumerate(tfe.Iterator(dataset_train)):
    # minimize()는 grad()와 apply_gradients()를 호출하여 사용하는 것과 동일합니다.
    optimizer.minimize(lambda: loss(model, x, y),
                       global_step=tf.train.get_or_create_global_step())
```

### 변수와 최적화 (Variables and optimizers)

`tfe.Variable` 객체는 변화가능한 `tf.Tensor` 값을 저장하며, 학습이 진행되는 동안 접근되며 자동으로 쉽게 미분 가능하도록 합니다. 모델의 파라미터는 변수로 클래스 내에서 캡슐화 될 수 있습니다.

더 나은 모델 파라미터의 캡슐화 방법은 `tfe.Variable`에서 `tfe.GradientTape`를 이용하는 방식입니다. 예를 들어, 위의 자동 미분의 예제는 다음과 같이 재작성될 수 있습니다:

```py
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tfe.Variable(5., name='weight')
    self.B = tfe.Variable(10., name='bias')
  def predict(self, inputs):
    return inputs * self.W + self.B

#  3 * x + 2 주위의 여러 점들을 포함하는 간단한 데이터셋입니다.
NUM_EXAMPLES = 2000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# 최적화 되어야 할 loss 함수.
def loss(model, inputs, targets):
  error = model.predict(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tfe.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])

# 다음을 정의합니다:
# 1. 모델.
# 2. 모델 파라미터에 대한 loss 함수의 미분.
# 3. 미분 결과에 따라 변수를 갱신할 방법.
model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

# 학습 루프
for i in range(300):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.B]),
                            global_step=tf.train.get_or_create_global_step())
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))
```

결과 (수치는 달라질 수 있습니다):

```
Initial loss: 69.066
Loss at step 000: 66.368
Loss at step 020: 30.107
Loss at step 040: 13.959
Loss at step 060: 6.769
Loss at step 080: 3.567
Loss at step 100: 2.141
Loss at step 120: 1.506
Loss at step 140: 1.223
Loss at step 160: 1.097
Loss at step 180: 1.041
Loss at step 200: 1.016
Loss at step 220: 1.005
Loss at step 240: 1.000
Loss at step 260: 0.998
Loss at step 280: 0.997
Final loss: 0.996
W = 2.99431324005, B = 2.02129220963
```


## 상태 확인을 위해 객체를 사용해보자 (Use objects for state during eager execution)

그래프 실행 모드에서 (변수들과 같은) 프로그램 상태는 전역적으로 수집되어 저장되었으며, 생명 주기는 `tf.Session`에 의해 관리되었습니다. 반면, 즉시 실행 (eager execution)에서는 상태를 나타내는 객체들의 생명 주기는 해당하는 파이썬 객체에 따라 결정됩니다.

### 변수는 객체 (Variables are objects)

즉시 실행 (eager execution) 모드에서는 변수는 객체의 마지막 참조가 사라지기 전까지 존재하며, 참조가 사라지고 나면 삭제됩니다.

```py
with tf.device("gpu:0"):
  v = tfe.Variable(tf.random_normal([1000, 1000]))
  v = None  # v는 더이상 GPU 메모리에 상주하지 않습니다.
```

### 객체 기반 저장 (Object-based saving)

`tfe.Checkpoint` 는 `tfe.Variable`들을 체크포인트로부터 불러오고 저장할 수 있습니다:

```py
x = tfe.Variable(10.)

checkpoint = tfe.Checkpoint(x=x)  # "x"로 저장합니다.

x.assign(2.)   # 변수에 새 값을 할당하고 저장합니다.
save_path = checkpoint.save('./ckpt/')

x.assign(11.)  # 저장한뒤 값을 변경합니다.

# 체크포인트로 부터 값을 불러옵니다.
checkpoint.restore(save_path)

print(x)  # => 2.0
```
모델을 저장하고 불러오기 위해서, 숨겨진 변수들 없이도 `tfe.Checkpoint`는 객체의 내부적인 상태를 저장합니다. `model`, `optimizer`, 전역 스텝을 기록하기 위해서, 간단히 모두 `tfe.Checkpoint`에 전달하면 됩니다:

```py
model = MyModel()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
checkpoint_dir = ‘/path/to/model_dir’
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      model=model,
                      optimizer_step=tf.train.get_or_create_global_step())

root.save(file_prefix=checkpoint_prefix)
# or
root.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

### 객체 지향 지표 (Object-oriented metrics)

`tfe.metrics`는 객체 처럼 저장됩니다. 호출하는 형태로 새 값을 지표에 전달하여 업데이트 가능하고, 결과값은 `tfe.metrics.result`를 통해 반환할 수 있습니다. 예를 들어:

```py
m = tfe.metrics.Mean("loss")
m(0)
m(5)
m.result()  # => 2.5
m([8, 9])
m.result()  # => 5.5
```


#### Summaries and TensorBoard

[TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)는 모델 학습과정을 이해하고, 디버깅하고, 최적화하기 위한 가시화 도구입니다. 프로그램이 실행되는 동안 특정 이벤트들을 요약하는데 사용되죠.

`tf.contrib.summary`는 즉시 실행 (eager execution)과 그래프 실행 (graph execution) 환경 모두 호환됩니다. `tf.contrib.summary.scalar` 같은 요약 연산은 모델 제작과정에서 삽입됩니다. 예를 들어 매 100번 전역 스텝마다 요약을 기록하고 싶다면,

```py
writer = tf.contrib.summary.create_file_writer(logdir)
global_step=tf.train.get_or_create_global_step()  # 전역 스텝 변수를 반환

writer.set_as_default()

for _ in range(iterations):
  global_step.assign_add(1)
  # record_summaries 함수를 포함시켜야 합니다.
  with tf.contrib.summary.record_summaries_every_n_global_steps(100):
    # 사용자 모델의 코드는 여기 배치됩니다.
    tf.contrib.summary.scalar('loss', loss)
     ...
```

## 자동 미분 관련 고난이도 주제들 (Advanced automatic differentiation topics)

### 동적 모델 (Dynamic models)

`tfe.GradientTape`는 동적 모델에도 사용 가능합니다. 이 예제는 [backtracking line search](https://wikipedia.org/wiki/Backtracking_line_search) 알고리즘은 일반적인 NumPy 코드처럼 보이지만, 경사가 포함되어 있고 미분가능하고 복잡한 흐름을 제어해야 합니다:

```py
def line_search_step(fn, init_x, rate=1.0):
  with tfe.GradientTape() as tape:
    # 변수들은 자동으로 기록됩니다, 하지만 수동으로 텐서를 지정하여 지켜보도록 해야 합니다.
    tape.watch(init_x)
    value = fn(init_x)
  grad, = tape.gradient(value, [init_x])
  grad_norm = tf.reduce_sum(grad * grad)
  init_value = value
  while value > init_value - rate * grad_norm:
    x = init_x - rate * grad
    value = fn(x)
    rate /= 2.0
  return x, value
```

### 경사 계산을 위한 추가 기능 (Additional functions to compute gradients)

`tfe.GradientTape`는 경사를 계산하기 위해 강력한 인터페이스 이지만, 자동 미분을 위한 [Autograd](https://github.com/HIPS/autograd)-스타일의 API도 있습니다.
이 함수는 `tfe.Variables`없이 오직 tensor와 경사 함수로만 수식을 작성할 때 유용합니다:

* `tfe.gradients_function` — 입력 함수의 인자에 대해 미분을 계산하는 함수를 반환한다. 입력 합수는 반드시 스칼라 값을 반환한다. 반환되는 함수가 호출될 때, `tf.Tensor` 객체의 리스트를 반환한다. 하나의 원소들은 입력 함수에 대한 인자를 뜻한다. 관심대상이 무엇이던 모두 함수 파라미터로 전달되어야 하며, 만약 많은 학습 가능한 파라미터들에 의존성이 있다면 굉장히 다루기 까다로워진다.
* `tfe.value_and_gradients_function` — `tfe.gradients_function`과 유사하지만, 반환되는 함수가 호출될 때, 리스트에 존재하는 입력함수의 미분과 더불어 입력함수의 값도 반환한다.
*
* Similar to
  `tfe.gradients_function`, but when the returned function is invoked, it
  returns the value from the input function in addition to the list of
  derivatives of the input function with respect to its arguments.

아래 예제에서, `tfe.gradients_function`은 `square` 함수를 인자로 받고, `square`의 입력에 대한 편미분을 계산하는 함수를 반환합니다. `square`이 `3`일때 미분을 계산하면,  `grad(3.0)` 는 `6`을 반환합니다.


```py
def square(x):
  return tf.multiply(x, x)

grad = tfe.gradients_function(square)

square(3.)  # => 9.0
grad(3.)    # => [6.0]

# square의 2차 미분:
gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
gradgrad(3.)  # => [2.0]

# 3차 미분은 None:
gradgradgrad = tfe.gradients_function(lambda x: gradgrad(x)[0])
gradgradgrad(3.)  # => [None]


# 흐름 제어를 곁들여보면:
def abs(x):
  return x if x > 0. else -x

grad = tfe.gradients_function(abs)

grad(3.)   # => [1.0]
grad(-3.)  # => [-1.0]
```

### 맞춤형 경사 (Custom gradients)

맞춤형 경사 (custom gradients)는 즉시 실행 (eager execution)과 그래프 실행 (graph execution)에서 경사를 오버라이드 (override)하는 가장 쉬운 방법입니다. 전방 함수에서 입력, 출력, 중간 결과에 대한 경사를 정의합니다. 예를 들어 역방향 패스에 대해 clip_gradient_by_norm을 구현하기 가장 쉬운 방법은 다음과 같습니다:

```py
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
  y = tf.identity(x)
  def grad_fn(dresult):
    return [tf.clip_by_norm(dresult, norm), None]
  return y, grad_fn
```

맞춤형 경사는 보통 연속적인 연산들의 수치적으로 안정된 경사를 구하기 위해 사용됩니다:

```py
def log1pexp(x):
  return tf.log(1 + tf.exp(x))
grad_log1pexp = tfe.gradients_function(log1pexp)

# x = 0 일때 경사 계산은 잘 수행된다.
grad_log1pexp(0.)  # => [0.5]

# 하지만, x = 100일때 수치적으로 불안정해지므로, 계산이 수행되지 않는다.
grad_log1pexp(100.)  # => [nan]
```
여기서, `log1pexp` 함수는 분석적으로 맞춤형 경사로 단순화 시킬 수 있습니다. 아래 구현은 전방향으로 계산될 때 불필요한 계산을 제거함으로서 더 효율적으로 변경된 `tf.exp(x)`의 값을 재사용합니다:

```py
@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tf.log(1 + e), grad

grad_log1pexp = tfe.gradients_function(log1pexp)

# 이전 예제와 마찬가지로, x = 0 일때 계산이 잘 수행됩니다.
grad_log1pexp(0.)  # => [0.5]

# 이제 x = 100 일때도 계산이 잘 수행됩니다.
grad_log1pexp(100.)  # => [1.0]
```

## 성능 (Performance)

즉시 실행 (eager execution) 동안, 계산과정은 자동적으로 GPU와 분리됩니다. 만약 사용자가 `tf.device('/gpu:0')` 나 (동일한 CPU 환경)에 포함시켜 동작하길 원한다면:

```py
import time

def measure(x, steps):
  # TensorFlow 최초 사용될때 GPU를 초기화 하므로, 시간 측정에서 제외시킵니다.
  tf.matmul(x, x)
  start = time.time()
  for i in range(steps):
    x = tf.matmul(x, x)
    _ = x.numpy()  # 연산만 수행 하도록 하고, 결과를 받아오는건 포함시키지 않도록 합니다.
  end = time.time()
  return end - start

shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

# CPU에서 실행하기:
with tf.device("/cpu:0"):
  print("CPU: {} secs".format(measure(tf.random_normal(shape), steps)))

# 가능하다면, GPU에서 실행하기:
if tfe.num_gpus() > 0:
  with tf.device("/gpu:0"):
    print("GPU: {} secs".format(measure(tf.random_normal(shape), steps)))
else:
  print("GPU: not found")
```

결과 (수치는 달라질 수 있습니다):

```
Time to multiply a (1000, 1000) matrix by itself 200 times:
CPU: 4.614904403686523 secs
GPU: 0.5581181049346924 secs
```

`tf.Tensor` 객체는 다른 장비에 복사되어 해당 연산을 실행하도록 할 수 있습니다:

```py
x = tf.random_normal([10, 10])

x_gpu0 = x.gpu()
x_cpu = x.cpu()

_ = tf.matmul(x_cpu, x_cpu)    # CPU 실행
_ = tf.matmul(x_gpu0, x_gpu0)  # GPU:0 실행

if tfe.num_gpus() > 1:
  x_gpu1 = x.gpu(1)
  _ = tf.matmul(x_gpu1, x_gpu1)  # GPU:1 실행
```

### 벤치마크 (Benchmarks)

[ResNet50](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/resnet50)과 같이 무거운 모델을 계산할 경우, GPU 상에서 학습할 때, 즉시 실행 (eager execution)의 성능은 그래프 실행 (graph execution)과 비슷합니다. 하지만 적은 계산을 포함한 모델에 대해서는 그 격차가 커집니다. 그리고 수 많은 작은 연산들로 구성된 모델의 hot code path를 최적화 하는 일이 이루어져야 합니다.


## 그래프와 함께 하기 (Work with graphs)

즉시 실행 (eager execution)이 개발과 디버깅에 있어서 사용자에게 더 많은 상호작용을 가능하게 한다면, 그래프 실행 (graph execution)은 분산 학습, 성능 최적화, 상업화를 고려한 모델 배포 등의 장점을 지닙니다. 하지만 그래프 코드를 작성하는 것은 일반적인 파이썬 코드를 작성하는 것보다 훨씬 어렵게 느껴지고, 디버그 하기도 더욱 어렵습니다.

그래프로 제작된 모델을 만들고, 학습시키기 위해, 파이썬 프로그램은 계산 과정을 그래프 표현으로 생성합니다. 그 다음 `Session.run`을 호출하여 그래프를 전달하고 C++기반 런타임에서 실행하도록 합니다. 이 과정을 통해서:

* 정적 autodiff를 이용하여 자동 미분 가능.
* 플랫폼과 독립적인 서버에 간단히 배포 가능.
* 그래프 기반 최적화 도구들 (common subexpression elimination, constant-folding)
* Compilation and kernel fusion.
* 자동 분산화 및 모델 복제 가능 (그래프의 노드를 분산 시스템에 나눠서 할당 할 수 있습니다)

즉시 실행 (eager execution)으로 작성된 모델을 배포하는 것은 상대적으로 더 어렵습니다: 모델로 부터 그래프를 생성하거나, 파이썬 런타임이 실행되고 코드가 직접 서버에 상주하는 것입니다.

### 호환성 있는 코드 작성 (Write compatible code)

즉시 실행 (eager execution)을 위해 같은 코드가 작성되었다면, 그래프 실행 동안 그래프가 생성됩니다. 단순히 동일한 코드를 즉시 실행 (eager execution)을 활성화 시키지 않은 새로운 파이썬 세션에서 실행시키기만 하면 됩니다.

대부분의 TensorFlow 연산들은 즉시 실행 (eager execution)에 대해 동작하지만, 아래 사항들을 명심하길 바랍니다:
* 입력 처리를 위해 queue 대신에 `tf.data`를 사용하세요. 더 빠르고 쉽습니다.
* `tf.keras.layers` 과 `tf.keras.Model` 같은 객체 지향 레이어 API를 사용하세요.—이들은 명시적으로 변수들로 저장됩니다.
* 많은 모델 코드들은 즉시 실행 모드와 그래프 실행 모두 같이 동작합니다. 다만 예외사항은 존재합니다. (예를 들어, 입력에 따라 계산이 변경되는 파이썬 흐름 제어를 사용하는 동적 모델의 경우.)
* 한번 `tf.enable_eager_execution`를 이용하여 즉시 실행 (eager execution)이 활성화 되었다면, 끄는 방법이 없습니다. 새로운 파이썬 세션을 시작하면 그래프 실행 모드로 돌아오게 됩니다.

즉시 실행 (eager execution)과 그래프 실행 (graph execution) 모두를 위한 코드를 작성하는게 최고의 방법입니다. 이 가이드를 통해 즉시 실행의 상호작용을 통한 과정과 디버그 가능성 뿐만 아니라, 그래프 실행을 통한 분산 성능을 얻을 수 있을 것이다.

즉시 실행 (eager execution)을 통해 코드를 작성하고 디버그 하고 반복적으로 수행해보고, 모델 그래프를 불러와서 제품 레벨에서 배포하도록 하세요. `tfe.Checkpoint`를 사용하여 모델 변수들을 저장하고 불러올 수 있습니다. 이를 통해 즉시 실행 (eager execution) 환경과 그래프 실행 (graph execution) 환경을 넘나들 수 있습니다.

아래 예제들을 참고하세요:
[tensorflow/contrib/eager/python/examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples).

### 그래프 환경에서 즉시 실행 사용하기 (Use eager execution in a graph environment)

TensorFlow 그래프 환경에서 `tfe.py_func`를 이용하여 선택적으로 즉시 실행 (eager execuution)을 활성화 시키세요. 하지만 `tf.enable_eager_execution()`이 호출되지 *않았을 때만* 사용가능 합니다.

```py
def my_py_func(x):
  x = tf.matmul(x, x)  # tf 연산을 사용할 수 있습니다.
  print(x)  # 하지만 즉시 실행 모드로도 동작하죠!
  return x

with tf.Session() as sess:
  x = tf.placeholder(dtype=tf.float32)
  # Call eager function in graph!
  pf = tfe.py_func(my_py_func, [x], tf.float32)
  sess.run(pf, feed_dict={x: [[2.0]]})  # [[4.0]]
```
