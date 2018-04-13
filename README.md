# 즉시 실행 (Eager Execution)

즉시 실행 (Eager execution)은 ([NumPy](http://www.numpy.org)와 유사한) 명령형 프로그래밍 스타일을 텐서플로우에서 제공합니다.
사용자가 즉시 실행을 활성화 시킨다면, 텐서플로우 연산이 즉시 실행됩니다.
사용자는 미리 만들어진 그래프를 [`Session.run()`](https://www.tensorflow.org/api_docs/python/tf/Session)
으로 실행할 필요가 없습니다.

예를 들면, 텐서플로우에서 간단한 계산을 수행한다고 생각해봅시다:

```python
x = tf.placeholder(tf.float32, shape=[1, 1])
m = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(m, feed_dict={x: [[2.]]}))

# Will print [[4.]]
```

즉시 실행 (Eager execution)은 이 작업을 단순하게 할 수 있습니다:

```python
x = [[2.]]
m = tf.matmul(x, x)

print(m)
```

## 주의사항

본 기능은 초기 단계에 있으며, 분산 및 다수 GPU 학습과 성능에 대한 원활한 지원을 위해 개선사항이 남아 있습니다.

- [알려진 이슈 사항](https://github.com/tensorflow/tensorflow/issues?q=is%3Aissue%20is%3Aopen%20label%3Acomp%3Aeager)
- 피드백을 기다립니다, [문제 제기](https://github.com/tensorflow/tensorflow/issues/new)를 통해 피드백 주세요.

## 설치

즉시 실행 (Eager execution)은 텐서플로우 1.7 버전 이상에 포함되어 있습니다.
설치 방법은 [공식 홈페이지 설치문서](https://www.tensorflow.org/install/)를 참고하세요.

## 문서화

텐서플로우의 즉시 실행 (eager execution)을 설명하기 위해 작성된 문서가 있습니다. 다음을 참고하세요:

- [한글 사용자 가이드 (User Guides for Korean)](./guide.md) ([source](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/programmers_guide/eager.md))
- 노트북: [기본 사용법 (Basic Usages)](./examples/notebooks/1_basics.ipynb)
- 노트북: [경사도 (Gradients)](./examples/notebooks/2_gradients.ipynb)
- 노트북: [데이터 불러오기 (Importing Data)](./examples/notebooks/3_datasets.ipynb)

## 변화사항

- 2017/10/31: 초기 preview 버전 출시. (in TensorFlow 1.5)
- 2017/12/01: 동적 신경망의 예제:
  [SPINN: Stack-augmented Parser-Interpreter Neural Network](https://arxiv.org/abs/1603.06021).
  자세한 내용은 다음 문서를 참고하세요: [README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/spinn/README.md)
- 2017/03: TensorFlow 1.7 버전에서 핵심 기능들이 실험적인 tf.contrib 네임스페이스에서 벗어났습니다.