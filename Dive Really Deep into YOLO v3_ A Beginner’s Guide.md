https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e

Foreword
  When a self-driving car runs on a road, how does it know where are other vehicles in the camera image?
    : 자율주행 자동차가 길 위를 달릴때 카메라 이미지에서 다른 자동차들의 위치를 어떻게 알까?
  When an AI radiologist reading an X-ray, how does it know where the lesion (abnormal tissue) is?
    : AI 방사선 전문의가 X-ray를 읽을때 비정상 조직이 어디에 있는지 어떻게 알까?
  Today, I will walk through this fascinating algorithm, which can identify the category of the given image, and also locate the region of interest.
    : 오늘, 주어진 이미지의 범주를 식별하고 관심 영역을 찾을 수 있는 매혹적인 알고리즘을 살펴보겠다.
  There’s plenty of algorithms introduced in recent years to address object detection in a deep learning approach, such as R-CNN, Faster-RCNN, and Single Shot Detector.
    : 최근 몇 년 동안 R-CNN, Faster-RCNN 및 Single Shot Detector와 같은 딥러닝 접근 방식에서 객체 감지를 해결하기 위해 도입된 많은 알고리즘이 있다.
  Among those, I’m most interested in a model called YOLO — You Only Look Once.
    : 그 중에서 YOLO라는 모델에 가장 관심을 갖고있다.
  This model attracts me so much, not only because of its funny name, but also some practical design that truly makes sense for me.
    : 이 모델은 재미있는 이름을 갖고있을 뿐만 아니라 실제로 나에게 맞는 실용적인 디자인으로 인해 많은 관심을 끌었다.
  In 2018, this latest V3 of this model had been released, and it achieved many new State of the Art performance.
    : 2018년에 이 모델의 최신 V3이 출시되었으며 많은 새로운 최신 성능을 달성했다.
  Because I’ve programmed some GANs and image classification networks before, and also Joseph Redmon described it in the paper in a really easy-going way, I thought this detector would just be another stack of CNN and FC layers that just works well magically.
    : 이전에 일부 GAN 및 이미지 분류 네트워크를 프로그래밍했으며 Joseph Redmon도 논문에서이를 쉽게 설명하는 방식으로 설명했기 때문에이 검출기는 마술처럼 잘 작동하는 CNN 및 FC 레이어의 또 다른 스택 일 것이라고 생각했습니다.
  But I was wrong.
    : 그러나 나는 틀렸다.
  Perhaps it’s because I’m just dumber than usual engineers, I found it really difficult for me to translate this model from the paper to actual code.
    : 아마도 나는 일반적인 엔지니어보다 어리석기 때문에 이 모델을 논문에서 실제 코드로 변환하기가 정말 어렵다는 것을 알았다.
  And even when I managed to do that in a couple of weeks (I gave up once put it away for a few weeks), I found it even more difficult for me to make it work.
    : 그리고 몇 주 안에 그렇게 할 때조차도 (몇 주 동안 한 번 버려 두었습니다), 나는 그것이 작동하는 것이 훨씬 어렵다는 것을 알았습니다.
  There’re so quite a few blogs, GitHub repos about YOLO V3, but most of them just gave a very high-level overview of the architecture, and somehow they just succeed.
    : YOLO V3를 다루는 블로그들과 깃허브들이 있지만, 대부분은 아키텍처에 대한 매우 높은 수준의 개요를 제공했으며 어떻게든 성공했다.
  Even worse, the paper itself is too chill that it fails to provide many crucial details of implementation, and I have to read the author’s original C implementation (when is the last time did I write C? Maybe at college?) to confirm some of my guesses.
    : 더 나쁜것은, 논문 자체가 냉정해서 구현에 대한 많은 중요한 세부 사항을 제공하지 못하기 때문에 필자의 추축을 확인하기 위해서 저자의 원래 C 구현을 읽어야한다.
  When there’s a bug, I usually have no idea why it would occur. Then I end up manually debugging it step by step and calculating those formulas with my little calculator.
    : 버그가 있을때는 보통 왜 그런지 알 수 없다. 그 다음 단계별 수동으로 디버깅하고 작은 계산기로 수식을 계산했다.

  Fortunately, I didn’t give up this time and finally made it work.
    : 다행히도, 이번엔 포기하지 않았고 마침내 완성했다.
  But in the meantime, I also felt really strongly that there should be a more thorough guide out there on the internet to help dumb people like me to understand every detail of this system.
    : 그러나 그동안 저와 같은 어리석은 사람들이 이 시스템의 모든 세부 사항을 이해하도록 돕기 위해 인터넷에 더 철저한 가이드가 있어야 한다고 강력하게 느꼇다.
  After all, if one single detail is wrong, the whole system would go south quickly.
    : 결국 하나의 세부 사항이 잘못되면 전체 시스템이 빠르게 남족으로 이동한다.
  And I’m sure that if I don’t write these down, I would forget all these in few weeks too. So, here I am, presenting you this “Dive Really Deep into YOLO V3: A Beginner’s Guide”.
    : 나는 이 글을 쓰지 않으면 몇 주 안에 이 모든것을 잊어버릴 것이다. 자 이제 여기서 여러분께 'Dive Really Deep into YOLO V3: A Beginner's Guide'를 소개한다.

  I hope you’ll like it.
    : 당신이 좋아하길 바란다.

Prerequisite
  Before getting into the network itself, I’ll need to clarify with some prerequisites first. As a reader, you are expected to:
    : 네트워크 자체에 들어가기 전에 먼저 몇 가지 전제 조건을 명확히해야한다. 독자는 다음을 수행해야한다.
  1. Understand the basics of Convolutional Neural Network and Deep Learning
    : Convolutional Neural Network와 Deep Learning의 기초 이해
  2. Understand the idea of object detection task
    : 객체 탐지 과제의 이해
  3. Have curiosity about how the algorithm works internally
    : 알고리즘이 내부적으로 작동하는 방식에 대한 호기심

YOLO V3
  YOLO V3 is an improvement over previous YOLO detection networks.
    : YOLO V3는 이전 YOLO에 비해서 개선되었다.
  Compared to prior versions, it features multi-scale detection, stronger feature extractor network, and some changes in the loss function.
    : 이전 버전과의 차이점은 'features multi-scale detection', 'stronger feature extractor network', 'some changes in the loss function'이다.
  As a result, this network can now detect many more targets from big to small.
    : 결과적으로 이 네트워크는 이제 크고 작은 대상을 더 많이 감지 할 수 있다.
  And, of course, just like other single-shot detectors, YOLO V3 also runs quite fast and makes real-time inference possible on GPU devices.
    : 물론 다른 single-shot detector와 마찬가지로, YOLO V3도 매우 빠르게 실행되며 GPU 장치에서 실시간 추론이 가능하다.
  Well, as a beginner to object detection, you might not have a clear image of what do they mean here.
    : 아마 객체 탐지의 초보자로서 이 변화들이 무엇을 의미하는지 명확하게 이해하지 못했을 수도 있다.
  But you will gradually understand them later in my post.
    : 하지만, 이 게시물에서 점차적으로 이해할 것이다.
  For now, just remember that YOLO V3 is one of the best models in terms of real-time object detection as of 2019.
    : 지금은 YOLO V3가 2019년 현재 실시간 객테 탐지 측면에서 최고의 모델 중 하나라는 것만 기억해라

  **Network Architecture**
  Images -> Feature Extractor(Backbone) ---Multi-scale features---> Detector -> Bounding Boxes, Classes
  [](https://miro.medium.com/max/1400/1*hULeMGlxnjjvuf7Fx7kuaA.jpeg)

  First of all, let’s talk about how this network look like at a high-level diagram(Although, the network architecture is the least time-consuming part of implementation).
    : 우선 이 네트워크가 상위 수준 다이어그램에서 어떻게 보이는지 이야기하겠다(그러나 네트워크 아키텍처는 구현에서 가장 시간이 많이 걸리지 않는다.).
  The whole system can be divided into two major components: Feature Extractor and Detector; both are multi-scale.
    : 전체 시스템은 두 개의 주요 구성으로 나눌 수 있다: 'Feature Extractor와 Detector'; 둘 다 multi-scale이다.
  When a new image comes in, it goes through the feature extractor first so that we can obtain feature embeddings at three (or more) different scales.
    : 새로운 이미지가 들어오면, 먼저 'Feature Extractor'를 거쳐 3개 이상의 다른 스케일로 feature map을 얻을 수 있다.
  Then, these features are feed into three (or more) branches of the detector to get bounding boxes and class information.
    : 그 다음 이 feature map들은 'Detector'의 3개 이상의 분기로 입력되 bounding boxes와 class information을 얻는다.

  **DarkNet-53**
  The feature extractor YOLO V3 uses is called Darknet-53.
    : YOLO V3에서 사용하는 'Feature Extractor'는 Darknet-53이라고 불린다.
  You might be familiar with the previous Darknet version from YOLO V1, where there’re only 19 layers.
    : (의역)YOLO V1에서는 19개의 레이어만 있는 DarkNet을 사용했다.
  But that was like a few years ago, and the image classification network has progressed a lot from merely deep stacks of layers.
    : (구글 번역)그러나 그것은 몇 년 전과 같았으며 이미지 분류 네트워크는 단지 깊게 쌓인 레이어에서 많은 발전을 거듭했습니다.
  ResNet brought the idea of skip connections to help the activations to propagate through deeper layers without gradient diminishing.
    : ResNet은 그래디언트 감소없이 활성화가 더 깊은 레이어를 통해 전파 될 수 있도록 연결 건너 뛰기 아이디어를 가져 왔습니다.
  Darknet-53 borrows this idea and successfully extends the network from 19 to 53 layers, as we can see from the following diagram.
    : 다음 다이어그램에서 볼 수 있듯이 Darknet-53은이 아이디어를 빌려 19에서 53 개의 계층으로 네트워크를 성공적으로 확장합니다.
  [](https://miro.medium.com/max/1206/1*a_hU7H_Av2YiUEFaui635w.jpeg)

  This is very easy to understand.
    : 이 네트워크(Darknet-53)을 이해하는 것은 매우 쉽다.
  Consider layers in each rectangle as a residual block.
    : 각 사각형의 레이어(그림에)를 residual block이라고 고려해라.
  The whole network is a chain of multiple blocks with some strides 2 Conv layers in between to reduce dimension.
    : 전체 네트워크는 중간중간 차원을 줄여주기 위한 strides가 2인 Conv Layer가 있는 multiple block들의 체인이다.
  Inside the block, there’s just a bottleneck structure (1x1 followed by 3x3) plus a skip connection.
    : 블록 안에는 병목 구조(1x1 followed by 3x3)와 skip connection이 있다.
  If the goal is to do multi-class classification as ImageNet does, an average pooling and a 1000 ways fully connected layers plus softmax activation will be added.
    : 목표가 ImageNet처럼 멀티 클래스 분류를 수행하는 것이면, 'Feature Extractor'뒤에 average pooling과 softmax activation을 갖는 1000 fc layer가 추가된다.
  However, in the case of object detection, we won’t include this classification head.
    : 하지만, 객체 탐지의 경우 'Feature Extractor'뒤에 classification head를 포함하지 않는다.
  Instead, we are going to append a “detection” head to this feature extractor.
    : 그 대신에, 'Feature Extractor'뒤에 'Detection' head를 추가한다.
  And since YOLO V3 is designed to be a multi-scaled detector, we also need features from multiple scales.
    : 또한 YOLO V3는 multi-scaled detector로 설계되었으므로, multiple scales features도 필요하다.
  Therefore, features from last three residual blocks are all used in the later detection.
    : 따라서 마지막 3개의 residual block의 feature map들 모두 'Detection'에 사용된다.
  In the diagram below, I’m assuming the input is 416x416, so three scale vectors would be 52x52, 26x26, and 13x13. Please note that if the input size is different, the output size will differ too.
    : 아래 다이어그램(그림)에서 입력이 416x416이라고 가정하면 3개의 스케일 벡터는 52x52, 26x26, 13x13이된다.
    : 입력 크기가 다르면 출력 크기도 달라진다.
  [](https://miro.medium.com/max/1400/1*SzyNALdsE9pDCpCvtqH7ZQ.jpeg)

  **Multi-scale Detector**
  Once we have three features vectors, we can now feed them into the detector.
    : 3 개의 특징 벡터(52x52, 26x26, 13x13)를 얻으면 이제 'Detector'에 입력할 수 있다.
  But how should we structure this detector?
    : 하지만 이 검출기를 어떻게 구성해야하지?
  Unfortunately, the author didn’t bother to explain this part this his paper.
    : 불행하게도, 저자는 이 부분을 그의 논문에서 설명하지 않았다.
  But we could still take a look at the source code he published on Github.
    : 그러나 저자가 공개한 Github Code를 참고할 수 있다.
  Through this config file, multiple 1x1 and 3x3 Conv layers are used before a final 1x1 Conv layer to form the final output.
    : 이 config file을 통해 최종 출력 형태를 위한 마지막 1x1 Conv layer 전에 multiple 1x1 and 3x3 Conv layers가 사용되었다.
  For medium and small scale, it also concatenates features from the previous scale.
    : medium과 small scale의 경우 이전 스케일과 concatenates를 수행한다.
  By doing so, small scale detection can also benefit from the result of large scale detection.
    : 이렇게하면 small scale detection에서 large scale detection 결과의 이점을 얻을 수 있다.

  [1](https://miro.medium.com/max/1400/1*kVFx54oUhBWzUdHzbFDsiw.jpeg)

  Assuming the input image is (416, 416, 3), the final output of the detectors will be in shape of [(52, 52, 3, (4 + 1 + num_classes)), (26, 26, 3, (4 + 1 + num_classes)), (13, 13, 3, (4 + 1 + num_classes))].
    : 입력 이미지가 (416, 416, 3)이라고 가정했을때, 'Detector'의 최종 출력은 [(52, 52, 3, (4 + 1 + num_classes)), (26, 26, 3, (4 + 1 + num_classes)), (13, 13, 3, (4 + 1 + num_classes))]이다.
  The three items in the list represent detections for three scales.
    : 리시트에 3 가지 items는 3 가지 스케일에 대한 detection을 나타낸다.
  But what do the cells in this 52x52x3x(4+1+num_classes) matrix mean? Good questions.
    : 그러나 이 (52x52x3x(4+1+num_classes)) matrix가 무엇을 의미하는지?? 좋은 질문이다.
  This brings us to the most important notion in pre-2019 object detection algorithm: anchor box (prior box).
    : 이것은 2019 이전의 객체 탐지 알고리즘에서 가장 중요한 개념인 anchor box(prior box)를 불러온다.

  **Anchor Box**
  The goal of object detection is to get a bounding box and its class.
    : 객체 탐지의 목표는 bounding box와 class를 얻는 것이다.
  Bounding box usually represents in a normalized xmin, ymin, xmax, ymax format.
    : Bounding box는 정규화된 xmin, ymin, xmax, ymax 형식으로 나타난다.
  For example, 0.5 xmin and 0.5 ymin mean the top left corner of the box is in the middle of the image.
    : 예를들어, 0.5 xmin과 0.5 ymin은 왼쪽 상단 모서리가 이미지의 중앙에 위치한다는 것을 의미한다.
  Intuitively, if we want to get a numeric value like 0.5, we are facing a regression problem.
    : 직관적으로 0.5와 같은 숫자 값을 얻으려면 회귀 문제가 발생한다.
  We may as well just have the network predict for values and use Mean Square Error to compare with the ground truth.
    : 네트워크의 예측 값을 얻고 Mean Square Error(MSE)를 사용해 예측 값과 gt(ground truth)를 비교할 수 있다.
  However, due to the large variance of scale and aspect ratio of boxes, researchers found that it’s really hard for the network to converge if we just use this “brute force” way to get a bounding box.
    : 그러나 box의 규모와 종횡비가 다양하기 때문에, 연구원들은 bounding box를 얻기위해 'brute force' 방식을 사용하면 네트워크가 수렴하기 어렵다는 사실을 발견했다.
  Hence, in Faster-RCNN paper, the idea of an anchor box is proposed.
    : 따라서 Faster-RCNN 논문에서 앵커 박스에 대한 아이디어가 제안됐다.

  Anchor box is a prior box that could have different pre-defined aspect ratios.
    : Anchor box는 다른 미리 정의된 종횡비를 가질 수 있는 prior box이다.
  These aspect ratios are determined before training by running K-means on the entire dataset.
    : 이러한 종횡비는 전체 데이터 세트에서 K-mean을 실행하여 훈련 전에 결정된다.
  But where does the box anchor to? We need to introduce a new notion called the grid.
    : 우리는 grid라고 불리는 새로운 개념을 도입 할 필요가 있다.
  In the “ancient” year of 2013, algorithms detect objects by using a window to slide through the entire image and running image classification on each window.
    : 2013년의 "고전적" 년도에 알고리즘은 window를 사용해 전체 이미지를 sliding하고 각 window에서 이미지 분류를 실행함으로써 사물을 탐지했다.
  However, this is so inefficient that researchers proposed to use Conv net to calculate the whole image all in once (technically, only when your run convolution kernels in parallel.)
    : 그러나, 이것은 매우 비효율적이어서, 연구자들은 Conv net을 사용하여 전체 이미지를 한 번에 계산할 것을 제안했다. (기술적으로, 당신의 런 convolution 커널이 병렬로 있을 때만)
  Since the convolution outputs a square matrix of feature values (like 13x13, 26x26, and 52x52 in YOLO), we define this matrix as a “grid” and assign anchor boxes to each cell of the grid.
    : convolution은 feature value의 제곱 행렬(YOLO에서 13x13, 26x26 및 52x52와 같은)을 출력하므로, 우리는 이 행렬을 'grid'로 정의하고 grid의 각 셀에 앵커 박스를 할당한다.
  In other words, anchor boxes anchor to the grid cells, and they share the same centroid.
    : 즉, anchor boxes는 grid cell에 고정되며, 동일한 중심을 공유한다.
  And once we defined those anchors, we can determine how much does the ground truth box overlap with the anchor box and pick the one with the best IOU and couple them together.
    : 그리고 일단 anchor들을 정희하고 나면, 우리는 ground truth box가 anchor box와 얼마나 겹치는지 알아낼 수 있고, 최고의 IOU를 가진 것을 골라 그 둘을 결합시킬 수 있다.
  I guess you can also claim that the ground truth box anchors to this anchor box.
    : ground truth box가 이 anchor box에 고정되어 있다고 주정할 수도 있을것 같다.
  In our later training, instead of predicting coordinates from the wild west, we can now predict offsets to these bounding boxes.
    : 이후 훈련에서는, 야생 서부에서 좌표를 예측하는 대신에, 이제 우리는 이 경계 상자들에 대한 상쇄를 예측할 수 있다.
  This works because our ground truth box should look like the anchor box we pick, and only subtle adjustment is needed, which gives us a great head start in training.
    : 이는 우리의 ground truth box가 우리가 선택한 anchor box처럼 보여야 하고, 미묘한 조정만 필요하기 때문에 훈련에서 유리한 출발을 할 수 있게 해준다.

  [](https://miro.medium.com/max/1400/1*yQFsd4vMDWVWikme3-Mb8g.jpeg)

  In YOLO v3, we have three anchor boxes per grid cell.
    : YOLO v3에서는 grid ceel당 세 개의 ancho boxes가 있다.
  And we have three scales of grids.
    : 그리고 세 가지 scale의 gride가 있다.
  Therefore, we will have 52x52x3, 26x26x3 and 13x13x3 anchor boxes for each scale.
    : 따라서 우리는 규모별로 52x52x3, 26x26x3, 13x13x3 anchor boxes를 갖추게 될 것이다.
  For each anchor box, we need to predict 3 things:
    : 각 anchor box에 대해 다음 3가지를 예측해야 한다.
    1. The location offset against the anchor box: tx, ty, tw, th. This has 4 values.
      : anchor box에 대한 위치 offset: tx, ty, tw, th. 4개의 값을 가지고 있다.
    2. The objectness score to indicate if this box contains an object. This has 1 value.
      : 이 box에 객체가 포함되어 있는지 여부를 나타내는 objectness score. 1개의 값을 가지고 있다.
    3. The class probabilities to tell us which class this box belongs to. This has num_classes values.
      : 이 box가 어느 클래스에 속하는지 알려주는 클래스 확률. class의 수 만큼 값을 가지고 있다.
  In total, we are predicting 4 + 1 + num_classes values for one anchor box, and that’s why our network outputs a matrix in shape of 52x52x3x(4+1+num_classes) as I mentioned before. tx, ty, tw, th isn’t the real coordinates of the bounding box.
    : 총 4 + 1 + num_classes 값을 하나의 anchor box에 대해 예측하고 있으며, 그래서 우리 네트워크는 앞에서 말한 것처럼 52x52x3x(4+1+num_classes) 모양의 매트릭스를 출력하는 것이다. tx, ty, tw, th는 bounding box의 실제 좌표가 아니다.
  It’s just the relative offsets compared with a particular anchor box.
    : 특정 anchor box와 다르게 상대적인 offsets일 뿐이다.
  I’ll explain these three predictions more in the Loss Function section after.
    : 이 세 가지 예측에 대해서는 다음 Loss Function 섹션에서 더 설명한다.

  Anchor box not only makes the detector implementation much harder and much error-prone, but also introduced an extra step before training if you want the best result.
    : Anchor box는 검출기 구현을 훨씬 더 어렵게, 오류 발생 가능성이 훨씬 높을 뿐만 아니라, 최상의 결과를 원하면 훈련 전에 추가 단계를 도입하게 했다.
  So, personally, I hate it very much and feel like this anchor box idea is more a hack than a real solution.
    : 그래서 개인적으로 나는 그것을 매우 싫어하고 이 anchor box 아이디어가 진짜 해결책이라기보다는 해킹이라고 느낀다.
  In 2018 and 2019, researchers start to question the need for anchor box.
    : 2018년과 2019년부터 연구자들은 anchor box의 필요성에 의문을 제기하기 시작한다.
  Papers like CornerNet, Object as Points, and FCOS all discussed the possibility of training an object detector from scratch without the help of an anchor box.
    : CornerNet, Object as Points, FCOS와 같은 논문들은 모두 앵커 박스의 도움 없이 처음부터 객체 감지기를 훈련시킬 수 있는 가능성에 대해 논의 하였다.

**Loss Function**
  With the final detection output, we can calculate the loss against the ground truth labels now.
    : 최종 검출 출력만 있으면 ground truth labels을 기준으로 loss를 계산할 수 있다.
  The loss function consists of four parts (or five, if you split noobj and obj):
    : 손실 함수는 4개 부분 (또는 noobj와 obj를 분할하는 경우 5개)으로 구성된다.
    1. centroid (xy) loss
    2. width and height loss
    3. objectness loss
    4. classification loss
  When putting together, the formula is like this:
    : 위 loss들을 합치면 아래와 같다
    Loss = Lambda_Coord * Sum(Mean_Square_Error((tx, ty), (tx’, ty’) * obj_mask) + Lambda_Coord * Sum(Mean_Square_Error((tw, th), (tw’, th’) * obj_mask) + Sum(Binary_Cross_Entropy(obj, obj’) * obj_mask) + Lambda_Noobj * Sum(Binary_Cross_Entropy(obj, obj’) * (1 -obj_mask) * ignore_mask) + Sum(Binary_Cross_Entropy(class, class’)

  It looks intimidating but let me break them down and explain one by one.
    : 위 식은 위협적으로 보이지만 하나씩 분해해서 설명해준다.

  **1. centroid loss**
    <xy_loss = Lambda_Coord * Sum(Mean_Square_Error((tx, ty), (tx’, ty’)) * obj_mask)>
    The first part is the loss for bounding box centroid.
      : 첫 번째 부분은 경계 상자 중심에 대한 손실이다.
    tx and ty is the relative centroid location from the ground truth.
      : tx와 ty는 ground truth에서 상대적인 중심 위치이다.
    tx' and ty' is the centroid prediction from the detector directly.
      : tx'와 ty'는 detector에서 직접적으로 발생하는 중심 예측이다.
    The smaller this loss is, the closer the centroids of prediction and ground truth are.
      : 이 손실이 작을수록 gt와 pbox의 중심은 더 가깝다.
    Since this is a regression problem, we use mean square error here.
      : 이것은 회귀 문제이기 때문에 여기서는 평균 제곱 오차를 사용한다.
    Besides, if there’s no object from the ground truth for certain cells, we don’t need to include the loss of that cell into the final loss.
      : 게다가, 특정 셀에 ground truth에 나온 물체가 없다면, 우리는 그 셀의 손실을 최종 손실에 포함시킬 필요가 없다.
    Therefore we also multiple by obj_mask here. obj_mask is either 1 or 0, which indicates if there’s an object or not.
      : 따라서 우리는 또한 obj_mask를 곱한다. obj_mask는 1 또는 0으로, 물체가 있는지 없는지를 나타낸다.
    In fact, we could just use obj as obj_mask, obj is the objectness score that I will cover later.
      : 사실, obj을 obj_mask로 사용하면 되는데, obj은 이후에 다룰 객관성 점수이다.
    One thing to note is that we need to do some calculation on ground truth to get this tx and ty.
      : 한 가지 주목할 점은 이 tx와 ty를 얻기 위해서는 ground truth에 대한 계산을 좀 해야 한다는 것이다.
    So, let’s see how to get this value first. As the author says in the paper:
      : 자, 먼저 이 값을 얻는 방법을 살펴보자, 저자가 논문에서 말한 바와 같이:
        bx = sigmoid(tx) + Cx
        by = sigmoid(ty) + Cy
      Here bx and by are the absolute values that we usually use as centroid location.
        : 여기서 bx와 by는 일반적으로 중심위치로 사용하는 절대값이다.
      For example, bx = 0.5, by = 0.5 means that the centroid of this box is the center of the entire image.
        : 예를 들어, bx = 0.5, by = 0.5는 이 상자의 중심이 전체 이미지의 중심임을 의미한다.
      However, since we are going to compute centroid off the anchor, our network is actually predicting centroid relative the top-left corner of the grid cell.
        : 그러나, 우리가 anchor로부터 중심을 계산하려고 하기 때문에, 우리의 네트워크는 grid cell의 왼쪽 상단 모서리에 상대적인 중앙을 예측하고있다.
      Why grid cell? Because each anchor box is bounded to a grid cell, they share the same centroid.
        : 왜 grid cell인가? 각 anchor box는 grid cell에 경계되기 때문에 동일한 중심을 공유한다.
      So the difference to grid cell can represent the difference to anchor box.
        : 그래서 grid cell과의 차이는 anchor box에 대한 차이를 나타낼 수 있다.
      In the formula above, sigmoid(tx) and sigmoid(ty) are the centroid location relative to the grid cell.
        : 위의 공식에서 sigmoid(tx)와 sigmoid(ty)는 grid cell에 상대적인 중심 위치이다.
      For instance, sigmoid(tx) = 0.5 and sigmoid(ty) = 0.5 means the centroid is the center of the current grid cell (but not the entire image).
        : 예를들어, sigmoid(tx) = 0.5이고 sigmoid(ty) = 0.5는 중심부가 현재 grid cell의 중심(전체 이미지에서 말고)임을 의미한다.
      Cx and Cy represents the absolute location of the top-left corner of the current grid cell.
        : Cx와 Cy는 현재 grid cell의 왼쪽 상단 모서리의 절대 위치를 나타낸다.
      So if the grid cell is the one in the SECOND row and SECOND column of a grid 13x13, then Cx = 1 and Cy = 1.
        : 따라서 grid cell이 grid 13x13의 두 번째 행과 두 번째 열에 있는 cell 이라면 Cx = Cy = 1이다.
      And if we add this grid cell location with relative centroid location, we will have the absolute centroid location bx = 0.5 + 1 and by = 0.5 + 1.
        : 그리고 이 grid cell 위치를 상대 중심 위치에 추가하면 절대 중심 위치 bx = 0.5 + 1과 by = 0.5 + 1이 된다.
      Certainly, the author won’t bother to tell you that you also need to normalize this by dividing by the grid size, so the true bx would be 1.5/13 = 0.115.
        : 확실히, 저자는 grid 크기로 나누어 bx와 by를 normalization할 필요가 있다는 것을 굳이 말하지 않는다.
        : 그래서 진정한 bx = 1.5/13 = 0.115가 될 것이다.
      Ok, now that we understand the above formula, we just need to invert it so that we can get tx from bx in order to translate our original ground truth into the target label.
        : 좋아, 이제 우리는 위의 공식을 이해했으므로, 우리의 원래 gt를 표적 라벨로 번역하기 위해 bx에서 tx를 얻을 수 있도록 그것을 뒤집기만 하면 된다.
      Lastly, Lambda_Coord is the weight that Joe introduced in YOLO v1 paper.
        : 마지막으로 Lambda_Coord는 Joe가 YOLO v1에서 소개한 가중치이다.
      This is to put more emphasis on localization instead of classification. The value he suggested is 5.
        : 분류 대신 localization에 무게를 두기 위해서 사용한다. 논문에서는 5를 제시

  **2. width and height loss**
    The next one is the width and height loss. Again, the author says:
      : 다음은 폭과 높이 손실이다. 이번에도 저자는 다음과 같이 말한다:
        bw = exp(tw) * pw
        bh = exp(th) * ph
    Here bw and bh are still the absolute width and height to the whole image.
      : 여기서 bw와 bh는 여전히 전체 이미지에 대한 절대 폭과 높이이다.
    pw and ph are the width and height of the prior box (aka. anchor box, why there’re so many names).
      : pw와 ph는 anchor box의 width와 height
    We take e^(tw) here because tw could be a negative number, but width won’t be negative in real world.
      : 우리는 tw가 음수가 될 수 있기 때문에 여기서 e^(tw)를 택하지만 실제로는 폭이 음수가 되지 않을 것이다.
    So this exp() will make it positive.
      : 그래서 이 exp()은 그것을 긍정적으로 만들것이다.
    And we multiply by prior box width pw and ph because the prediction exp(tw) is based off the anchor box.
      : 그리고 예측 exp(tw)는 anchor box에 기초하기 때문에 anchor box width pw와 ph로 곱한다.
    So this multiplication gives us real width. Same thing for height.
      : 그래서 이 곱셈은 우리에게 실제 width를 준다. height도 마찬가지로
    Similarly, we can inverse the formula above to translate bw and bh to tx and th when we calculate the loss.
      : 마찬가지로, 우리는 손실을 계산할 때 bw와 bh를 tx와 th로 변환하기 위해 위의 공식을 반전시킬 수 있다.

  **3,4. objectness and non-objectness score loss**
    obj_loss = sum(Binary_Cross_Entropy(obj, obj') * obj_mask)

    noobj_loss = Lambda_Noobj * sum(Binary_Cross_Entropy(obj, obj') * (1 - obj_mask) * ignore_mask)

    Objectness indicates how likely is there an object in the current cell.
      : objectness는 현재 cell에 객체가 있을 가능성이 얼마나 있는지를 나타낸다.
    Unlike YOLO v2, we will use binary cross-entropy instead of mean square error here.
      : YOLO v2와 달리 여기서는 mse 대신 binary cross-entropy를 사용한다.
    In the ground truth, objectness is always 1 for the cell that contains an object, and 0 for the cell that doesn’t contain any object.
      : gt에서 objectness는 객체를 포함하는 cell의 경우 항상 1이고, 어떤 객체도 포함하고 있지 않는 cell의 경우 0이다.
    By measuring this obj_loss, we can gradually teach the network to detect a region of interest.
      : 이 obj_loss를 측정함으로써, 우리는 네트워크로 하여금 관심 영역을 탐지하도록 점차적으로 가르칠 수 있다.
    In the meantime, we don’t want the network to cheat by proposing objects everywhere.
      : 그러는 동안 우리는 네트워크가 사방에서 객체를 제안하는 사기를 치는 것을 원치 않는다.
    Hence, we need noobj_loss to penalize those false positive proposals.
      : 따라서, 우리는 그러한 잘못된 제안을 처벌하기 위해 noobj_loss가 필요하다.
    We get false positives by masking prediciton with 1-obj_mask.
      : 우리는 1-obj_mask로 예측을 마스킹 함으로써 잘못된 긍정을 얻는다.
    The `ignore_mask` is used to make sure we only penalize when the current box doesn’t have much overlap with the ground truth box.
      : ignore_mask는 현재의 box가 gt box와 크게 겹치지 않을 때에만 벌칙을 적용하도록 하는데 사용된다.
    If there is, we tend to be softer because it’s actually quite close to the answer.
      : 만약에 있다면, 우리는 더 부드러운 경향이 있다. 왜냐하면 그것을 실제로 해답에 꽤 가깝기 때문이다.
    As we can see from the paper, “If the bounding box prior is not the best but does overlap a ground truth object by more than some threshold we ignore the prediction.”
      : 논문에서 알 수 있듯이, "만약 이전 bounding box가 최고는 아니지만 gt 객체와 임계값 이상 겹친다면, 그 예측을 무시한다."
    Since there are way too many noobj than obj in our ground truth, we also need this Lambda_Noobj = 0.5 to make sure the network won’t be dominated by cells that don’t have objects.
      : 우리의 gt에는 obj보다 noobj이 훨씬 많기 때문에, 네트워크가 객체가 없는 cell에 의해 지배되지 않도록 하기 위해서는 Lambda_Noobj = 0.5도 필요하다.

  **5. classification loss**
    class_loss = Sum(Binary_Cross_Entropy(class, class’) * obj_mask)

    The last loss is classification loss.
      : 마지막 손실은 분류 손실이다.
    If there’re 80 classes in total, the class and class’ will be the one-hot encoding vector that has 80 values.
      : 총 80개의 클래스가 있다면 '클래스와 클래스'는 80개의 값을 가진 ont-hot encoding 벡터가 될 것이다.
    In YOLO v3, it’s changed to do multi-label classification instead of multi-class classification.
      : YOLO v3에서는 multi-class classification 대신 multi-label classification를 하는 것으로 변경되었다.
    Why? Because some dataset may contains labels that are hierarchical or related, eg woman and person.
      : 이유는 일부 데이터셋은 계층적이거나 관련이 있는 레이블(예: 여성과 사람)을 포함할 수 있기 때문이다.
    So each output cell could have more than 1 class to be true.
      : 따라서 각 출력 cell은 참이어야 할 클래스가 1개 이상일 수 있다.
    Correspondingly, we also apply binary cross-entropy for each class one by one and sum them up because they are not mutually exclusive.
      : 이에 대응하여, 우리는 또한 각 클래스마다 binary cross-entropy를 하나씩 적용하고, 그것들이 상호 배타적이지 않기 때문에 요약한다.
    And like we did to other losses, we also multiply by this obj_mask so that we only count those cells that have a ground truth object.
      : 그리고 다른 losses에서도 그랬듯이, 우리는 또한 이 obj_mask를 곱해서 gt 객체가 있는 cell에서만 적용되도록 한다.

    To fully understand how this loss works, I suggest you manually walk through them with a real network prediction and ground truth.
      : 이 손실이 어떻게 발생하는지 완전히 이해하려면 실제 네트워크 예측과 gt를 직접 살펴봐라
    Calculating the loss by your calculator (or tf.math) can really help you to catch all the nitty-gritty details.
      : 계산기(또는 tf.math)로 손실을 계산하는 것은 정말 모든 사손한 세부 사항을 파악하는데 도움이 될 수 있다.
    And I did that by myself, which helped me find lots of bugs. After all, the devil is in the detail.
      : 그리고 나 혼자 그렇게 했고, 이 과정은 맣은 bug들을 찾는데 도움을 주었다. 결국 악마는 디테일에 있었다.

**Implementation**
  If I stop writing here, my post will just be like another “YOLO v3 Review” somewhere on the web.
    : 만약 내가 여기서 쓰는 것을 그만둔다면, 내 게시물은 웹 어딘가에 있는 또 다른 'YOLO v3 Review'와 같을 것이다.
  Once you digest the general idea of YOLO v3 from the previous section, we are now ready to go explore the remaining 90% of our YOLO v3 journey: Implementation.
    : 이전 섹션에서 YOLO v3에 대한 일반적인 아이디어를 요약했다. 이제 YOLO v3 여행의 나머지 90%를 살펴보자: 구현

  **Framework**
    At the end of September, Google finally released TensorFlow 2.0.0.
      : 9월 말 구글은 마침내 tensorflow 2.0.0을 출시했다.
    This is a fascinating milestone for TF. Nevertheless, new design doesn’t necessarily mean less pain for developers.
      : 이것은 TF에게 매력적인 이정표이다. 그럼에도 불구하고, 새로운 디자인이 개발자들에게 반드시 덜 고통스러운 것을 의미하지는 않는다.
    I’ve been playing around TF 2 since very early of 2019 because I always wanted to write TensorFlow code in the way I did for PyTorch.
      : 필자는 PyTorch에게 했던 방식대로 tensorflow 코드를 늘 쓰고 싶어 2019년 초부터 TF 2를 중심으로 플레이해 왔다.
    If it’s not because of TensorFlow’s powerful production suite like TF Serving, TF lite, and TF Board, etc., I guess many developers will not choose TF for new projects.
      : TF 서빙, TF 라이트, TF 보드 등 tensorflow의 강력한 생산 제품군 때문이 아니라면 신규 프로젝트에는 TF를 선택하지 않는 개발자가 많을 것 같다.
    Hence, if you don’t have a strong demand for production deployment, I would suggest you implement YOLO v3 in PyTorch or even MXNet.
      : 따라서, 프로덕션 배포에 대한 강력한 수요가 없다면 PyTorch 또는 MXNet에서 YOLO v3를 구현하는 것이 좋다.
    However, if you made your mind to stick with TensorFlow, please continue reading.
      : 하지만, tensorflow를 고수하기로 마음먹었다면 계속 읽어라.

    TensorFlow 2 officially made eager mode a first-tier citizen.
      : Tensorflow 2는 공식적으로 열성모드를 1군 시민으로 만들었다.
    To put it simply, instead of using TensorFlow specific APIs to calculate in a graph, you can now leverage native Python code to run the graph in a dynamic mode.
      : 간단히 말해, Tensorflow 특정 API를 사용하여 그래프에서 계산하는 대신, 이제 기본 Python 코드를 활용하여 동적 모드에서 그래프를 실행할 수 있다.
    No more graph compilation and much easier debugging and control flow.
      : 더 이상 그래프 컴파일 없이 훨씬 쉬운 디버깅 및 제어 흐름.
    In the case where performance is more important, a handy tf.function decorator is also provided to help compile the code into a static graph.
      : 성과가 더 중요한 경우 손쉬운 tf.function decorator도 제공되어 코드를 정적 그래프로 컴파일하는 것을 돕는다.
    But, the reality is, eager mode and tf.function are still buggy or not well documented sometimes, which makes your life even harder in a complicated system like YOLO v3.
      : 그러나 현실은, eager mode와 tf.funtion은 여전히 버그(buggy)이거나 때로는 잘 문서화되지 않아, YOLO v3와 같은 복잡한 시스템에서 당신의 삶을 더욱 힘들게 한다.
    Also, Keras model isn’t quite flexible, while the custom training loop is still quite experimental.
      : 또한 케라스 모델은 유연성이 떨어지는 반면, 맞춤 훈련 루프는 여전히 실험적이다.
    Therefore, the best strategy for you to write YOLO v3 in TF 2 is to start with a minimum working template first, and gradually add more logic to this shell.
      : 따라서 TF 2에서 YOLO v3를 작성하는 최선의 전략은 먼저 최소 작업 템플릿부터 시작하여 점차적으로 이 셸에 논리를 추가하는 것이다.
    By doing so, we can fail early and fix the bug before it hides too deeply in a giant nested graph.
      : 그렇게 함으로써, 우리는 일찍 실패하고 버그가 거대한 중첩 그래프에 너무 깊이 숨기 전에 버그를 고필 수 있다.

  **Datset**
    Aside from the framework to choose, the most important thing for successful training is the dataset.
      : 선택할 프레임워크와는 별도로, 성공적인 학습을 위해 가장 중요한 것은 데이터셋이다.
    In the paper, the author used MSCOCO dataset to validate his idea.
      : 논문에서 저자는 자신의 아이디어를 검증하기 위해 MSCOCO 데이터 집합을 사용했다.
    Indeed, this is a great dataset, and we should aim for a good accuracy on this benchmark dataset for our model.
      : 사실, 이것은 훌륭한 데이터 집합이고, 우리는 우리의 모델에 대한 이 벤치마크 데이터셋에 대한 좋은 정확도를 목표로 삼아야한다.
    However, a big dataset like this could also hide some bugs in your code.
      : 하지만, 이와 같은 큰 데이터 집합은 당신의 코드에 버그를 숨길 수도 있다.
    For example, if the loss is not dropping, how do you know if it just needs more time to converge, or your loss function is wrong? Even with 8x V100 GPU, the training is still not fast enough for you to quickly iterate and fix things.
      : 예를 들어 손실이 떨어지지 않는다면 수렴하는데 시간이 더 필요한 것인지, 손실 기능이 잘못된 것인지 어떻게 알 수 있는가? V100 GPU 8배에도 빠르게 반복하고 고칠 수 있을 만큼 훈련 속도가 빠르지 않다.
    Therefore, I recommend you to build a development set which contains tens of images to make sure your code looks “working” first.
      : 그러므로 먼저 당신의 코드가 "작동"해 보이도록 하기 위해 수십 개의 이미지가 포함된 개발 세트를 만들 것을 권한다.
    Another option is to use VOC 2007 dataset, which only has 2500 training images.
      : 또 다른 옵션은 2,500개의 훈련 영상만 있는 VOC 2007 데이터셋을 사용하는 것이다.

  **Preprocessing**
    Preprocessing stands for the operations to translate raw data into a proper input format of the network.
      : 원시 데이터를 네트워크의 적절한 입력 형식으로 변환하는 작업을 의미한다.
    For the image classification task, we usually just need to resize the image, and one-hot encode the label.
      : 이미지 분류 작업은 보통 이미지 크기만 조정하면 되고, 라벨을 one-hot encoding하면된다.
    But things are a bit more complicated for YOLO v3.
      : 그러나 YOLO v3는 상황이 좀 더 복잡하다.
    Remember I said the output of the network is like 52x52x3x(4+1+num_classes) and has three different scales? Since we need to calculate the delta between ground truth and prediction, we also need to format our ground truth into such a matrix first.
      : 네트워크의 출력이 52x52x3x(4+1+num_classes)와 같은 세 가지 다른 척도를 가지고 있다고 말한 것을 기억하는가?
      : gt와 prediction 사이의 델타를 계산해야 하기 때문에 gt를 먼저 그런 매트릭스로 포맷해야한다.

    For each ground truth bounding box, we need to pick the best scale and anchor for it. For example, a tiny kite in the sky should be in the small scale (52x52).
      : 각각의 gt bounding box에 대해, 우리는 그것에 가장 좋은 scale와 anchor를 선택해야한다. 예를 들어, 하늘에 있는 작연 연은 small scale (52x52)에 있어야한다.
    And if the kite is more like a square in the image, we should also pick the most square-shaped anchor in that scale.
      : 그리고 연이 이미지의 사각형에 더 가깝다면 scale에서 가장 정사각형 모양의 anchor를 골라야 한다.
    In YOLO v3, the author provides 9 anchors for 3 scales.
      : YOLO v3에서 저자는 3개의 scale로 9개의 anchor를 제공한다.
    All we need to do is to choose the one that matches our ground truth box the most.
      : 우리가 해야 할 일은 우리의 gt box와 가장 일치하는 것을 선택하는 것이다.
    When I implement this, I thought I need the coordinates of the anchor box as well to calculate IOU.
      : 이것을 실행했을 때 IOU를 계산하기 위해서는 anchor box의 좌표도 필요하다고 생각했다.
    In fact, you don’t need to.
      : 사실 그럴 필요 없다.
    Since we just want to know which anchor fits our ground truth box best, we can just assume all anchors and the ground truth box share the same centroid.
      : 우리는 단지 어떤 anchor가 우리의 gt box에 가장 잘 맞는지 알고 싶을 뿐이므로, 우리는 모든 anchor와 gt box가 같은 중심을 공유하고 있다고 가정할 수 있다.
    And with this assumption, the degree of matching would be the overlapping area, which can be calculated by min width * min height.
      : 그리고 이러한 가정으로 일치의 정도는 겹치는 면적이 될 것이며, 이는 최소 width * 최소 height로 계산할 수 있다.

    During the transformation, one could also add some data augmentation to increase the variety of training set virtually.
      : 변환한는 동안, 가상으로 훈련 세트의 다양성을 증가시키기 위해 일부 데이터 증강을 추가할 수도 있다.
    For example, typical augmentation includes random flipping, random cropping, and random translating.
      : 예를 들어, 일반적인 증가는 무작위 플립, 무작위 자르기, 무작위 translating을 포함한다.
    However, these augmentations won’t block you from training a working detector, so I won’t cover much about this advanced topic.
      : 하지만, 이러한 증강은 작동하는 탐지기를 훈련시키는 것을 막지 못할 것이기 때문에, 나는 이 진보된 주제에 대해 별로 다루지 않을 것이다.

  **Training**
    After all these discussions, you finally have a chance to run “python train.py” and start your model training.
      : 이 모든 논의 끝에 마침내 "python train.py"를 실행하고 모델 교육을 시작할 수 있는 기회를 갖게 된다.
    And this is also when you meet most of your bugs.
      : 그리고 이것은 또한 여러분이 대부분의 버그들을 만날 때 이다.
    You could refer to my training script here when you are blocked.
      : 네가 막혔을 때 여기 있는 내 training script를 참조할 수 있다.
    Meanwhile, I want to provide some tips that are helpful for my own training.
      : 한편, 나는 나 자신의 훈련에 도움이 되는 몇 가지 팁을 제공하고 싶다.

    *Non Loss*
      1. Check your learning rate and make sure it’s not too high to explode your gradient.
        : 합습 속도를 확인하고 경사를 폭발시킬 정도로 높지 않은지 확인해라
      2. Check for 0 in binary cross-entropy because ln(0) is not a number. You can clip the value from (epsilon, 1 — epsilon).
      3. Find an example and walk through your loss step by step. Find out which part of your loss goes to NaN. For example, if width/height loss went to NaN, it could be because the way you calculate from tw to bw is wrong.
        : 예를 찾아 차근차근 손실을 헤쳐나가십시오. 손실의 어느 부분이 NaN에게 가는지 알아보세요. 예를 들어, 너비/높이 손실이 NaN으로 갔다면, 그것은 당신이 tw에서 bw까지 계산하는 방법이 잘못되었기 때문일 수 있다.

    *Loss remains high*
      1. Try to increase your learning rate to see if it can drop faster. Mine starts at 0.01. But I’ve seen 1e-4 and 1e-5 works too.
        : 학습 속도가 더 빨리 떨어질 수 있는지 알아보기 위해 학습 속도를 높여라. 하지만 1e-4와 1e-5도 작동하는 것을 본 적이 있다.
      2. Visualize your preprocessed ground truth to see if it makes sense. One problem I had before is that my output grid is in [y][x] instead of [x][y], but my ground truth is reversed.
        : 사전 처리된 gt를 시각화하여 이해가 되는지 확인해라. 전에 내가 겪었던 한 가지 문제는 출력 grid가 [x][y]가 아닌 [y][x]에 있다는 것인데, 나의 gt가 거꾸로 되어 있었다는 것이다.
      3. Again, manually walk through your loss with a real example. I had a mistake of calculating cross-entropy between objectness and class probabilities.
        : 다시 한번 실제 예시로 하나하나 손실을 헤쳐 나가라. 나는 objectness와 class probabilities 사이의 cross-entropy를 계산하는 실수를 했다.
      4. My loss also remains around 40 after 50 epochs of MSCOCO. However, the result isn’t that bad.
        :
      5. Double-check the coordinates format throughout your code. YOLO requires xywh (centroid x, centroid y, width and height), but most of dataset comes as x1y1x2y2 (xmin, ymin, xmax, ymax).
        :
      6. Double-check your network architecture. Don’t get misled by the diagram from a post called “A Closer Look at YOLOv3 — CyberAILab”.
        :
      7. tf.keras.losses.binary_crossentropy isn’t the sum of binary cross-entropy you need.

    *Loss is low, but the prediction is off*
      1. Adjusting lambda_coord or lambda_noobj to the loss based on your observation.
        : 관측치에 따라 lambda_coord 또는 lambda_noobj를 손실에 맞게 조정해라.
      2. If you are traininig on your own dataset, and the dataset is relative small (< 30k images), you should intialize weights from a COCO pretrained model first.
        : 자체 데이터 집합에서 훈련을 받고 있고 데이터 집합이 상대적으로 작은 경우 (<30k 이미지) 먼저 COCO 사전 검증된 모델에서 가중치를 초기화 해야 한다.
      3. Double-check your non max suppression code and adjust some threshold (I’ll talk about NMS later).
        : 당신의 non-max-suppression를 두 번 확인하고 임계값을 조정해라 (NMS에 대해서는 뒤에 설명함)
      4. Make sure your obj_mask in the loss function isn’t mistakenly taking out necessary elements.
        : 손실 함수의 obj_mask가 필요한 요소를 실수로 빼내지 않도록 해라.
      5. Again and again, your loss function. When calculating loss, it uses relative xywh in a cell (also called tx, ty, tw, th). When calculating ignore mask and IOU, it uses absolute xywh in the whole image, though. Don’t mix them up.
        : 몇 번이고, 또 다시, 당신의 손실 기능. 손실을 계산할 때 셀에서 상대 xywh(tx, ty, tw, th라고도 함)를 사용한다. 그러나 무시 마스크와 IOU를 계산할 때는 전체 이미지에서 절대 xywh를 사용한다. 그것들을 섞지 마라.

    *Loss is low, but there's no prediction*
      1.  If you are using a custom dataset, please check the distribution of your ground truth boxes first. The amount and quality of the boxes could really affect what the network learn (or cheat) to do.
        : 커스텀 데이터셋을 사용하는 경우, 먼저 gt boxes의 분포를 확인해라. 상자의 양과 품질은 네트워크가 무엇을 하기 위해 배우거나 부정행위를 하는지에 정말로 영향을 미칠 수 있다.
      2. Predict on your training set to see if your model can overfit on the training set at least.
        : 모델이 적어도 교육 세트에 오버핏할 수 있는지 확인하기 위해 교육 세트에 대해 예측하십시오.

    *Multi-GPU training*
      PASS

  **Postprocessing**  !! eye tacking에서는 필요 없을듯
    The final component in this detection system is a post-processor.
      : 이 검출 시스템의 최종 구성요소는 후처리장치다.
    Usually, postprocessing is just about trivial things like replacing machine-readable class id with human-readable class text.
      :
    In object detection, though, we have one more crucial step to do to get final human-readable results. This is called non maximum suppression.
      : 그러나 물체 감지에서 우리는 인간이 읽을 수 있는 최종 결과를 얻기 위해 해야 할 중요한 단계가 하나 더 있다. 이것을 non maximum supression(NMS)라고 한다.
    Let’s recall our objectness loss. When is false proposal has great overlap with ground truth, we won’t penalize it with noobj_loss.
      : 우리의 objectness loss를 생각해보자, 거짓 제안이 gt와 중첩될 때, 우리는 noobj_loss로 그것을 처벌하지 않을 것이다.
    This encourages the network to predict close results so that we can train it more easily.
      : 이것은 우리가 그것을 더 쉽게 훈련할 수 있도록 네트워크가 가까운 결과를 예측하도록 장려한다.
    Also, although not used in YOLO, when the sliding window approach is used, multiple windows could predict the same object.
      : 또한, YOLO에서는 사용되지 않지만 슬라이딩 윈도우 기법을 사용할 때 여러 개의 윈도우가 동일한 물체를 예측할 수 있다.
    In order to eliminate these duplicate results, smart researchers designed an algorithm called non maximum supression (NMS).
      : 이러한 중복된 결과를 제거하기 위해, 스마트한 연구자들은 NMS라는 알고리즘을 설계했다.

    The idea of NMS is quite simple. Find out the detection box with the best confidence first, add it to the final result, and then eliminates all other boxes which have IOU over a certain threshold with this best box.
      : NMS에 대한 생각은 꽤 간단하다.
      : 가장 confidence 값이 가장 높은 box를 먼저 찾아 최종 결과에 추가한 다음 이 best box를 사용하여 IOU가 특정 임계값을 초과하는 다른 모든 상자를 제거해라
    Next, you choose another box with the best confidence in the remaining boxes and do the same thing over and over until nothing is left.
      : 다음으로 남은 box들 중 confidence 값이 가장 높은 box를 선택하고 아무것도 남지 않을 때까지 같은 일을 반복한다.
    In the code, since TensorFlow needs explicit shape most of the time, we will usually define a max number of detection and stop early if that number is reached.
      : 코드에서 텐서플로우는 대부분 explicit shape(명시적 형상)이 필요하기 때문에 최대 검출 횟수를 정의하고 그 수에 도달하면 조기 정지한다.
    In YOLO v3, our classification is not mutually exclusive anymore, and one detection could have more than one true class.
      : YOLO v3에서 우리의 분류는 더 이상 상호 배타적이지 않으며, 하나의 검출은 둘 이상의 참 클래스를 가질 수 있다.
    However, some existing NMS code doesn’t take that into consideration, so be careful when you use them.
      : 단, 기존 NMS 코드는 이를 고려하지 않으므로 사용할 때 주의해라.
