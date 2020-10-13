---
layout: category-post
title:  "Paper recap: Focal Loss for Dense Object Detection"
date:   2020-10-12
categories: paper-recap
---
## Focal Loss for Dense Object Detection
[Lin et al, 2020](https://arxiv.org/abs/1708.02002)

## Research Topic
- Category (General): Deep Learning
- Category (Specific): Computer Vision

## Paper summary
- Propose a new loss function tackling the extreme class imbalance problem encountered in object detections.
- Reshaping Cross Entropy Loss Function such that it down-weights the loss assigned to well-classified examples.
- Understand if one-stage de-tectors can match or surpass the accuracy of two-stage detectors while running at similar or faster speeds.
- Introduce the RetinaNet to evaluate the effectiveness of the loss, but since the author stated: "Our simple detector achieves top results not based on innovations in network design but due to our novel loss.", I'm gonna skip the RetinaNet in this recap.

## Explain Like I'm 5 (ELI5) 👶👶👶
- Basically it is Balanced Cross Entropy Loss Function but with a tunable modulating factor such that it controls the weight loss of easy/hard samples.

## Issues addressed by the paper
- Current state-of-the-art object detectors are based on a two-stage, proposal-driven mechanism, which  consistently achieves top accuracy on the challenging COCO benchmark.
- Could a simple one-stage detector achieve similar accuracy?
-  Recent work on one-stage detectors yield faster detectors with accuracy within 10-40% relative to state-of-the-art two-stage methods, but takes also a lot of time.
-  Class imbalance is addressed in R-CNN-like (one-stage) detectors by a two-stage cascade and sampling heuristics.
-  An one-stage detector must process a much larger set of candidate object locations regularly sampled across an image, which makes the model inefficient as the training procedure is still dominated by easily classified background examples.
- Balanced CE loss function: while α balances the importance of positive/negative examples, it does not differentiate between easy/hard examples.

## Approach/Method
- Present a one-stage object detector (RetinaNet) that, for the first time (by the time of publish), matches the state-of-the-art COCO AP.
-  To achieve this result:
    - Identify class imbalance during training as the main obstacle impeding one-stage detector from achieving state-of-the-art accuracy,
    - Propose a new loss function that eliminates this barrier.

- Focal Loss:
    - Dynamically scaled cross entropy loss.
    - Automatically down-weight the contribution of easy examples (inliers) during training and rapidly focus the model on hard examples.
    - Focal loss performs the _opposite role_ of a robust loss (e.g: Huber Loss): it focuses training on a sparse set of hard examples.
    - They first defined the probability function like this:

<p align="center">
    $$
        p_t = \left\{
            \begin{array}{ll}
                p & if \ y=1 \\
                1-p & otherwise.
            \end{array}
        \right.
    $$
</p>    

  - Using this definition, we can define CE loss func like this: \\(CE(p,y) = CE(p_t) = -log(p_t)\\) for comparision.
  - Focal Loss definition:

$$
    FL(p_t) = -(1-p_t)^{\gamma}log(p_t)
$$

## Best practice


## Results


## Conclusions


### The author's conclusions


### Rating
![rating](https://media.giphy.com/media/z8rEcJ6I0hiUM/giphy.gif)

### My Conclusion


## Paper implementation


## Cited references and used images from:


## Papers needs to conquer next 👏👏👏
