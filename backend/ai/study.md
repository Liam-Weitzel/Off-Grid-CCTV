---
title: "How can the performance of various object detection AI models be optimized when trained on small amounts of data and applied to analog video streams in conditions of interference?"
author:
  - Liam Weitzel
  - 20211316
  - 04/24/2024
---

\newpage

## Abstract

The expansion of the modern trend to utilize object detection in real time digital CCTV footage to wireless analog CCTV systems still presents challenges due to the high levels of noise of analog video. In this paper we aim to present a benchmark of various object detection model architectures on five new small datasets consisting of images from analog video streams with conditions of interference. The model architectures tested are Yolov8, Faster-RCNN, Masked-RCNN, and Retinanet. We compared the models using different hyperparameters, amount of training, and with or without pre-training. From this comparison we found that Yolov8 performed the best in terms of inference time, although Faster-RCNN showed the highest accuracy. We provide the observation that live object detection on analog video streams with conditions of interference is possible with commendable performance and accuracy.

## Introduction

Modern wireless closed-circuit television systems (CCTV) generally use IP cameras and wifi due to wifi’s high throughput and IP camera's high resolution. Although these technologies are capable of transmitting high-quality video streams, the wifi standards reliance on the 2.4 GHz and 5.8 GHz bands limits the range of these wireless systems significantly. These high-frequency bands can carry 450 Mbps and over 1 Gbps respectively but make a trade-off with range for this high throughput. For a long-range wireless CCTV application where a single video stream might only require a throughput of ~3 Mbps, this is excessive. The use of analog cameras in combination with lower radio frequencies can increase the range and penetration of the wireless link significantly. The lowest legally available frequency for hobbyist use that meets our new throughput requirement of 6 Mhz is the 900 Mhz band. Unfortunately, the 900 Mhz band is widely used by communication protocols, Internet of Things (IoT) devices, and mobile broadband which makes it susceptible to heavy interference in long-range wireless applications.

It is common practice to analyze the video streams of modern CCTV systems in real-time using object detection artificial intelligence models for security purposes. The present paper will explore how the performance of various object detection AI models can be optimized when trained on small amounts of data and applied to analog video streams in conditions of interference. Literature specifically covering object detection on analog video systems is non-existent. Regardless, the type of noise that analog video introduces is not unique, and literature regarding object detection on degraded and noisy video streams is plentiful. This is the gap in the current literature that the present paper aims to bridge.

The research topic will be explored by training various object detection models on five different analog CCTV cameras. The data for each camera was gathered over the course of 24 hours with an interval of four and a half minutes. The hyperparameters, amount of training, and whether the model was pre-trained will be adjusted and compared using industry-standard evaluation metrics.

## Literature Review
### Latest developments in object detection

From a high-level overview of the recent literature, it is clear that the state-of-the-art object detection models use either a one-stage or two-stage method. One-stage methods are designed to decrease inference speed as much as possible whereas two-stage methods are designed to increase detection accuracy. Examples of one-stage models include RetinaNet and YOLO. Examples of two-stage models include faster R-CNN and mask R-CNN. COCO is a family of benchmark datasets that are commonly used to evaluate the performance of various models. The Mean Average Precision metric is generally used to compare and contrast models. Contrary to the latest developments in large language models, the use of transformers is limited among top-performing object detection models. As Tu et al. (2022) explain, "the lack of scalability of self-attention mechanisms with respect to image size has limited their (transformers) wide adoption in state-of-the-art vision backbones". Nonetheless, "the remarkable success of transformers in largescale language models, vision transformers (ViTs), have also swept the computer vision field and are becoming the primary choice for the research and practice of large-scale vision foundation models." (Wang et al. 2022).

Zong et al. (2022) conducted "extensive experiments to evaluate the effectiveness of the proposed approach on DETR variants". The results show a detailed benchmark table of the models: Co-DETR (Swin-L), Co-DETR and Co-DETR (single-scale) evaluated using the COCO minival, COCO test-dev, and LVIS v1.0 minival datasets. Using the box Mean Average Precision (mAP) metric it is clear that the co-DETR has the highest accuracy in this comparison. Using this mAP metric we can determine the global ranking of this model and see that it outperforms any existing current model. 

Fang et al. (2022) introduce a new model 'EVA' which outperforms many modern models and "can efficiently scale up EVA to one billion parameters". Although EVA's architecture has a heavy focus on scalability and inference speed over accuracy, the results boast a box mAP value of 64.7 which places it among the top 10 on the global rankings when trained on the COCO test-dev dataset. Furthermore, when evaluated using the COCO-O dataset EVA boasts an Average mAP of 57.8 outperforming all models to date on this dataset.

Ghiasi et al. (2020) use "a systematic study of the Copy-Paste augmentation" where the training pipeline "randomly pastes objects onto an image". "Furthermore, we show Copy-Paste is additive with semi-supervised methods that leverage extra data through pseudo labeling (e.g. self-training). On COCO instance segmentation, we achieve 49.1 mask AP and 57.3 box AP, an improvement of +0.6 mask AP and +1.5 box AP over the previous state-of-the-art. We further demonstrate that Copy-Paste can lead to significant improvements on the LVIS benchmark. Our baseline model outperforms the LVIS 2020 Challenge winning entry by +3.6 mask AP on rare categories.". But most notably for our use case, object detection, when evaluated using the PASCAL VOC 2007 dataset, they achieved a MAP of 89.3% outperforming all models on this dataset.

Li et al. (2023) with their release of YOLOv6, which is a model architecture that prioritizes inference speed over accuracy, "achieve better accuracy performance (50.0%/52.8% respectively) than other detectors at a similar inference speed". According to their published benchmarks, YOLOv6-L6 outperforms all models on the COCO 2017 val dataset. Furthermore, their YOLOX-L model outperforms all models when evaluated using the Waymo 2D detection all_ns f0val and Manga109-s 15test datasets.

MaxViT is the first model to implement the use of Transformers in its architecture. Tu et al's paper introduces MaxVit, along with its novel architecture. Tu et al's results "demonstrate the effectiveness of [MaxVit] on a broad spectrum of vision tasks". Notably for object detection, MaxVit-B outperforms all existing object detection models when evaluated using the COCO 2017 data set.

Wang et al. (2022) present "a new large-scale CNN-based foundation model, termed InternImage". "Different from the recent CNNs that focus on large dense kernels, InternImage takes deformable convolution as the core operator, so that our model not only has the large effective receptive field required for downstream tasks such as detection and segmentation, but also has the adaptive spatial aggregation conditioned by input and task information". InternImage takes the approach many modern transformer models take by architecture-level design, scaling-up parameters, and training on massive amounts of data in hopes of increasing accuracy whilst still having the benefits of a CNN architecture. This novel approach demonstrates high levels of accuracy in datasets such as COCO minival, COCO test-dev, and ADE20K. InternImage scored particularly high when evaluated using the CrowdHuman (full body) dataset outperforming all other models.

Dagli, R. and Shaikh, A. (2021) present an interesting dataset made up of medical personal protective equipment. This dataset posed challenging for models such as FasterRCNN, YOLOv3, VarifocalNet, RegNet, Deformable Convolutional Network, and Double Heads. Although the selection of models tested is not up to date, the custom model Dagli, R. and Shaikh, A. designed and trained specifically for this data set, termed TridentNet, outperformed all other models tested.

\newpage

\centerline{Table 1. Datasets listed with their best performing model}  

\begin{tabular}[t]{ll}
    **Dataset** & **Best model** \\
    COCO test-dev & Co-DETR \\
    COCO minival & Co-DETR \\
    COCO-O & EVA \\
    PASCAL VOC 2007 & Cascade Eff-B7 NAS-FPN (Copy Paste pre-training, single-scale) \\
    COCO 2017 val & YOLOv6-L6(46 fps, V100, bs1) \\
    COCO 2017 & MaxViT-B \\
    CrowdHuman (full body) & InternImage-H \\
    CPPE-5 & TridentNet \\
    Waymo detection allns f0val & YOLOX-L \\
    Manga109-s 15test & YOLOX-L \\
\end{tabular}  

\begin{center}
This table summarizes the paragraphs above, listing each dataset alongside the model architecture that had the highest normalized score across all reported metrics.
\end{center}

Lastly, Shinya, Y. (2021) further justified the results from the aforementioned studies in the study "USB: Universal-Scale Object Detection Benchmark". This study aims to fairly compare and benchmark various object detection models by mixing popular datasets such as COCO, with newer datasets such as Waymo Open Dataset and Manga109-s. This amalgamated dataset hopes to address some of the issues observed when benchmarking with COCO like datasets.

### Types of noise and their impact

There are too many sources of noise present in a CCTV camera transmitting analog video over 900 MHz to list. The nature of the noise is hard to predict due to the medium over which the video signal travels. There are many more variations in air than a standard cable. Perhaps one camera has some foilage in the signal path, other camera systems might have a direct line of sight. Regardless, there are also some constant sources of noise present in this system. The two primary sources of predictable, and constant noise are the analog-to-digital conversion circuit and the de-interlacing process of the analog video. These two components introduce a distinct type of interference which we can discuss and improve. Analog to digital video conversion is known to introduce 'salt-and-pepper' noise. "This kind of noise randomly changes intensities of some pixels to the maximum or minimum values of the intensity range on the image/video" (Veerakumar et al. 2011). Furthermore, analog video has certain characteristics which can also be interpreted as 'noise'. One of which is caused by the de-interlacing process upon which both NTSC and PAL analog video protocols are built, causing analog video to have distinct visible lines going across the image. Lastly, both NTSC and PAL have a limit of transmitting 0.3 Megapixel images which can be interpreted as a 'low resolution' image and hence also 'noise'.

An exhaustive search reveals that only a limited amount of literature exists on the effects of more generally, 'noise', on the performance of object detection models. It is important to note that this search did not reveal any sources specifically covering noise induced from analog video transmitted over any radio frequency. 

Interestingly, Momeny et al. (2021) discuss the use of noise as a data augmentation method to improve accuracy for image classification CNN models. An increase in training speed and accuracy was observed although the increase in accuracy was primarily observed in noisy images.

Nazaré et al. (2018), aim to "evaluate the generalization of models learned by different networks using noisy images". The results indicate that if the application's image quality is prone to noise, training the image classification model on noisy images will generally increase its accuracy. This directly contradicts Paranhos Da Costa et al. (2016) in the study "An empirical study on the effects of different types of noise in image classification tasks", who speculate "noise makes classiﬁcation more difficult due to the fact that models trained with a particular noisy/restored training set version – and tested on images with the same noise conﬁguration – usually perform worse than a model trained and tested on the original data".

Rodríguez et al. (2024) discuss the impact of many different types of noise on object detection models. This paper has a particular focus on Gaussian noise and brightness noise which are both types of noise that are also present in our video. The effects of these two types of noise are tested on various YOLO object detection models, as well as FasterRCNN ResNet50. The results show that "the size of objects to be detected is a factor that, together with noise and brightness factors, has a considerable impact on their performance".

Veerakumar et al. (2011) propose an algorithm that effectively removes 'salt-and-pepper' noise from any given image. The proposed solution uses a 3x3 pixel search grid and determines the median pixel RGB value. This process effectively blurs the image, removing noise but also significantly lowering the resolution. Nonetheless, it is untested how applying this algorithm would impact the performance of object detection models.

### Impact of low-resolution video on object detection

As previously discussed, the impact of radio channel interference-induced noise on the performance of object detection models on videos streamed over 900 MHz is not well researched, thus the impact of low resolution is covered in this section in hopes that the strategies covered to increase performance can be used interchangeably between these two contexts. Uzkent et al. (2019) explore an approach to increase the performance of a reinforcement learning object detection agent in high-resolution images. The agent chooses parts of the image to analyze, effectively lowering the resolution and maintaining accuracy while increasing performance. This approach resulted in a 50% increase in runtime without compromising accuracy. Instead, Lu et al. (2015) utilize a deep convolutional neural network to improve object detection on high-resolution images.

Contrarily, Kidwell et al. (2015) directly address the challenge of maintaining detection performance with low-resolution images. Kidwell et al. (2015) investigate object detection in low-resolution overhead imagery presenting a novel detection system that employs a fast sliding-window detector using a histogram of oriented gradient (HOG) and a supervised support vector machine classifier. This approach shows promising accuracy but sacrifices performance. Furthermore, Kidwell et al. (2015) state that this approach shows an increase in accuracy even with less annotated training data compared to their benchmark object detection algorithm. Pava et al. (2011) discuss object detection on low-resolution 3-D model animations. Pava et al. (2011) propose an elaborate system compromising of preprocessing, background modeling, information extraction, and postprocessing stages. Similarly, Kidwell et al. (2015) utilized techniques such as Histogram of Oriented Gradient (HOG) and equalization, background subtraction, and various filtering techniques. Zhang et al. (2007) introduce a novel multi-resolution framework for object detection which is inspired by human visual search patterns. Zhang et al. (2007)’s proposed algorithm artificially lowers the resolution of the image and attempts to detect the object in the image. If the confidence score of the result is low, increase the resolution and try again. This approach allows for an 'early exit' designed for real-time analysis significantly increasing performance while maintaining high detection accuracy. Most notably, Clark et al. (2023) investigate the resolution-performance trade-off of object detection models. Clark et al. (2023) found a linear relationship between resolution and detection accuracy highlighting the crucial role of resolution in object detection. The present paper will use the prior techniques discussed to attempt to increase the performance and accuracy of object detection on video streams that are missing information due to interference on the 900 MHz band.

\newpage

## Methods
### Procuring the data

To evaluate the performance of various object detection models on analog video under the condition of significant interference, five live analog camera video streams were selected with varying levels of noise. The five cameras are located at:

Place: Hadji Dimitar Square, Sliven, Bulgaria  
Coordinates: 41.940300 / 25.569401

Place: Keskväljak, Paide, Estonia  
Coordinates: 59.433899 / 24.728100

Place: Duomo, Noto, Sicily, Italy  
Coordinates: 36.890140 / 15.069290

Place: Kielce University of Technology, Kielce, Poland  
Coordinates: 50.833302 / 20.666700

Place: Toggenburg Alpaca Ranch, Stein, Switzerland  
Coordinates: 47.366699 / 8.550000

Using ffmpeg and bash a frame was captured from each camera at a 4-and-a-half minute interval: `while true; do ffmpeg -i INPUT -frames:v 1 -vcodec mjpeg -f image2 \~/Downloads/frames/image$(date +%Y%m%d%H%M%S).jpg; sleep 270; done`.

Data collection lasted for 24 hours, on all cameras simultaneously, on the 2nd of January 224, resulting in 323 images (frames) captured per camera.

Each frame was labeled by hand, over the course of the next week, using a webapp called roboflow. The resulting dataset for each camera has a 70%/20%/10% split between training, test, and validation data respectively.

No augmentation methods of any kind are applied to the final datasets.

### Models

The model architectures that will be compared in the present paper are Faster-RCNN, Masked-RCNN, Retinanet, Yolov8.

### Hyper parameters

Before training, hyper parameter tuning was conducted for each dataset using Yolov8 using the AdamW optimization algorithm, for 300 iterations of 50 epochs each.

\newpage

### Training and evaluation

To evaluate the aforementionend model architectures the industry standard metrics: precision, recall, F1-Score, and confidence will be used. To be specific, for each trained model, the graphs: precision over confidence, recall over confidence, F1 over confidence, and precision over recall will be used to evaluate each model. For all models trained using the yolov8 architecture, a normalized and non-normalized confusion matrix will also be used. Each model architecture will be trained on each dataset for 150 epochs. Whether the model has peen pre-trained, and/ or hyperparameter tuned will be varied. Each model will be evaluated every 50 epochs.

Independent variables:  
Model Type: Yolov8, Faster-RCNN, Masked-RCNN, Retinanet-RCNN.  
Training Epochs: The number of epochs each model will be trained for.  
Pre-training: Whether the models are pre-trained or not.  
Parameter Tuning: Whether the models will use stock parameters or will be hyperparameter-tuned.  

Dependent variables:  
Performance Metrics: Precision, recall, F1-score, confidence, confusion matrices.  
Resource Utilization: All models will be trained on the same hardware with the same resource utilization.  

\newpage

## Results
### Explain the metrics

Precision is the ratio of true positives to the sum of true and false positives.
$$ Precision = (True Positives+False Positives) / True Positives $$

Recall is the ratio of true positives to the sum of true positives and false negative.
$$ Recall = (True Positives+False Negatives) / True Positives $$

Using precision we can effectively deduce how many objects were correctly labeled out of all labeled objects.
Similarly, using recall we can deduce how many objects were correctly labeled out of all objects whether they were labeled or not.

The F1 score is simply a harmonic mean of precision and recall, allowing us to combine precision and recall into a 'score'.
$$ F1 = 2 * ((Precision + Recall) / (Precision * Recall)) $$

Before analyzing the results, it is important to note that not all results are shown in the present paper. For a complete breakdown of the results see the supplementary data. A selection of representative results has been assembled and will be reviewed in the interest of brevity instead.

\newpage

### Best performing final model comparison

A comparison of the best models trained on the Doumo dataset:  
![](./saved_runs/train34/F1_curve.png){width=50%}
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-duomo-valid-images_f1_over_confidence.png){width=50%}  
a. \hfill b.  

![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-duomo-valid-images_f1_over_confidence.png){width=50%}
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-duomo-valid-images_f1_over_confidence.png){width=50%}  
c. \hfill d.  

\begin{center}
Figure 1. F1 over confidence graph for the pre-trained and hyperparameter-tuned a) Yolov8n, b) Faster-RCNN, c) Masked-RCNN, and d) Retinanet models trained for 150 epochs on the Duomo dataset.
\end{center}

Figure 1 indicates that the models trained using the Masked-RCNN (c) and Faster-RCNN (b) architectures undoubtedly outpreformed the models trained using the Retinanet (d) and Yolov8n (a) architectures on the Duomo dataset.  

\newpage

A comparison of the best models trained on the Hadji Dimitar Square dataset:  
![](./saved_runs/train37/F1_curve.png){width=50%}
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png){width=50%}  
a. \hfill b.

![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png){width=50%}
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png){width=50%}  
c. \hfill d.  

\begin{center}
Figure 2. F1 over confidence graph for the pre-trained and hyperparameter-tuned a) Yolov8n, b) Faster-RCNN, c) Masked-RCNN, and d) Retinanet models trained for 150 epochs on the Hadji Dimitar Square dataset.
\end{center}

Figure 2 further supports the observations made from figure 1. Models trained using the Masked-RCNN (c) and Faster-RCNN (b) architectures again clearly outpreformed the models trained using the Retinanet (d) and Yolov8n (a) architectures on the Hadji Dimitar Square dataset.  

\newpage

### Effect of hyper-parameter tuning 

A comparison of a hyperparameter tuned model against a non-hyperparameter tuned model trained on the Hadji Dimitar Square dataset:  
![](./saved_runs/train37/F1_curve.png){width=50%}
![](./saved_runs/train6/F1_curve.png){width=50%}  
a. \hfill b.  

\begin{center}
Figure 3. F1 over confidence graph for the a) hyperparameter tuned, and b) non-hyperparmeter-tuned Yolov8n model trained for 150 epochs on the Hadji Dimitar Square dataset.
\end{center}

Figure 3 indicates that the hyperparameter tuned model expressed a slight performance increase over its non-hyperparameter tuned counterpart.

A comparison of a hyperparameter tuned model against a non-hyperparameter tuned model trained on the Keskväljak dataset:  
![](./saved_runs/train40/F1_curve.png){width=50%}
![](./saved_runs/train9/F1_curve.png){width=50%}  
a. \hfill b.  

\begin{center}
Figure 4. F1 over confidence graph for the a) hyperparameter tuned, and b) non-hyperparmeter-tuned Yolov8n model trained for 150 epochs on the Keskväljak dataset.
\end{center}

Figure 4, contrarily to figure 3 indicates that the hyperparameter tuned model expressed a slight performance decrease over its non-hyperparameter tuned counterpart. Through further analysis of the supplementary data, the conclusion is derived that the effect of hyperparameter tuning on a models performance on the tested datasets is unpredictable.

\newpage

### Effect of pre-training

Comparing the peformance of a Yolov8n pre-trained model against a Yolov8n non-pre-trained model that were both trained on the Doumo dataset for 150 epochs:  
![](./saved_runs/train19/F1_curve.png){width=50%}
![](./saved_runs/train3/F1_curve.png){width=50%}  
a. \hfill b.  

\begin{center}
Figure 5. F1 over confidence graph for the a) pre-trained, and b) non pre-trained Yolov8n model trained for 150 epochs on the Doumo dataset.
\end{center}

A negligable effect of pre-training can be observed from figure 5. The pre-trained model peformed slightly better than the non pre-trained model on the Duomo dataset.

Comparing the peformance of a Faster-RCNN pre-trained model against a Faster-RCNN non pre-trained model that were both trained on the Keskväljak dataset for 150 epochs:  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-keskvaljak-valid-images_f1_over_confidence.png){width=50%}
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-keskvaljak-valid-images_f1_over_confidence.png){width=50%}  
a. \hfill b.  

\begin{center}
Figure 6. F1 over confidence graph for the a) pre-trained, and b) non pre-trained Faster-RCNN model trained for 150 epochs on the Keskväljak dataset.
\end{center}

Similarly to figure 5, figure 6 indicates that models trained on the Keskväljak dataset express a negligable increase in performance when pre-trained.

Comparing the peformance of a Yolov8n pre-trained model against a Yolov8n non-pre-trained model that were both trained on the Keskväljak dataset for 150 epochs:  
![](./saved_runs/train25/F1_curve.png){width=50%}
![](./saved_runs/train9/F1_curve.png){width=50%}  
a. \hfill b.  

\begin{center}
Figure 7. F1 over confidence graph for the a) pre-trained, and b) non pre-trained Yolov8n model trained for 150 epochs on the Keskväljak dataset.
\end{center}

A very significant increase in performance is observed in figure 7, notably on the same dataset as figure 6, Keskväljak. Figure 6, and figure 7 only differ in the model architecture used, and a significant change was observed. This indicates that the effect of pre-training fluctuates dramatically based on the model architecture.

Comparing the peformance of a Masked-RCNN pre-trained model against a Masked-RCNN non-pre-trained model that were both trained on the Keskväljak dataset for 150 epochs:  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-keskvaljak-valid-images_f1_over_confidence.png){width=50%}
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-keskvaljak-valid-images_f1_over_confidence.png){width=50%}  
a. \hfill b.  

\begin{center}
Figure 8. F1 over confidence graph for the a) pre-trained, and b) non pre-trained Masked-RCNN model trained for 150 epochs on the Keskväljak dataset.
\end{center}

Figure 8 exhibits the same characteristics as figure 7, further supporting the observations made from figure 6 and 7. Again, only the model architecture used during training was varied between figure 6, 7, and 8.

\newpage

### Effect of training on specialized data

A comparison of the effectiveness of training on data similar to the evaluation data. The following graphs demonstrate how the evaluation is effected during training. The first sample is an evaluation taken at 0 epochs, the last, at 150 epochs, with an evaluation at an interval of 50 epochs.

![](./saved_runs/val2/F1_curve.png){width=25%}
![](./saved_runs/train/F1_curve.png){width=25%}
![](./saved_runs/train2/F1_curve.png){width=25%}
![](./saved_runs/train3/F1_curve.png){width=25%}  
a. \hfill b. \hfill c. \hfill d.  

\begin{center}
Figure 9. F1 over confidence graph for the pre-trained Yolov8n model trained for a) 0, b) 50, c) 100, d) 150 epochs on the Duomo dataset.
\end{center}

From figure 9 and further analysis of the supplementary data, it is clear that training increases performance significantly until reaching 100 epochs at which point peformance stagnates. This was observed on all datasets and all model architectures.

## Discussion

Suprisingly, when trained for 150 epochs, the model architectures: Masked-RCNN, and Faster-RCNN consistently outperformed Yolov8n eventhough Yolov8n was the only model that was hyperparameter tuned for 300 iterations using the AdamW optimization algorithm. Furthermore, both the Masked-RCNN and Faster-RCNN use a Resnet50 backend. This backend is considerably outdated compared to Yolov8 which was released in 2023. Nonetheless, it's important to mention that Yolov8n has a considerably faster inference time than Masked-RCNN and Faster-RCNN and is able to keep up with a 3 fps live stream when Masked-RCNN and Faster-RCNN are only able to infer one frame every two seconds on equivalent hardware.

The effect of hyper parameter tuning on Yolov8 across all datasets did not give any conclusive results. Depending on the dataset, the accuracy increased or decreased. There was no generalizable result observed.

Similarly, the effect of pre-training, which is also commonly known as transfer learning, also gave inconclusive results. Some data sets benefited greatly from pre-training where as other data sets were affected negatively. Pre-training did seem to generally increase the peformance of each model particularly when only trained for a short amount of time. Regardless, without a statistical analysis its not possible to derive a conclusion.

The observed effect of training on specialized data is doubtlessly positive. There has not been a single case where training the model on any of the datasets decreased the performance when evaluated using a similar evaluation dataset. This aligns directly with the findings of Dagli, R. and Shaikh, A. (2021). 

These results are promising for wireless analog CCTV system applications as it has been shown that high accuracy real time object detection is possible with a small dataset of 323 specialized images even under the condition of interference. From the results of the present paper it can be deduced that the type of noise that analog video introduces does not pose challenge for modern object detection model architectures, bridging this gap in the current literature.

The methods used in the present paper has many limitations. Unfortunately, the training and evaluation pipeline written in python for all models created with the Retinanet model architecture failed and produced incomprihensible results. This is a major oversight and could have been easily avoided with more rigorous testing. This is also the reason for Retinanet's exclusion in many of the comparisons in the results. A full breakdown of the results produced by any model trained using the Retinanet architecture can be found in the supplementary data.

Furthermore, another major limitation of the present paper is that not enough model architectures were compared to come to a conclusive answer. The literature review revealed the vast amount of model architectures and benchmarking datasets available. Having only rigorously tested and compared four different model architectures, the present paper would benefit significantly from a wider comparison.

Lastly, all datasets had a significant amount of analog noise. This leads to a question whether the analog noise in the datasets actually had any impact at all. To test this an analog noise generation algorithm would have to be applied to images that do not have any analog noise present. This data augmentation method could be explored as a future research.

## Conclusion

The use of modern object detection models on wireless analog CCTV systems is more than feasable. A small dataset of 323 images can be collected and labeled within a day. Without this small dataset of specialized data, the performance of any pre-trained model will be subpar. All five datasets procured in the present paper, with varying levels of analog noise, were able to train a model with sufficient performance for CCTV and security applications. Furthermore, the use of pre-training and hyperparameter tuning can have a positive impact on performance but also showed to have a negative impact depending on the dataset and chosen model architecture.

## References

Adavanne, S., Pertila, P. and Virtanen, T. (2017) “Sound event detection using spatial features and convolutional recurrent neural network,” IEEE International Conference on Acoustics, Speech, and Signal Processing, pp. 771–775. Available at: https://doi.org/10.1109/ICASSP.2017.7952260.

Agrawal, N.K. and Shankhdhar, A. (2022) “An Enhanced and Effective Approach for Object Detection using Deep Learning Techniques,” SMART, pp. 1482–1486. Available at: https://doi.org/10.1109/SMART55829.2022.10046678.

Atrey, P.K., Maddage, N.C. and Kankanhalli, M.S. (2006) “Audio Based Event Detection for Multimedia Surveillance,” 2006 IEEE International Conference on Acoustics Speech and Signal Processing Proceedings, 5. Available at: https://doi.org/10.1109/ICASSP.2006.1661400.

Boisbunon, A. et al. (2014) “Large Scale Sparse Optimization for Object Detection in High Resolution Images.”

Chavdar, M. et al. (2020) “Towards a system for automatic traffic sound event detection,” Telecommunications Forum [Preprint]. Available at: https://doi.org/10.1109/TELFOR51502.2020.9306592.

Choi, I. et al. (2016) “Dnn-based Sound Event Detection With Exemplar-based Approach for Noise Reduction.”

Clark, C.N. et al. (2023) “Investigating the Resolution-Performance Trade-off of Object Detection Models in Support of the Sustainable Development Goals,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 16, pp. 5695–5713. Available at: https://doi.org/10.1109/JSTARS.2023.3284309.

Dagli, R. and Shaikh, A.M. (2021) “CPPE-5: Medical Personal Protective Equipment Dataset.”

Elizalde, B. et al. (2016) “An approach for self-training audio event detectors using web data,” European Signal Processing Conference, 2017-January, pp. 1863–1867. Available at: https://doi.org/10.23919/EUSIPCO.2017.8081532.

Fang, Y. et al. (2022) “EVA: Exploring the Limits of Masked Visual Representation Learning at Scale.”

Foggia, P. et al. (2015) “Reliable detection of audio events in highly noisy environments,” Pattern Recognition Letters, 65, pp. 22–28. Available at: https://doi.org/10.1016/J.PATREC.2015.06.026.

Gemmeke, J.F. et al. (2013) “An exemplar-based NMF approach to audio event detection,” IEEE Workshop on Applications of Signal Processing to Audio and Acoustics [Preprint]. Available at: https://doi.org/10.1109/WASPAA.2013.6701847.

Ghiasi, G. et al. (2020) “Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation.”

Hanus, S. and Gleissner, F. (2008) “Co-channel and Adjacent Channel Interference Measurement of UMTS and GSM/EDGE Systems in 900 MHz Radio Band.”

Howard, A. and Padgett, C. (2003) “An adaptive learning methodology for intelligent object detection in novel imagery data,” Neurocomputing, 51, pp. 1–11. Available at: https://doi.org/10.1016/S0925-2312(02)00598-2.

Jadhav, R. et al. (2022) “Drone Based Object Detection using AI,” 2022 International Conference on Signal and Information Processing, IConSIP 2022 [Preprint]. Available at: https://doi.org/10.1109/ICONSIP49665.2022.10007476.

Jain, A. et al. (2021) “AI-Enabled Object Detection in UAVs: Challenges, Design Choices, and Research Directions,” IEEE Network, 35(4), pp. 129–135. Available at: https://doi.org/10.1109/MNET.011.2000643.

Jensen, R. (1977) “900-MHz mobile radio propagation in the Copenhagen area,” IEEE Transactions on Vehicular Technology, VT-26(4), pp. 328–331. Available at: https://doi.org/10.1109/T-VT.1977.23702.

Kidwell, P. and Boakye, K. (2015) “Object Detection in Low Resolution Overhead Imagery,” 2015 IEEE Winter Applications and Computer Vision Workshops, pp. 21–27. Available at: https://doi.org/10.1109/WACVW.2015.16.

Küçükbay, S.E. and Sert, M. (2015) “Audio-based event detection in office live environments using optimized MFCC-SVM approach,” Proceedings of the 2015 IEEE 9th International Conference on Semantic Computing (IEEE ICSC 2015), pp. 475–480. Available at: https://doi.org/10.1109/ICOSC.2015.7050855.

Li, B. et al. (2022) “Analysis of Automotive Camera Sensor Noise Factors and Impact on Object Detection,” IEEE Sensors Journal, 22(22), pp. 22210–22219. Available at: https://doi.org/10.1109/JSEN.2022.3211406.

Li, C. et al. (2023) “YOLOv6 v3.0: A Full-Scale Reloading.”

Lu, Y. and Javidi, T. (2015) “Efficient object detection for high resolution images,” Allerton Conference on Communication, Control, and Computing, pp. 1091–1098. Available at: https://doi.org/10.1109/ALLERTON.2015.7447130.

Meinedo, H. and Neto, J. (2005) “A stream-based audio segmentation, classification and clustering pre-processing system for broadcast news using ANN models,” Interspeech, pp. 237–240. Available at: https://doi.org/10.21437/INTERSPEECH.2005-117.

Mesaros, A. et al. (2010) “Acoustic event detection in real life recordings,” European Signal Processing Conference [Preprint].

Momeny, M. et al. (2021) “A noise robust convolutional neural network for image classification.” Available at: https://doi.org/10.1016/j.rineng.2021.100225.

Nazaré, T.S. et al. (2018) “Deep convolutional neural networks and noisy images,” Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 10657 LNCS, pp. 416–424. Available at: https://doi.org/10.1007/978-3-319-75193-1_50/TABLES/3.

Okamoto, D. et al. (2016) “Performance evaluation of digital TV and LTE systems operating in the 700 MHz band under the effect of mutual interference,” Journal of Microwaves, Optoelectronics and Electromagnetic Applications, 15(4), pp. 441–456. Available at: https://doi.org/10.1590/2179-10742016V15I4831.

Paranhos Da Costa, G.B. et al. (2016) “An empirical study on the effects of different types of noise in image classification tasks.” Available at: https://ruiminpan.wordpress.com/2016/03/10/ (Accessed: April 25, 2024).

Park, D., Ramanan, D. and Fowlkes, C. (2010) “Multiresolution models for object detection,” Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 6314 LNCS(PART 4), pp. 241–254. Available at: https://doi.org/10.1007/978-3-642-15561-1_18.

Pava, D. and Rhodes, W. (2011) “Object Detection and Motion Analysis in a Low Resolution 3-D Model.”

-Rodríguez, R. et al. (2024) “The Impact of Noise and Brightness on Object Detection Methods,” Sensors 2024, Vol. 24, Page 821, 24(3), p. 821. Available at: https://doi.org/10.3390/S24030821.

Shinya, Y. (2021) “USB: Universal-Scale Object Detection Benchmark.”

de Sousa Chaves, F. and Ruismaki, R. (2014) “LTE 700 MHz: Evaluation of the Probability of Interference to Digital TV,” IEEE Vehicular Technology Conference [Preprint]. Available at: https://doi.org/10.1109/VTCFALL.2014.6966043.

Surampudi, N., Srirangan, M. and Christopher, J. (2019) “Enhanced Feature Extraction Approaches for Detection of Sound Events,” IEEE International Advance Computing Conference, pp. 223–229. Available at: https://doi.org/10.1109/IACC48062.2019.8971574.

Tu, Z. et al. (2022) “MaxViT: Multi-Axis Vision Transformer.”

Uzawa, H. et al. (2021) “High-definition object detection technology based on AI inference scheme and its implementation,” IEICE Electronics Express, 18(22). Available at: https://doi.org/10.1587/ELEX.18.20210323.

Uzkent, B., Yeh, C. and Ermon, S. (2019) “Efficient Object Detection in Large Images Using Deep Reinforcement Learning,” IEEE Workshop/Winter Conference on Applications of Computer Vision, pp. 1813–1822. Available at: https://doi.org/10.1109/WACV45572.2020.9093447.

Valenti, M. et al. (2017) “A neural network approach for sound event detection in real life audio,” European Signal Processing Conference, 2017-January, pp. 2754–2758. Available at: https://doi.org/10.23919/EUSIPCO.2017.8081712.

Veerakumar, T., Esakkirajan, S. and Vennila, I. (2011) “Salt and pepper noise removal in video using adaptive decision based median filter,” 2011 International Conference on Multimedia, Signal Processing and Communication Technologies, IMPACT 2011, pp. 87–90. Available at: https://doi.org/10.1109/MSPCT.2011.6150444.

Wang, W. et al. (2022) “InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions.”

Wang, Z. et al. (2018) “A variable resolution feedback improving the performances of object detection and recognition,” Proceedings of the Institution of Mechanical Engineers, Part I: Journal of Systems and Control Engineering, 232(4), pp. 417–427. Available at: https://doi.org/10.1177/0959651817721404.

Xia, X. et al. (2020) “Sound Event Detection Using Multiple Optimized Kernels,” IEEE/ACM Transactions on Audio Speech and Language Processing, 28, pp. 1745–1754. Available at: https://doi.org/10.1109/TASLP.2020.2998298.

Zhang, W., Zelinsky, G. and Samaras, D. (2007) “Real-time Accurate Object Detection using Multiple Resolutions,” IEEE International Conference on Computer Vision [Preprint]. Available at: https://doi.org/10.1109/ICCV.2007.4409057.

Zhuang, X. et al. (2010) “Real-world acoustic event detection,” Pattern Recognition Letters, 31(12), pp. 1543–1551. Available at: https://doi.org/10.1016/J.PATREC.2010.02.005.

Zong, Z., Song, G. and Liu, Y. (2022) “DETRs with Collaborative Hybrid Assignments Training.”

## Supplementary data

Figure s1. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s2. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s3. precision_over_recall, fasterrcnn, pre-trained=False, epochs=50, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-duomo-valid-images_precision_over_recall.png)  
Figure s4. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s5. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s6. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s7. precision_over_recall, fasterrcnn, pre-trained=False, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s8. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s9. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s10. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s11. precision_over_recall, fasterrcnn, pre-trained=False, epochs=50, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s12. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s13. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s14. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s15. precision_over_recall, fasterrcnn, pre-trained=False, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s16. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s17. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s18. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s19. precision_over_recall, fasterrcnn, pre-trained=False, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s20. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_50_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s21. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s22. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s23. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s24. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s25. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s26. precision_over_recall, fasterrcnn, pre-trained=False, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s27. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s28. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s29. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s30. precision_over_recall, fasterrcnn, pre-trained=False, epochs=100, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s31. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s32. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s33. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s34. precision_over_recall, fasterrcnn, pre-trained=False, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s35. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s36. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s37. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s38. precision_over_recall, fasterrcnn, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s39. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_100_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s40. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s41. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s42. precision_over_recall, fasterrcnn, pre-trained=False, epochs=150, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-duomo-valid-images_precision_over_recall.png)  
Figure s43. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s44. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s45. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s46. precision_over_recall, fasterrcnn, pre-trained=False, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s47. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s48. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s49. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s50. precision_over_recall, fasterrcnn, pre-trained=False, epochs=150, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s51. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s52. f1_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s53. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s54. precision_over_recall, fasterrcnn, pre-trained=False, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s55. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s56. over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s57. precision_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s58. precision_over_recall, fasterrcnn, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s59. recall_over_confidence, fasterrcnn, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_False_150_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s60. confidence, fasterrcnn, pre-trained=True, epochs=0, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s61. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s62. precision_over_recall, fasterrcnn, pre-trained=True, epochs=0, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-duomo-valid-images_precision_over_recall.png)  
Figure s63. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s64. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s65. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s66. precision_over_recall, fasterrcnn, pre-trained=True, epochs=0, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s67. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s68. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s69. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s70. precision_over_recall, fasterrcnn, pre-trained=True, epochs=0, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s71. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s72. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s73. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s74. precision_over_recall, fasterrcnn, pre-trained=True, epochs=0, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s75. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s76. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s77. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s78. precision_over_recall, fasterrcnn, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s79. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_0_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s80. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s81. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s82. precision_over_recall, fasterrcnn, pre-trained=True, epochs=50, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-duomo-valid-images_precision_over_recall.png)  
Figure s83. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s84. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s85. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s86. precision_over_recall, fasterrcnn, pre-trained=True, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s87. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s88. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s89. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s90. precision_over_recall, fasterrcnn, pre-trained=True, epochs=50, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s91. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s92. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s93. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s94. precision_over_recall, fasterrcnn, pre-trained=True, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s95. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s96. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s97. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s98. precision_over_recall, fasterrcnn, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s99. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_50_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s100. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s101. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s102. precision_over_recall, fasterrcnn, pre-trained=True, epochs=100, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-duomo-valid-images_precision_over_recall.png)  
Figure s103. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s104. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s105. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s106. precision_over_recall, fasterrcnn, pre-trained=True, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s107. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s108. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s109. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s110. precision_over_recall, fasterrcnn, pre-trained=True, epochs=100, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s111. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s112. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s113. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s114. precision_over_recall, fasterrcnn, pre-trained=True, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s115. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s116. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s117. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s118. precision_over_recall, fasterrcnn, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s119. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_100_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s120. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s121. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s122. precision_over_recall, fasterrcnn, pre-trained=True, epochs=150, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-duomo-valid-images_precision_over_recall.png)  
Figure s123. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=duomo  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s124. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s125. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s126. precision_over_recall, fasterrcnn, pre-trained=True, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s127. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s128. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s129. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s130. precision_over_recall, fasterrcnn, pre-trained=True, epochs=150, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s131. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=keskvaljak  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s132. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s133. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s134. precision_over_recall, fasterrcnn, pre-trained=True, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s135. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s136. f1_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s137. precision_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s138. precision_over_recall, fasterrcnn, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s139. recall_over_confidence, fasterrcnn, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/fasterrcnn_resnet50_fpn_True_150_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s140. f1_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s141. precision_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s142. precision_over_recall, maskrcnn, pre-trained=False, epochs=50, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-duomo-valid-images_precision_over_recall.png)  
Figure s143. recall_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s144. f1_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s145. precision_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s146. precision_over_recall, maskrcnn, pre-trained=False, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s147. recall_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s148. f1_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s149. precision_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s150. precision_over_recall, maskrcnn, pre-trained=False, epochs=50, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s151. recall_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s152. f1_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s153. precision_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s154. precision_over_recall, maskrcnn, pre-trained=False, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s155. recall_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s156. f1_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s157. precision_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s158. precision_over_recall, maskrcnn, pre-trained=False, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s159. recall_over_confidence, maskrcnn, pre-trained=False, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_False_50_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s160. f1_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s161. precision_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s162. precision_over_recall, maskrcnn, pre-trained=False, epochs=100, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-duomo-valid-images_precision_over_recall.png)  
Figure s163. recall_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s164. f1_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s165. precision_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s166. precision_over_recall, maskrcnn, pre-trained=False, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s167. recall_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s168. f1_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s169. precision_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s170. precision_over_recall, maskrcnn, pre-trained=False, epochs=100, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s171. recall_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s172. f1_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s173. precision_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s174. precision_over_recall, maskrcnn, pre-trained=False, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s175. recall_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s176. f1_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s177. precision_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s178. precision_over_recall, maskrcnn, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s179. recall_over_confidence, maskrcnn, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_False_100_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s180. f1_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s181. precision_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s182. precision_over_recall, maskrcnn, pre-trained=False, epochs=150, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-duomo-valid-images_precision_over_recall.png)  
Figure s183. recall_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s184. f1_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s185. precision_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s186. precision_over_recall, maskrcnn, pre-trained=False, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s187. recall_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s188. f1_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s189. precision_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s190. precision_over_recall, maskrcnn, pre-trained=False, epochs=150, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s191. recall_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s192. f1_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s193. precision_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s194. precision_over_recall, maskrcnn, pre-trained=False, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s195. recall_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s196. f1_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s197. precision_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s198. precision_over_recall, maskrcnn, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s199. recall_over_confidence, maskrcnn, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_False_150_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s200. f1_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s201. precision_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s202. precision_over_recall, maskrcnn, pre-trained=True, epochs=0, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-duomo-valid-images_precision_over_recall.png)  
Figure s203. recall_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s204. f1_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s205. precision_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s206. precision_over_recall, maskrcnn, pre-trained=True, epochs=0, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s207. recall_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s208. f1_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s209. precision_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s210. precision_over_recall, maskrcnn, pre-trained=True, epochs=0, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s211. recall_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s212. f1_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s213. precision_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s214. precision_over_recall, maskrcnn, pre-trained=True, epochs=0, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s215. recall_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s216. f1_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s217. precision_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s218. precision_over_recall, maskrcnn, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s219. recall_over_confidence, maskrcnn, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_0_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s220. f1_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s221. precision_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s222. precision_over_recall, maskrcnn, pre-trained=True, epochs=50, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-duomo-valid-images_precision_over_recall.png)  
Figure s223. recall_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s224. f1_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s225. precision_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s226. precision_over_recall, maskrcnn, pre-trained=True, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s227. recall_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s228. f1_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s229. precision_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s230. precision_over_recall, maskrcnn, pre-trained=True, epochs=50, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s231. recall_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s232. f1_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s233. precision_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s234. precision_over_recall, maskrcnn, pre-trained=True, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s235. recall_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s236. f1_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s237. precision_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s238. precision_over_recall, maskrcnn, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s239. recall_over_confidence, maskrcnn, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_50_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s240. f1_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s241. precision_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s242. precision_over_recall, maskrcnn, pre-trained=True, epochs=100, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-duomo-valid-images_precision_over_recall.png)  
Figure s243. recall_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s244. f1_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s245. precision_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s246. precision_over_recall, maskrcnn, pre-trained=True, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s247. recall_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s248. f1_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s249. precision_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s250. precision_over_recall, maskrcnn, pre-trained=True, epochs=100, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s251. recall_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s252. f1_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s253. precision_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s254. precision_over_recall, maskrcnn, pre-trained=True, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s255. recall_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s256. f1_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s257. precision_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s258. precision_over_recall, maskrcnn, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s259. recall_over_confidence, maskrcnn, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_100_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s260. f1_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s261. precision_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s262. precision_over_recall, maskrcnn, pre-trained=True, epochs=150, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-duomo-valid-images_precision_over_recall.png)  
Figure s263. recall_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=duomo  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s264. confidence, maskrcnn, pre-trained=True, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s265. precision_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s266. precision_over_recall, maskrcnn, pre-trained=True, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s267. recall_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s268. f1_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s269. precision_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s270. precision_over_recall, maskrcnn, pre-trained=True, epochs=150, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s271. recall_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=keskvaljak  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s272. f1_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s273. precision_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s274. precision_over_recall, maskrcnn, pre-trained=True, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s275. recall_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s276. f1_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s277. precision_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s278. precision_over_recall, maskrcnn, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s279. recall_over_confidence, maskrcnn, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/maskrcnn_resnet50_fpn_True_150_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s280. f1_over_confidence, retinanet, pre-trained=True, epochs=0, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s281. precision_over_confidence, retinanet, pre-trained=True, epochs=0, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s282. precision_over_recall, retinanet, pre-trained=True, epochs=0, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-duomo-valid-images_precision_over_recall.png)  
Figure s283. recall_over_confidence, retinanet, pre-trained=True, epochs=0, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s284. f1_over_confidence, retinanet, pre-trained=True, epochs=0, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s285. precision_over_confidence, retinanet, pre-trained=True, epochs=0, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s286. precision_over_recall, retinanet, pre-trained=True, epochs=0, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s287. recall_over_confidence, retinanet, pre-trained=True, epochs=0, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s288. f1_over_confidence, retinanet, pre-trained=True, epochs=0, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s289. precision_over_confidence, retinanet, pre-trained=True, epochs=0, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s290. precision_over_recall, retinanet, pre-trained=True, epochs=0, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s291. recall_over_confidence, retinanet, pre-trained=True, epochs=0, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s292. f1_over_confidence, retinanet, pre-trained=True, epochs=0, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s293. precision_over_confidence, retinanet, pre-trained=True, epochs=0, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s294. precision_over_recall, retinanet, pre-trained=True, epochs=0, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s295. recall_over_confidence, retinanet, pre-trained=True, epochs=0, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s296. f1_over_confidence, retinanet, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s297. precision_over_confidence, retinanet, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s298. precision_over_recall, retinanet, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s299. recall_over_confidence, retinanet, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_0_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s300. f1_over_confidence, retinanet, pre-trained=True, epochs=50, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s301. precision_over_confidence, retinanet, pre-trained=True, epochs=50, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s302. precision_over_recall, retinanet, pre-trained=True, epochs=50, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-duomo-valid-images_precision_over_recall.png)  
Figure s303. recall_over_confidence, retinanet, pre-trained=True, epochs=50, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s304. f1_over_confidence, retinanet, pre-trained=True, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s305. precision_over_confidence, retinanet, pre-trained=True, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s306. precision_over_recall, retinanet, pre-trained=True, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s307. recall_over_confidence, retinanet, pre-trained=True, epochs=50, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s308. f1_over_confidence, retinanet, pre-trained=True, epochs=50, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s309. precision_over_confidence, retinanet, pre-trained=True, epochs=50, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s310. precision_over_recall, retinanet, pre-trained=True, epochs=50, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s311. recall_over_confidence, retinanet, pre-trained=True, epochs=50, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s312. f1_over_confidence, retinanet, pre-trained=True, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s313. precision_over_confidence, retinanet, pre-trained=True, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s314. precision_over_recall, retinanet, pre-trained=True, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s315. recall_over_confidence, retinanet, pre-trained=True, epochs=50, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s316. f1_over_confidence, retinanet, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s317. precision_over_confidence, retinanet, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s318. precision_over_recall, retinanet, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s319. recall_over_confidence, retinanet, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_50_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s320. f1_over_confidence, retinanet, pre-trained=True, epochs=100, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s321. precision_over_confidence, retinanet, pre-trained=True, epochs=100, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s322. precision_over_recall, retinanet, pre-trained=True, epochs=100, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-duomo-valid-images_precision_over_recall.png)  
Figure s323. recall_over_confidence, retinanet, pre-trained=True, epochs=100, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s324. f1_over_confidence, retinanet, pre-trained=True, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s325. precision_over_confidence, retinanet, pre-trained=True, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s326. precision_over_recall, retinanet, pre-trained=True, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s327. recall_over_confidence, retinanet, pre-trained=True, epochs=100, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s328. f1_over_confidence, retinanet, pre-trained=True, epochs=100, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s329. precision_over_confidence, retinanet, pre-trained=True, epochs=100, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s330. precision_over_recall, retinanet, pre-trained=True, epochs=100, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s331. recall_over_confidence, retinanet, pre-trained=True, epochs=100, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s332. f1_over_confidence, retinanet, pre-trained=True, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s333. precision_over_confidence, retinanet, pre-trained=True, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s334. precision_over_recall, retinanet, pre-trained=True, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s335. recall_over_confidence, retinanet, pre-trained=True, epochs=100, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s336. f1_over_confidence, retinanet, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s337. precision_over_confidence, retinanet, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s338. precision_over_recall, retinanet, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s339. recall_over_confidence, retinanet, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_100_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s340. f1_over_confidence, retinanet, pre-trained=True, epochs=150, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-duomo-valid-images_f1_over_confidence.png)  
Figure s341. precision_over_confidence, retinanet, pre-trained=True, epochs=150, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-duomo-valid-images_precision_over_confidence.png)  
Figure s342. precision_over_recall, retinanet, pre-trained=True, epochs=150, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-duomo-valid-images_precision_over_recall.png)  
Figure s343. recall_over_confidence, retinanet, pre-trained=True, epochs=150, data=duomo  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-duomo-valid-images_recall_over_confidence.png)  
Figure s344. f1_over_confidence, retinanet, pre-trained=True, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_f1_over_confidence.png)  
Figure s345. precision_over_confidence, retinanet, pre-trained=True, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_precision_over_confidence.png)  
Figure s346. precision_over_recall, retinanet, pre-trained=True, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_precision_over_recall.png)  
Figure s347. recall_over_confidence, retinanet, pre-trained=True, epochs=150, data=hadji_dimitar_square  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-hadji_dimitar_square-valid-images_recall_over_confidence.png)  
Figure s348. f1_over_confidence, retinanet, pre-trained=True, epochs=150, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-keskvaljak-valid-images_f1_over_confidence.png)  
Figure s349. precision_over_confidence, retinanet, pre-trained=True, epochs=150, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-keskvaljak-valid-images_precision_over_confidence.png)  
Figure s350. precision_over_recall, retinanet, pre-trained=True, epochs=150, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-keskvaljak-valid-images_precision_over_recall.png)  
Figure s351. recall_over_confidence, retinanet, pre-trained=True, epochs=150, data=keskvaljak  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-keskvaljak-valid-images_recall_over_confidence.png)  
Figure s352. f1_over_confidence, retinanet, pre-trained=True, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-kielce_university_of_technology-valid-images_f1_over_confidence.png)  
Figure s353. precision_over_confidence, retinanet, pre-trained=True, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-kielce_university_of_technology-valid-images_precision_over_confidence.png)  
Figure s354. precision_over_recall, retinanet, pre-trained=True, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-kielce_university_of_technology-valid-images_precision_over_recall.png)  
Figure s355. recall_over_confidence, retinanet, pre-trained=True, epochs=150, data=kielce_university_of_technology  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-kielce_university_of_technology-valid-images_recall_over_confidence.png)  
Figure s356. f1_over_confidence, retinanet, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-toggenburg_alpaca_ranch-valid-images_f1_over_confidence.png)  
Figure s357. precision_over_confidence, retinanet, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-toggenburg_alpaca_ranch-valid-images_precision_over_confidence.png)  
Figure s358. precision_over_recall, retinanet, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-toggenburg_alpaca_ranch-valid-images_precision_over_recall.png)  
Figure s359. recall_over_confidence, retinanet, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch  
![](./saved_runs/retinanet_resnet50_fpn_True_150_-data-toggenburg_alpaca_ranch-valid-images_recall_over_confidence.png)  
Figure s360. f1_over_confidence, yolov8n, pre-trained=True, epochs=50, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train/F1_curve.png)  
Figure s361. precision_over_recall, yolov8n, pre-trained=True, epochs=50, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train/PR_curve.png)  
Figure s362. precision_over_confidence, yolov8n, pre-trained=True, epochs=50, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train/P_curve.png)  
Figure s363. recall_over_confidence, yolov8n, pre-trained=True, epochs=50, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train/R_curve.png)  
Figure s364. confusion_matrix, yolov8n, pre-trained=True, epochs=50, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train/confusion_matrix.png)  
Figure s365. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=50, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train/confusion_matrix_normalized.png)  
Figure s366. f1_over_confidence, yolov8n, pre-trained=True, epochs=100, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train2/F1_curve.png)  
Figure s367. precision_over_recall, yolov8n, pre-trained=True, epochs=100, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train2/PR_curve.png)  
Figure s368. precision_over_confidence, yolov8n, pre-trained=True, epochs=100, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train2/P_curve.png)  
Figure s369. recall_over_confidence, yolov8n, pre-trained=True, epochs=100, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train2/R_curve.png)  
Figure s370. confusion_matrix, yolov8n, pre-trained=True, epochs=100, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train2/confusion_matrix.png)  
Figure s371. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=100, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train2/confusion_matrix_normalized.png)  
Figure s372. f1_over_confidence, yolov8n, pre-trained=True, epochs=150, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train3/F1_curve.png)  
Figure s373. precision_over_recall, yolov8n, pre-trained=True, epochs=150, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train3/PR_curve.png)  
Figure s374. precision_over_confidence, yolov8n, pre-trained=True, epochs=150, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train3/P_curve.png)  
Figure s375. recall_over_confidence, yolov8n, pre-trained=True, epochs=150, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train3/R_curve.png)  
Figure s376. confusion_matrix, yolov8n, pre-trained=True, epochs=150, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train3/confusion_matrix.png)  
Figure s377. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=150, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train3/confusion_matrix_normalized.png)  
Figure s378. f1_over_confidence, yolov8n, pre-trained=True, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train4/F1_curve.png)  
Figure s379. precision_over_recall, yolov8n, pre-trained=True, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train4/PR_curve.png)  
Figure s380. precision_over_confidence, yolov8n, pre-trained=True, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train4/P_curve.png)  
Figure s381. recall_over_confidence, yolov8n, pre-trained=True, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train4/R_curve.png)  
Figure s382. confusion_matrix, yolov8n, pre-trained=True, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train4/confusion_matrix.png)  
Figure s383. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train4/confusion_matrix_normalized.png)  
Figure s384. f1_over_confidence, yolov8n, pre-trained=True, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train5/F1_curve.png)  
Figure s385. precision_over_recall, yolov8n, pre-trained=True, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train5/PR_curve.png)  
Figure s386. precision_over_confidence, yolov8n, pre-trained=True, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train5/P_curve.png)  
Figure s387. recall_over_confidence, yolov8n, pre-trained=True, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train5/R_curve.png)  
Figure s388. confusion_matrix, yolov8n, pre-trained=True, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train5/confusion_matrix.png)  
Figure s389. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train5/confusion_matrix_normalized.png)  
Figure s390. f1_over_confidence, yolov8n, pre-trained=True, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train6/F1_curve.png)  
Figure s391. precision_over_recall, yolov8n, pre-trained=True, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train6/PR_curve.png)  
Figure s392. precision_over_confidence, yolov8n, pre-trained=True, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train6/P_curve.png)  
Figure s393. recall_over_confidence, yolov8n, pre-trained=True, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train6/R_curve.png)  
Figure s394. confusion_matrix, yolov8n, pre-trained=True, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train6/confusion_matrix.png)  
Figure s395. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train6/confusion_matrix_normalized.png)  
Figure s396. f1_over_confidence, yolov8n, pre-trained=True, epochs=50, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train7/F1_curve.png)  
Figure s397. precision_over_recall, yolov8n, pre-trained=True, epochs=50, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train7/PR_curve.png)  
Figure s398. precision_over_confidence, yolov8n, pre-trained=True, epochs=50, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train7/P_curve.png)  
Figure s399. recall_over_confidence, yolov8n, pre-trained=True, epochs=50, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train7/R_curve.png)  
Figure s400. confusion_matrix, yolov8n, pre-trained=True, epochs=50, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train7/confusion_matrix.png)  
Figure s401. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=50, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train7/confusion_matrix_normalized.png)  
Figure s402. f1_over_confidence, yolov8n, pre-trained=True, epochs=100, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train8/F1_curve.png)  
Figure s403. precision_over_recall, yolov8n, pre-trained=True, epochs=100, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train8/PR_curve.png)  
Figure s404. precision_over_confidence, yolov8n, pre-trained=True, epochs=100, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train8/P_curve.png)  
Figure s405. recall_over_confidence, yolov8n, pre-trained=True, epochs=100, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train8/R_curve.png)  
Figure s406. confusion_matrix, yolov8n, pre-trained=True, epochs=100, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train8/confusion_matrix.png)  
Figure s407. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=100, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train8/confusion_matrix_normalized.png)  
Figure s408. f1_over_confidence, yolov8n, pre-trained=True, epochs=150, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train9/F1_curve.png)  
Figure s409. precision_over_recall, yolov8n, pre-trained=True, epochs=150, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train9/PR_curve.png)  
Figure s410. precision_over_confidence, yolov8n, pre-trained=True, epochs=150, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train9/P_curve.png)  
Figure s411. recall_over_confidence, yolov8n, pre-trained=True, epochs=150, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train9/R_curve.png)  
Figure s412. confusion_matrix, yolov8n, pre-trained=True, epochs=150, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train9/confusion_matrix.png)  
Figure s413. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=150, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train9/confusion_matrix_normalized.png)  
Figure s414. f1_over_confidence, yolov8n, pre-trained=True, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train10/F1_curve.png)  
Figure s415. precision_over_recall, yolov8n, pre-trained=True, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train10/PR_curve.png)  
Figure s416. precision_over_confidence, yolov8n, pre-trained=True, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train10/P_curve.png)  
Figure s417. recall_over_confidence, yolov8n, pre-trained=True, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train10/R_curve.png)  
Figure s418. confusion_matrix, yolov8n, pre-trained=True, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train10/confusion_matrix.png)  
Figure s419. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train10/confusion_matrix_normalized.png)  
Figure s420. f1_over_confidence, yolov8n, pre-trained=True, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train11/F1_curve.png)  
Figure s421. precision_over_recall, yolov8n, pre-trained=True, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train11/PR_curve.png)  
Figure s422. precision_over_confidence, yolov8n, pre-trained=True, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train11/P_curve.png)  
Figure s423. recall_over_confidence, yolov8n, pre-trained=True, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train11/R_curve.png)  
Figure s424. confusion_matrix, yolov8n, pre-trained=True, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train11/confusion_matrix.png)  
Figure s425. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train11/confusion_matrix_normalized.png)  
Figure s426. f1_over_confidence, yolov8n, pre-trained=True, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train12/F1_curve.png)  
Figure s427. precision_over_recall, yolov8n, pre-trained=True, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train12/PR_curve.png)  
Figure s428. precision_over_confidence, yolov8n, pre-trained=True, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train12/P_curve.png)  
Figure s429. recall_over_confidence, yolov8n, pre-trained=True, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train12/R_curve.png)  
Figure s430. confusion_matrix, yolov8n, pre-trained=True, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train12/confusion_matrix.png)  
Figure s431. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train12/confusion_matrix_normalized.png)  
Figure s432. f1_over_confidence, yolov8n, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train13/F1_curve.png)  
Figure s433. precision_over_recall, yolov8n, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train13/PR_curve.png)  
Figure s434. precision_over_confidence, yolov8n, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train13/P_curve.png)  
Figure s435. recall_over_confidence, yolov8n, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train13/R_curve.png)  
Figure s436. confusion_matrix, yolov8n, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train13/confusion_matrix.png)  
Figure s437. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train13/confusion_matrix_normalized.png)  
Figure s438. f1_over_confidence, yolov8n, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train15/F1_curve.png)  
Figure s439. precision_over_recall, yolov8n, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train15/PR_curve.png)  
Figure s440. precision_over_confidence, yolov8n, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train15/P_curve.png)  
Figure s441. recall_over_confidence, yolov8n, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train15/R_curve.png)  
Figure s442. confusion_matrix, yolov8n, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train15/confusion_matrix.png)  
Figure s443. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train15/confusion_matrix_normalized.png)  
Figure s444. f1_over_confidence, yolov8n, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train16/F1_curve.png)  
Figure s445. precision_over_recall, yolov8n, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train16/PR_curve.png)  
Figure s446. precision_over_confidence, yolov8n, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train16/P_curve.png)  
Figure s447. recall_over_confidence, yolov8n, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train16/R_curve.png)  
Figure s448. confusion_matrix, yolov8n, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train16/confusion_matrix.png)  
Figure s449. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train16/confusion_matrix_normalized.png)  
Figure s450. f1_over_confidence, yolov8n, pre-trained=False, epochs=50, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train17/F1_curve.png)  
Figure s451. precision_over_recall, yolov8n, pre-trained=False, epochs=50, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train17/PR_curve.png)  
Figure s452. precision_over_confidence, yolov8n, pre-trained=False, epochs=50, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train17/P_curve.png)  
Figure s453. recall_over_confidence, yolov8n, pre-trained=False, epochs=50, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train17/R_curve.png)  
Figure s454. confusion_matrix, yolov8n, pre-trained=False, epochs=50, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train17/confusion_matrix.png)  
Figure s455. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=50, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train17/confusion_matrix_normalized.png)  
Figure s456. f1_over_confidence, yolov8n, pre-trained=False, epochs=100, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train18/F1_curve.png)  
Figure s457. precision_over_recall, yolov8n, pre-trained=False, epochs=100, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train18/PR_curve.png)  
Figure s458. precision_over_confidence, yolov8n, pre-trained=False, epochs=100, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train18/P_curve.png)  
Figure s459. recall_over_confidence, yolov8n, pre-trained=False, epochs=100, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train18/R_curve.png)  
Figure s460. confusion_matrix, yolov8n, pre-trained=False, epochs=100, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train18/confusion_matrix.png)  
Figure s461. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=100, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train18/confusion_matrix_normalized.png)  
Figure s462. f1_over_confidence, yolov8n, pre-trained=False, epochs=150, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train19/F1_curve.png)  
Figure s463. precision_over_recall, yolov8n, pre-trained=False, epochs=150, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train19/PR_curve.png)  
Figure s464. precision_over_confidence, yolov8n, pre-trained=False, epochs=150, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train19/P_curve.png)  
Figure s465. recall_over_confidence, yolov8n, pre-trained=False, epochs=150, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train19/R_curve.png)  
Figure s466. confusion_matrix, yolov8n, pre-trained=False, epochs=150, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train19/confusion_matrix.png)  
Figure s467. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=150, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/train19/confusion_matrix_normalized.png)  
Figure s468. f1_over_confidence, yolov8n, pre-trained=False, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train20/F1_curve.png)  
Figure s469. precision_over_recall, yolov8n, pre-trained=False, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train20/PR_curve.png)  
Figure s470. precision_over_confidence, yolov8n, pre-trained=False, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train20/P_curve.png)  
Figure s471. recall_over_confidence, yolov8n, pre-trained=False, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train20/R_curve.png)  
Figure s472. confusion_matrix, yolov8n, pre-trained=False, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train20/confusion_matrix.png)  
Figure s473. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train20/confusion_matrix_normalized.png)  
Figure s478. confusion_matrix, yolov8n, pre-trained=False, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train21/confusion_matrix.png)  
Figure s479. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train21/confusion_matrix_normalized.png)  
Figure s480. f1_over_confidence, yolov8n, pre-trained=False, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train22/F1_curve.png)  
Figure s481. precision_over_recall, yolov8n, pre-trained=False, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train22/PR_curve.png)  
Figure s482. precision_over_confidence, yolov8n, pre-trained=False, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train22/P_curve.png)  
Figure s483. recall_over_confidence, yolov8n, pre-trained=False, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train22/R_curve.png)  
Figure s484. confusion_matrix, yolov8n, pre-trained=False, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train22/confusion_matrix.png)  
Figure s485. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/train22/confusion_matrix_normalized.png)  
Figure s490. confusion_matrix, yolov8n, pre-trained=False, epochs=50, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train23/confusion_matrix.png)  
Figure s491. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=50, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train23/confusion_matrix_normalized.png)  
Figure s496. confusion_matrix, yolov8n, pre-trained=False, epochs=100, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train24/confusion_matrix.png)  
Figure s497. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=100, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train24/confusion_matrix_normalized.png)  
Figure s498. f1_over_confidence, yolov8n, pre-trained=False, epochs=150, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train25/F1_curve.png)  
Figure s499. precision_over_recall, yolov8n, pre-trained=False, epochs=150, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train25/PR_curve.png)  
Figure s500. precision_over_confidence, yolov8n, pre-trained=False, epochs=150, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train25/P_curve.png)  
Figure s501. recall_over_confidence, yolov8n, pre-trained=False, epochs=150, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train25/R_curve.png)  
Figure s502. confusion_matrix, yolov8n, pre-trained=False, epochs=150, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train25/confusion_matrix.png)  
Figure s503. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=150, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/train25/confusion_matrix_normalized.png)  
Figure s508. confusion_matrix, yolov8n, pre-trained=False, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train26/confusion_matrix.png)  
Figure s509. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train26/confusion_matrix_normalized.png)  
Figure s514. confusion_matrix, yolov8n, pre-trained=False, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train27/confusion_matrix.png)  
Figure s515. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train27/confusion_matrix_normalized.png)  
Figure s516. f1_over_confidence, yolov8n, pre-trained=False, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train28/F1_curve.png)  
Figure s517. precision_over_recall, yolov8n, pre-trained=False, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train28/PR_curve.png)  
Figure s518. precision_over_confidence, yolov8n, pre-trained=False, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train28/P_curve.png)  
Figure s519. recall_over_confidence, yolov8n, pre-trained=False, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train28/R_curve.png)  
Figure s520. confusion_matrix, yolov8n, pre-trained=False, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train28/confusion_matrix.png)  
Figure s521. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/train28/confusion_matrix_normalized.png)  
Figure s526. confusion_matrix, yolov8n, pre-trained=False, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train29/confusion_matrix.png)  
Figure s527. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train29/confusion_matrix_normalized.png)  
Figure s528. f1_over_confidence, yolov8n, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train30/F1_curve.png)  
Figure s529. precision_over_recall, yolov8n, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train30/PR_curve.png)  
Figure s530. precision_over_confidence, yolov8n, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train30/P_curve.png)  
Figure s531. recall_over_confidence, yolov8n, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train30/R_curve.png)  
Figure s532. confusion_matrix, yolov8n, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train30/confusion_matrix.png)  
Figure s533. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train30/confusion_matrix_normalized.png)  
Figure s534. f1_over_confidence, yolov8n, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train31/F1_curve.png)  
Figure s535. precision_over_recall, yolov8n, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train31/PR_curve.png)  
Figure s536. precision_over_confidence, yolov8n, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train31/P_curve.png)  
Figure s537. recall_over_confidence, yolov8n, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train31/R_curve.png)  
Figure s538. confusion_matrix, yolov8n, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train31/confusion_matrix.png)  
Figure s539. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/train31/confusion_matrix_normalized.png)  
Figure s540. f1_over_confidence, yolov8n, pre-trained=True, epochs=50, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train32/F1_curve.png)  
Figure s541. precision_over_recall, yolov8n, pre-trained=True, epochs=50, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train32/PR_curve.png)  
Figure s542. precision_over_confidence, yolov8n, pre-trained=True, epochs=50, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train32/P_curve.png)  
Figure s543. recall_over_confidence, yolov8n, pre-trained=True, epochs=50, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train32/R_curve.png)  
Figure s544. confusion_matrix, yolov8n, pre-trained=True, epochs=50, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train32/confusion_matrix.png)  
Figure s545. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=50, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train32/confusion_matrix_normalized.png)  
Figure s546. f1_over_confidence, yolov8n, pre-trained=True, epochs=100, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train33/F1_curve.png)  
Figure s547. precision_over_recall, yolov8n, pre-trained=True, epochs=100, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train33/PR_curve.png)  
Figure s548. precision_over_confidence, yolov8n, pre-trained=True, epochs=100, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train33/P_curve.png)  
Figure s549. recall_over_confidence, yolov8n, pre-trained=True, epochs=100, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train33/R_curve.png)  
Figure s550. confusion_matrix, yolov8n, pre-trained=True, epochs=100, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train33/confusion_matrix.png)  
Figure s551. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=100, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train33/confusion_matrix_normalized.png)  
Figure s552. f1_over_confidence, yolov8n, pre-trained=True, epochs=150, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train34/F1_curve.png)  
Figure s553. precision_over_recall, yolov8n, pre-trained=True, epochs=150, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train34/PR_curve.png)  
Figure s554. precision_over_confidence, yolov8n, pre-trained=True, epochs=150, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train34/P_curve.png)  
Figure s555. recall_over_confidence, yolov8n, pre-trained=True, epochs=150, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train34/R_curve.png)  
Figure s556. confusion_matrix, yolov8n, pre-trained=True, epochs=150, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train34/confusion_matrix.png)  
Figure s557. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=150, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train34/confusion_matrix_normalized.png)  
Figure s558. f1_over_confidence, yolov8n, pre-trained=True, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train35/F1_curve.png)  
Figure s559. precision_over_recall, yolov8n, pre-trained=True, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train35/PR_curve.png)  
Figure s560. precision_over_confidence, yolov8n, pre-trained=True, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train35/P_curve.png)  
Figure s561. recall_over_confidence, yolov8n, pre-trained=True, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train35/R_curve.png)  
Figure s562. confusion_matrix, yolov8n, pre-trained=True, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train35/confusion_matrix.png)  
Figure s563. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train35/confusion_matrix_normalized.png)  
Figure s564. f1_over_confidence, yolov8n, pre-trained=True, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train36/F1_curve.png)  
Figure s565. precision_over_recall, yolov8n, pre-trained=True, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train36/PR_curve.png)  
Figure s566. precision_over_confidence, yolov8n, pre-trained=True, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train36/P_curve.png)  
Figure s567. recall_over_confidence, yolov8n, pre-trained=True, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train36/R_curve.png)  
Figure s568. confusion_matrix, yolov8n, pre-trained=True, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train36/confusion_matrix.png)  
Figure s569. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train36/confusion_matrix_normalized.png)  
Figure s570. f1_over_confidence, yolov8n, pre-trained=True, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train37/F1_curve.png)  
Figure s571. precision_over_recall, yolov8n, pre-trained=True, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train37/PR_curve.png)  
Figure s572. precision_over_confidence, yolov8n, pre-trained=True, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train37/P_curve.png)  
Figure s573. recall_over_confidence, yolov8n, pre-trained=True, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train37/R_curve.png)  
Figure s574. confusion_matrix, yolov8n, pre-trained=True, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train37/confusion_matrix.png)  
Figure s575. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train37/confusion_matrix_normalized.png)  
Figure s576. f1_over_confidence, yolov8n, pre-trained=True, epochs=50, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train38/F1_curve.png)  
Figure s577. precision_over_recall, yolov8n, pre-trained=True, epochs=50, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train38/PR_curve.png)  
Figure s578. precision_over_confidence, yolov8n, pre-trained=True, epochs=50, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train38/P_curve.png)  
Figure s579. recall_over_confidence, yolov8n, pre-trained=True, epochs=50, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train38/R_curve.png)  
Figure s580. confusion_matrix, yolov8n, pre-trained=True, epochs=50, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train38/confusion_matrix.png)  
Figure s581. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=50, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train38/confusion_matrix_normalized.png)  
Figure s582. f1_over_confidence, yolov8n, pre-trained=True, epochs=100, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train39/F1_curve.png)  
Figure s583. precision_over_recall, yolov8n, pre-trained=True, epochs=100, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train39/PR_curve.png)  
Figure s584. precision_over_confidence, yolov8n, pre-trained=True, epochs=100, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train39/P_curve.png)  
Figure s585. recall_over_confidence, yolov8n, pre-trained=True, epochs=100, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train39/R_curve.png)  
Figure s586. confusion_matrix, yolov8n, pre-trained=True, epochs=100, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train39/confusion_matrix.png)  
Figure s587. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=100, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train39/confusion_matrix_normalized.png)  
Figure s588. f1_over_confidence, yolov8n, pre-trained=True, epochs=150, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train40/F1_curve.png)  
Figure s589. precision_over_recall, yolov8n, pre-trained=True, epochs=150, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train40/PR_curve.png)  
Figure s590. precision_over_confidence, yolov8n, pre-trained=True, epochs=150, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train40/P_curve.png)  
Figure s591. recall_over_confidence, yolov8n, pre-trained=True, epochs=150, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train40/R_curve.png)  
Figure s592. confusion_matrix, yolov8n, pre-trained=True, epochs=150, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train40/confusion_matrix.png)  
Figure s593. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=150, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train40/confusion_matrix_normalized.png)  
Figure s594. f1_over_confidence, yolov8n, pre-trained=True, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train41/F1_curve.png)  
Figure s595. precision_over_recall, yolov8n, pre-trained=True, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train41/PR_curve.png)  
Figure s596. precision_over_confidence, yolov8n, pre-trained=True, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train41/P_curve.png)  
Figure s597. recall_over_confidence, yolov8n, pre-trained=True, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train41/R_curve.png)  
Figure s598. confusion_matrix, yolov8n, pre-trained=True, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train41/confusion_matrix.png)  
Figure s599. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train41/confusion_matrix_normalized.png)  
Figure s600. f1_over_confidence, yolov8n, pre-trained=True, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train42/F1_curve.png)  
Figure s601. precision_over_recall, yolov8n, pre-trained=True, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train42/PR_curve.png)  
Figure s602. precision_over_confidence, yolov8n, pre-trained=True, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train42/P_curve.png)  
Figure s603. recall_over_confidence, yolov8n, pre-trained=True, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train42/R_curve.png)  
Figure s604. confusion_matrix, yolov8n, pre-trained=True, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train42/confusion_matrix.png)  
Figure s605. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train42/confusion_matrix_normalized.png)  
Figure s606. f1_over_confidence, yolov8n, pre-trained=True, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train43/F1_curve.png)  
Figure s607. precision_over_recall, yolov8n, pre-trained=True, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train43/PR_curve.png)  
Figure s608. precision_over_confidence, yolov8n, pre-trained=True, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train43/P_curve.png)  
Figure s609. recall_over_confidence, yolov8n, pre-trained=True, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train43/R_curve.png)  
Figure s610. confusion_matrix, yolov8n, pre-trained=True, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train43/confusion_matrix.png)  
Figure s611. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train43/confusion_matrix_normalized.png)  
Figure s612. f1_over_confidence, yolov8n, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train44/F1_curve.png)  
Figure s613. precision_over_recall, yolov8n, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train44/PR_curve.png)  
Figure s614. precision_over_confidence, yolov8n, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train44/P_curve.png)  
Figure s615. recall_over_confidence, yolov8n, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train44/R_curve.png)  
Figure s616. confusion_matrix, yolov8n, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train44/confusion_matrix.png)  
Figure s617. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train44/confusion_matrix_normalized.png)  
Figure s618. f1_over_confidence, yolov8n, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train45/F1_curve.png)  
Figure s619. precision_over_recall, yolov8n, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train45/PR_curve.png)  
Figure s620. precision_over_confidence, yolov8n, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train45/P_curve.png)  
Figure s621. recall_over_confidence, yolov8n, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train45/R_curve.png)  
Figure s622. confusion_matrix, yolov8n, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train45/confusion_matrix.png)  
Figure s623. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train45/confusion_matrix_normalized.png)  
Figure s624. f1_over_confidence, yolov8n, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train46/F1_curve.png)  
Figure s625. precision_over_recall, yolov8n, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train46/PR_curve.png)  
Figure s626. precision_over_confidence, yolov8n, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train46/P_curve.png)  
Figure s627. recall_over_confidence, yolov8n, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train46/R_curve.png)  
Figure s628. confusion_matrix, yolov8n, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train46/confusion_matrix.png)  
Figure s629. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train46/confusion_matrix_normalized.png)  
Figure s630. f1_over_confidence, yolov8n, pre-trained=False, epochs=50, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train47/F1_curve.png)  
Figure s631. precision_over_recall, yolov8n, pre-trained=False, epochs=50, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train47/PR_curve.png)  
Figure s632. precision_over_confidence, yolov8n, pre-trained=False, epochs=50, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train47/P_curve.png)  
Figure s633. recall_over_confidence, yolov8n, pre-trained=False, epochs=50, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train47/R_curve.png)  
Figure s634. confusion_matrix, yolov8n, pre-trained=False, epochs=50, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train47/confusion_matrix.png)  
Figure s635. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=50, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train47/confusion_matrix_normalized.png)  
Figure s636. f1_over_confidence, yolov8n, pre-trained=False, epochs=100, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train48/F1_curve.png)  
Figure s637. precision_over_recall, yolov8n, pre-trained=False, epochs=100, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train48/PR_curve.png)  
Figure s638. precision_over_confidence, yolov8n, pre-trained=False, epochs=100, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train48/P_curve.png)  
Figure s639. recall_over_confidence, yolov8n, pre-trained=False, epochs=100, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train48/R_curve.png)  
Figure s640. confusion_matrix, yolov8n, pre-trained=False, epochs=100, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train48/confusion_matrix.png)  
Figure s641. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=100, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train48/confusion_matrix_normalized.png)  
Figure s642. f1_over_confidence, yolov8n, pre-trained=False, epochs=150, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train49/F1_curve.png)  
Figure s643. precision_over_recall, yolov8n, pre-trained=False, epochs=150, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train49/PR_curve.png)  
Figure s644. precision_over_confidence, yolov8n, pre-trained=False, epochs=150, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train49/P_curve.png)  
Figure s645. recall_over_confidence, yolov8n, pre-trained=False, epochs=150, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train49/R_curve.png)  
Figure s646. confusion_matrix, yolov8n, pre-trained=False, epochs=150, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train49/confusion_matrix.png)  
Figure s647. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=150, data=duomo, hyperparameter-tuned=True  
![](./saved_runs/train49/confusion_matrix_normalized.png)  
Figure s648. f1_over_confidence, yolov8n, pre-trained=False, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train50/F1_curve.png)  
Figure s649. precision_over_recall, yolov8n, pre-trained=False, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train50/PR_curve.png)  
Figure s650. precision_over_confidence, yolov8n, pre-trained=False, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train50/P_curve.png)  
Figure s651. recall_over_confidence, yolov8n, pre-trained=False, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train50/R_curve.png)  
Figure s652. confusion_matrix, yolov8n, pre-trained=False, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train50/confusion_matrix.png)  
Figure s653. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=50, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train50/confusion_matrix_normalized.png)  
Figure s658. confusion_matrix, yolov8n, pre-trained=False, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train51/confusion_matrix.png)  
Figure s659. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=100, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train51/confusion_matrix_normalized.png)  
Figure s664. confusion_matrix, yolov8n, pre-trained=False, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train52/confusion_matrix.png)  
Figure s665. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=150, data=hadji_dimitar_square, hyperparameter-tuned=True  
![](./saved_runs/train52/confusion_matrix_normalized.png)  
Figure s670. confusion_matrix, yolov8n, pre-trained=False, epochs=50, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train53/confusion_matrix.png)  
Figure s671. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=50, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train53/confusion_matrix_normalized.png)  
Figure s676. confusion_matrix, yolov8n, pre-trained=False, epochs=100, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train54/confusion_matrix.png)  
Figure s677. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=100, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train54/confusion_matrix_normalized.png)  
Figure s682. confusion_matrix, yolov8n, pre-trained=False, epochs=150, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train55/confusion_matrix.png)  
Figure s683. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=150, data=keskvaljak, hyperparameter-tuned=True  
![](./saved_runs/train55/confusion_matrix_normalized.png)  
Figure s688. confusion_matrix, yolov8n, pre-trained=False, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train56/confusion_matrix.png)  
Figure s689. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=50, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train56/confusion_matrix_normalized.png)  
Figure s694. confusion_matrix, yolov8n, pre-trained=False, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train57/confusion_matrix.png)  
Figure s695. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=100, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train57/confusion_matrix_normalized.png)  
Figure s696. f1_over_confidence, yolov8n, pre-trained=False, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train58/F1_curve.png)  
Figure s697. precision_over_recall, yolov8n, pre-trained=False, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train58/PR_curve.png)  
Figure s698. precision_over_confidence, yolov8n, pre-trained=False, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train58/P_curve.png)  
Figure s699. recall_over_confidence, yolov8n, pre-trained=False, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train58/R_curve.png)  
Figure s700. confusion_matrix, yolov8n, pre-trained=False, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train58/confusion_matrix.png)  
Figure s701. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=150, data=kielce_university_of_technology, hyperparameter-tuned=True  
![](./saved_runs/train58/confusion_matrix_normalized.png)  
Figure s706. confusion_matrix, yolov8n, pre-trained=False, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train59/confusion_matrix.png)  
Figure s707. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=50, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train59/confusion_matrix_normalized.png)  
Figure s708. f1_over_confidence, yolov8n, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train60/F1_curve.png)  
Figure s709. precision_over_recall, yolov8n, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train60/PR_curve.png)  
Figure s710. precision_over_confidence, yolov8n, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train60/P_curve.png)  
Figure s711. recall_over_confidence, yolov8n, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train60/R_curve.png)  
Figure s712. confusion_matrix, yolov8n, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train60/confusion_matrix.png)  
Figure s713. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=100, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train60/confusion_matrix_normalized.png)  
Figure s714. f1_over_confidence, yolov8n, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train61/F1_curve.png)  
Figure s715. precision_over_recall, yolov8n, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train61/PR_curve.png)  
Figure s716. precision_over_confidence, yolov8n, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train61/P_curve.png)  
Figure s717. recall_over_confidence, yolov8n, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train61/R_curve.png)  
Figure s718. confusion_matrix, yolov8n, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train61/confusion_matrix.png)  
Figure s719. confusion_matrix_normalized, yolov8n, pre-trained=False, epochs=150, data=toggenburg_alpaca_ranch, hyperparameter-tuned=True  
![](./saved_runs/train61/confusion_matrix_normalized.png)  
Figure s720. f1_over_confidence, yolov8n, pre-trained=True, epochs=0, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/val2/F1_curve.png)  
Figure s721. precision_over_recall, yolov8n, pre-trained=True, epochs=0, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/val2/PR_curve.png)  
Figure s722. precision_over_confidence, yolov8n, pre-trained=True, epochs=0, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/val2/P_curve.png)  
Figure s723. recall_over_confidence, yolov8n, pre-trained=True, epochs=0, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/val2/R_curve.png)  
Figure s724. confusion_matrix, yolov8n, pre-trained=True, epochs=0, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/val2/confusion_matrix.png)  
Figure s725. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=0, data=duomo, hyperparameter-tuned=False  
![](./saved_runs/val2/confusion_matrix_normalized.png)  
Figure s726. f1_over_confidence, yolov8n, pre-trained=True, epochs=0, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/val3/F1_curve.png)  
Figure s727. precision_over_recall, yolov8n, pre-trained=True, epochs=0, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/val3/PR_curve.png)  
Figure s728. precision_over_confidence, yolov8n, pre-trained=True, epochs=0, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/val3/P_curve.png)  
Figure s729. recall_over_confidence, yolov8n, pre-trained=True, epochs=0, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/val3/R_curve.png)  
Figure s730. confusion_matrix, yolov8n, pre-trained=True, epochs=0, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/val3/confusion_matrix.png)  
Figure s731. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=0, data=hadji_dimitar_square, hyperparameter-tuned=False  
![](./saved_runs/val3/confusion_matrix_normalized.png)  
Figure s732. f1_over_confidence, yolov8n, pre-trained=True, epochs=0, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/val4/F1_curve.png)  
Figure s733. precision_over_recall, yolov8n, pre-trained=True, epochs=0, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/val4/PR_curve.png)  
Figure s734. precision_over_confidence, yolov8n, pre-trained=True, epochs=0, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/val4/P_curve.png)  
Figure s735. recall_over_confidence, yolov8n, pre-trained=True, epochs=0, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/val4/R_curve.png)  
Figure s736. confusion_matrix, yolov8n, pre-trained=True, epochs=0, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/val4/confusion_matrix.png)  
Figure s737. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=0, data=keskvaljak, hyperparameter-tuned=False  
![](./saved_runs/val4/confusion_matrix_normalized.png)  
Figure s738. f1_over_confidence, yolov8n, pre-trained=True, epochs=0, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/val5/F1_curve.png)  
Figure s739. precision_over_recall, yolov8n, pre-trained=True, epochs=0, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/val5/PR_curve.png)  
Figure s740. precision_over_confidence, yolov8n, pre-trained=True, epochs=0, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/val5/P_curve.png)  
Figure s741. recall_over_confidence, yolov8n, pre-trained=True, epochs=0, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/val5/R_curve.png)  
Figure s742. confusion_matrix, yolov8n, pre-trained=True, epochs=0, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/val5/confusion_matrix.png)  
Figure s744. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=0, data=kielce_university_of_technology, hyperparameter-tuned=False  
![](./saved_runs/val5/confusion_matrix_normalized.png)  
Figure s745. f1_over_confidence, yolov8n, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/val6/F1_curve.png)  
Figure s746. precision_over_recall, yolov8n, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/val6/PR_curve.png)  
Figure s747. precision_over_confidence, yolov8n, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/val6/P_curve.png)  
Figure s748. recall_over_confidence, yolov8n, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/val6/R_curve.png)  
Figure s749. confusion_matrix, yolov8n, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/val6/confusion_matrix.png)  
Figure s750. confusion_matrix_normalized, yolov8n, pre-trained=True, epochs=0, data=toggenburg_alpaca_ranch, hyperparameter-tuned=False  
![](./saved_runs/val6/confusion_matrix_normalized.png)  
