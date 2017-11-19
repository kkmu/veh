# Vehicle Make/Model Prediction (incl. of indian transports such as auto rickshaws and lorries.)
This submission is being done as part of hack2innovate mumbai edition.
## NN Architechture & DL Framework
Our model is fine-tuned on VGG16 and trained over apache mxnet framework.
## Data Preparation
We have started with Stanford Cars Dataset and added carefully curated custom classes such as lorries and auto-rickshaws.
We have pruned Stanford Cars Dataset to only contain 29 classes (owning to resource/time constraints) and with 2 custom clasess we have 31 vehicles with different make/models.
We have ignored "year of the make" for our analysis owing to very less data for make/model/year subset.
### Build .lst and .Rec files necessary for training over mxnet.
python build_dataset.py
~/mxnet/bin/im2rec /home/ubuntu/veh/datasets/lists/train.lst "" /home/ubuntu/veh/datasets/rec/train.rec resize=256 encoding='.jpg' quality=100
~/mxnet/bin/im2rec /home/ubuntu/veh/datasets/lists/test.lst "" /home/ubuntu/veh/datasets/rec/test.rec resize=256 encoding='.jpg' quality=100
~/mxnet/bin/im2rec /home/ubuntu/veh/datasets/lists/val.lst "" /home/ubuntu/veh/datasets/rec/val.rec resize=256 encoding='.jpg' quality=100
### Model Fine-Tuning
Fine-tuning exercise was repeated with SGD optimizer and various learning rates multiple times and loss/accuracty is monitored by each epoch/learning rates. 
python fine_tune_cars.py --vgg vgg16/vgg16 --checkpoints checkpoints --prefix vggnet
### Model Validation
We have evaluated the classifier on the test data with metrics "rank-1" and "rank-5".
python test_cars.py --checkpoints checkpoints --prefix vggnet --epoch 62

///for epoch 62, we got the highest rank-1 and rank-5 accuracy of 90.18% and 97.97% respectively///
(faceiq) ubuntu@ip-172-31-4-222:~/veh$ python test_cars.py --checkpoints checkpoints --prefix vggnet --epoch 62
[08:11:59] src/io/iter_image_recordio_2.cc:153: ImageRecordIOParser2: /home/ubuntu/veh/datasets/rec/test.rec, use 1   threads for decoding..
[INFO] loading pre-trained model...
[08:12:12] src/operator/././cudnn_algoreg-inl.h:112: Running performance tests to find the best convolution algorithm, this can take a while... (setting env variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)
[INFO] evaluating model...
[INFO] rank-1: 90.18%
[INFO] rank-5: 97.95%

### Visualizing Classification
python vis_classification.py --checkpoints checkpoints --prefix vggnet --epoch 62
///Some sample predictions///
[INFO] actual=Toyota:Corolla
        [INFO] predicted=Toyota:Corolla, probability=99.60%
        [INFO] predicted=Nissan:240SX, probability=0.19%
        [INFO] predicted=Toyota:Camry, probability=0.12%
        [INFO] predicted=Plymouth:Neon, probability=0.06%
        [INFO] predicted=Volkswagen:Golf, probability=0.03%
[INFO] actual=Volkswagen:Golf
        [INFO] predicted=Volkswagen:Golf, probability=99.55%
        [INFO] predicted=Volvo:240, probability=0.39%
        [INFO] predicted=Volvo:XC90, probability=0.05%
        [INFO] predicted=Plymouth:Neon, probability=0.00%
        [INFO] predicted=Mercedes-Benz:E-Class, probability=0.00%
[INFO] actual=lorry:Indian
        [INFO] predicted=lorry:Indian, probability=100.00%
        [INFO] predicted=Scion:xD, probability=0.00%
        [INFO] predicted=auto:Indian, probability=0.00%
        [INFO] predicted=Toyota:Corolla, probability=0.00%
        [INFO] predicted=Nissan:Leaf, probability=0.00%
[INFO] actual=auto:Indian
      [INFO] predicted=auto:Indian, probability=100.00%
      [INFO] predicted=Scion:xD, probability=0.00%
      [INFO] predicted=smart:fortwo, probability=0.00%
      [INFO] predicted=lorry:Indian, probability=0.00%
      [INFO] predicted=Mercedes-Benz:Sprinter, probability=0.00%

## Note: We have achieved rank-1/rank-5 accuracy of over 90.18% and 97.95% respectively with just 3000+ images for 31 class problem with an average of less than 100 images/class. With more labelled data, we can get much better accuracy. We have attached word document with predicted images for your reference.





