![image](https://github.com/user-attachments/assets/e648fe98-013c-4cc2-8583-1ecd12c774f6)# Road-signs-classification
This model is capable of identifying about 44 types of road signs, with 94% accuracy

The task is for building a model capable of identifying about 44 different traffic signs

1. Data

The data was obtained from [Kaggle](https://www.kaggle.com/datasets/tuanai/traffic-signs-dataset)

### Training.
The first part of training involved using EfficientNetV2B2 as the model's backbone to perform Feature extraction. The model achieved a low accuracy of 48% which was lower than the paper.

**The experimental training happened in two steps:**
1. On the feature extraction model

* With non-augmented data

## Loss Curves
![Training Loss Curves](https://github.com/user-attachments/assets/02d19841-2ba9-449b-963d-15a9b0895f5d)

## Accuracy Curves
![Accuracy Curves](https://github.com/user-attachments/assets/6caf5a9a-ff1a-4cdf-b599-38bbd66927bb)

* With Augmented data

## Loss Curves
![Training Loss Curves](https://github.com/user-attachments/assets/fed855ad-d1ca-4db1-95db-737139607c71)


## Accuracy Curves
![Accuracy Curves](https://github.com/user-attachments/assets/de2327e9-374a-4aca-884f-9760efde75d4)


2. Fine Tuning The feature extraction model

* With non-augmented data

## Loss Curves
![Training Loss Curves](https://github.com/user-attachments/assets/c13cfed1-7811-47b9-831e-178722bec674)

## Accuracy Curves
![Accuracy Curves](https://github.com/user-attachments/assets/c14509d2-80f2-4552-be05-ac10a45ca295)


* With Augmented data

## Loss Curves
![Training Loss Curves](https://github.com/user-attachments/assets/b5357426-5dd9-4a03-bba7-2bbe70816c7e)


## Accuracy Curves
![Accuracy Curves](https://github.com/user-attachments/assets/5fb58a09-6d20-4a85-b2e1-cf8f120c232a)


The best overall model, was the fine-tuned model, without data augmentation.

# Making Predictions

![image](https://github.com/user-attachments/assets/2538ef6d-cbea-4f10-8e1f-1e70de5bbba7)
![image](https://github.com/user-attachments/assets/25677be2-db08-409d-9535-1ed65f5f9aa1)
![image](https://github.com/user-attachments/assets/7b24ce51-d15a-4830-9f2a-f6f6e9077000)
![image](https://github.com/user-attachments/assets/ab884b9a-c580-4578-9b86-6d6afb7ddc44)


