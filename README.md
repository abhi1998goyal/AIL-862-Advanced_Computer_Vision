Ass-1
You received a dataset where inadvertently overlapping scenes and same scenes with different noise augmentation were mixed up. You want to identify such scene pairs. For this, you need to design a mechanism that can compute similarity between image pairs (definition of similarity is in the context of this problem). The structural_similarity function (from skimage.metrics) or other such 1 line command is not allowed. Using deep learning or deep learning based features is allowed, but not mandatory. One can potentially use traditional shallow features, too.

Ass-2
In this assignment, you will work on the UC Merced dataset to develop  classification model(s) under noisy label conditions in the training data. Your task is to perform various analyses to understand and mitigate the effects of label noise. You are free to decide what all analyses you want to perform. This assignment carries 4 marks.
Key Tasks: Build and train  classification model on the UC Merced dataset with varying levels of label noise in the training data. Some examples of possible  analysis: the impact of different degrees of label noise on model performance; the effect of using different model architectures. 
Document your findings in a concise one-page report. 

Ass-3
Problem Statement
Choose a few distinct image classes of your choice.
Use a text-to-image generation model  to generate synthetic images for each class. 
Divide your synthetic dataset into two splits - training and validation.
Train a deep learning classifier using only the synthetic dataset.
Test your classifier on a real dataset consisting of the given classes. Measure performance gap of your model between real dataset and synthetic dataset validation set. 
See if this performance gap can be reduced by merely increasing image numbers in the synthetic dataset or some other simple trick during text to image generation.
Consider that you have an unlabeled dataset that mostly has images from the classes of your interest but 10% of this dataset are images from other classes.
Use the above-mentioned unlabeled dataset, potentially with some domain adaptation technique, to further improve the model trained on synthetic data. 
Report Format

Refer to IEEE conference (two column) format, please submit 1-2 page report.

Ass-4
Problem: Use a pretrained DINO model (without additional training or fine-tuning) to perform semantic segmentation on the images from Assignment 1.

Ass-5
Assume that you have the following:
Item A:  Just one example image (image and the corresponding segmentation mask of the target class) 
Item B: The textual description of the target class. 
Item C: A number of unlabeled images, some (but not all) of which may have the target class
Item D: A test set, with images and the segmentation masks of the target class.
Use trained CLIP and SAM along with items A and B to annotate the images from Item C. We can call this pseudo labeled dataset as C_prime. Now, use this annotated dataset C_prime to finetune TransUNet or any other ViT-based  semantic segmentation model. Since I am asking to finetune, you can start from a model which has been already trained on some other dataset.
Use the dataset in item D for only evaluating the model, i.e., do not use it for training purposes.
Specific A, B, C, D - you are free to choose.
Code submission - All codes must be in the "codes" folder. This folder should have a main.py in addition to the other files. Code should be submitted in such a way that running $python main.py from this folder should automatically complete all the steps and produce the final fine-tuned model and furthermore print on screen the accuracy indices of the fine-tuned model on the test dataset from item D.
It is recommended to include a requirements file.

Report -  About 2 pages, similar to previous assignments.
