# IFT6759 - Door Number Detection Project [Advanced projects in machine learning]

## Course Overview
IFT6759 is a course about implementing deep learning theory to real-world industrial projects.

**Course professor:**
* Aaron Courville

**Door Number Detection project assistants:**
* Margaux Luck
* Jeremy Pinto

**Other course assistants:**
* Mathieu Germain
* Francis Grégoire
* Simon Blackburn
* Arsène Fansi Tchango
* Joumana Ghosn
* Gaétan Marceau Caron

## Door Number Detection Project
The goal of this project is to help blind persons to find their way around by making sure they are at the right house when they want for example visit a friend or a family member, go to a specific store, etc.

In developing this project we must keep in mind the different constraints of this application notably for the selection and development of the models we will use like the execution time, online vs. offline, the memory usage (in the case of a mobile application), etc.

### Datset used
[SVHN dataset](https://github.com/mila-iqia/digit-detection/blob/master/project/data/SVHN.md)
* This dataset is used to train a multi-task classifier to identify digits contained within a bounding-box

[Avenue Dataset]()
* A synthetic dataset generated by the Unity engine. It contains houses with door numbers and bounding boxes associated to the numbers as well as their contents.

## Using this repository

The course was divided in to 3 "blocks", or milestones. During each milestone, students had to accomplish a specific task. To access the code associated to each block, checkout the branch with the associated block name, i.e. to access block 2, use

`git checkout bloc2`

### Block 1

The purpose of the first block was to introduce students to tools commonly used in industry, such as git, pytorch, shell scripting and launching jobs on a cluster with shared resources (i.e. GPUs).

The first task was to implement a classifier predicting the door number sequence length of bounding boxes from the SVHN dataset. We followed ideas proposed in [Goodfellow et al.](https://arxiv.org/abs/1312.6082).

### Block 2

In block 2, we performed full digit recognition still based on the SVHN dataset and following the full architecture presented in [Goodfellow et al.](https://arxiv.org/abs/1312.6082). Students also had to implement checkpointing of models,  hyperparameter optimization, and reproducible experiments. Students had to choose and justify which models to use (i.e. resnet, vgg, densenet, etc.).

### Block 3

In Block 3, students were provided with the Avenue synthetic dataset kindly provided by ElementAI. The goal of the block was to use FasterRCNN to perform object detection of bounding boxes containing door numbers in an image and using models from block 2 to perform sequence recognition. They had to implement this in a "pipeline" approach, i.e. not an end to end approach. This was mainly due to time and complexity constraints. Block 2 models had to be retrained on the Avenue dataset for better results.

## Important files
- The [Syllabus](https://github.com/mila-iqia/digit-detection/blob/master/syllabus.md).
- The [Weekly Agenda](https://github.com/mila-iqia/digit-detection/blob/master/agenda.md) which contains the __deadlines__, the __links to the material__ and the __homeworks__.
- The [How-to submit](https://github.com/mila-iqia/digit-detection/blob/master/project/evaluation/guidelines/howto-submit.md).
- The [How-to submit a code review](https://github.com/mila-iqia/digit-detection/blob/master/project/evaluation/guidelines/howto-codereview.md).
- The [frequently asked questions (FAQ)](https://github.com/mila-iqia/digit-detection/blob/master/faq.md).
- How to run the [containers](https://github.com/mila-iqia/digit-detection/tree/master/project/container).
