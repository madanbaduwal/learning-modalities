# learning-paradigms / type of learning in machine learning / machine learning modalities

## Supervise Learning

* Label Datasets
* We have an input x, and explicit label y
* **Model update model parameters based on input x and explicit label y**

### Self-supervise learning

* We have an input x, and implicit label y i.e label is already in input
* Example of such task is language modeling 
* Example: a model can be trained to predict the missing word in a sentence, to predict the next word given the previous ones, or to classify the rotation angle of an image.
* **Model update model parameters based on input and implicit label y**


## Unsupervise Learning
* Unlabel Datasets
* We have an input x, and no label y, but it can use x as a label y
* **Model update model parameters based on no label or input x**

## Reinforcement Learning
* **Model update parameters based on reward or punishment**
reinforcement learning: https://en.m.wikipedia.org/wiki/Reinforcement_learning

## Semi-Supervise Learning
* (hybrid)(half label data + half unlabel data)
* We have an input x, some data points are labeled lots of them are unlabeled, we predict label of unlabeled data and train model on whole data.
* **Model update parameters based on first some labelled data and later on whole data**


## Transfer Learning

* **Model(pretrained) don't update model parameters(freez), used as a feature extractor(only work for forward pass). Add some specific layers(classifier layer) at top of that, train it(update model parameters) for our specific task.**
* (statistical inference)
* In transfer learning, we first train a base network on a base dataset and task, and then we repurpose the learned features, or transfer them, to a second target network to be trained on a target dataset and task. This process will tend to work if the features are general, meaning suitable to both base and target tasks, instead of specific to the base task.



## Fine tunning learning
* **Model(prettained) update its parameters based on our specific task**
In deep learning, fine-tuning is an approach to transfer learning in which the weights of a pre-trained model are trained on new data.[1] Fine-tuning can be done on the entire neural network, or on only a subset of its layers, in which case the layers that are not being fine-tuned are "frozen" (not updated during the backpropagation step).[2] A model may also be augmented with "adapters" that consist of far fewer parameters than the original model, and fine-tuned in a parameter-efficient way by tuning the weights of the adapters and leaving the rest of the model's weights frozen.

### Transfer learning vs fine-tuning
* **In transfer learning we load pre-trained data freez layers, add clasifier layer only and train on your data i.e used pre-trained model as a feature extractor and added classifier layer as a classifier. In fine-tuning we update pre-trained model parameters based on our custom data.**


## Multi-Instance Learning
* ( Hybrid learning)(supervise + unsupervise + reinforcement)
* * 

## Feature Learning
automatically finds and improves key patterns, characteristics, or structures (called "features") from raw data. 

## Inductive Learning

Inductive learning, also known as discovery learning, is a process where the learner discovers rules by observing examples.

## Deductive Learning

Deductive learning can also refer to using an existing theory or model and deducing outcomes from it.

## Transductive Learning

Transduction is reasoning from observed, specific (training) cases to specific (test) cases. 

## Multi-Task Learning
Multiple learning tasks are solved at the same time, while exploiting commonalities and differences across tasks.

## Active Learning
Active learning is a special case of machine learning in which a learning algorithm can interactively query a human user (or some other information source), to label new data points with the desired outputs.

## Online Learning
Online machine learning is a technique that uses real-time data to train a model that adapts its predictive algorithm over time.

## Ensemble Learning
Ensemble learning in machine learning is the process of combining multiple machine learning models into a single model to improve performance. 

## Time Series Analysis
Time series analysis is a machine learning technique that involves extracting useful information from time-series data to gain insights and forecast. 

## Association Rule
Association rule learning is a machine learning method that uses a rule-based approach to identify relationships between variables in large databases.

## Federal Learning
Federated learning (FL) is a machine learning technique that involves multiple entities collaborating on training a model while keeping data decentralized.

## Sparse Dictionay Learning

Sparse dictionary learning (also known as sparse coding or SDL) is a representation learning method which aims at finding a sparse representation of the input data in the form of a linear combination of basic elements as well as those basic elements themselves. 

## Robot Learning

Robot learning is a research field that uses machine learning algorithms to help robots learn new skills and adapt to their environment. 

## Imitation learning

Generally, imitation learning is useful when it is easier for an expert to demonstrate the desired behaviour rather than to specify a reward function which would generate the same behaviour or to directly learn the policy.

## Generative Learning

Generative learning is a machine learning model that learns the patterns and distributions of data to create new, similar data. 

## Multi-Model Learning

Different modalities of data

## Meta-Learning
Meta learning is a machine learning paradigm that helps models learn new tasks on their own and improve their performance over time.


## Reinforcement Learning from Human Feedback (RLHF) / Fine tuning with human feedback:
Reinforcement Learning from Human Feedback (RLHF) is a machine learning technique that uses human feedback to improve ML models' ability to self-learn.


## Zero-shot learning : 

Zero-shot learning (ZSL) is a machine learning technique that allows an AI model to recognize and categorize objects or concepts without seeing any examples of those categories or concepts beforehand.


## Few-shot learning: 

Few-shot learning (FSL) is a machine learning technique that teaches AI models to learn from a small amount of labeled training data, often with only a few training samples. 

## Multi-view learning

## Contrastive learning
