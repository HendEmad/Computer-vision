# Basic definations in deep learning:
We split the training set into many batches. When we run the algorithm, it requires one epoch to analyze the full training set. An epoch is composed of many iterations (or batches).

***Iterations:*** The number of batches needed to compelete one epoch.

***Batch Size:*** The number of training samples used in one iteration.

***Epoch:*** One full cycle through the training dataset. One cycle is a composed of many iterations.

***Number of Steps per Epoch =*** (Total Number of Training Samples) / (Batch Size)

***For example:*** if we have a training set of 2000 images, this data is splitted into 10 batches ==> so the number of steps is 2000/10 = 200 step.

***So the order of operations here from smaller to bigger is:*** batch -> iteration -> epoch. 
We split the dataset into batches. a group of batches used for one ieration, then a group of iterations make the epoch.
