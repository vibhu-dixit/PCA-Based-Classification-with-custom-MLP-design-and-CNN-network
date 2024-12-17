# Phase 1:-

***Task 1. Feature normalization (Data conditioning)*** <br>
Need to normalize the data in the following way, before starting any subsequent tasks. Using all the training images (each viewed as a 784-D vector, X = [x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>784</sub>]t, as explained), compute the mean m<sub>i</sub> and standard deviation (STD) s<sub>i</sub> for each feature x<sub>i</sub> (remember that we have 784 features) from all the training samples. The mean and STD will be used to normalize all the data samples (training and testing): for each feature x<sub>i</sub>, in any given sample, the normalized feature will be, y<sub>i</sub> = (x<sub>i</sub> - m<sub>i</sub>)/s<sub>i</sub>.<br>
***Task 2. PCA using the training samples*** <br>
Use all the training samples to do PCA. Cannot use a built-in function PCA or similar, if your platform provides such a function. You have to explicitly code the key steps of PCA: computing the covariance matrix, doing eigen analysis (can use built-in functions for this), and then identify the principal components.<br>
***Task 3. Dimension reduction using PCA*** <br>
Consider 2-d projections of the samples on the first and second principal components. These are the new 2-d representations of the samples. Plot/Visualize the training and testing samples in this 2-d space. Observe how the two classes are clustered in this 2-D space. Does each class look like a normal distribution?<br>
***Task 4. Density estimation***<br>
We further assume in the 2-d space defined above, samples from each class follow a Gaussian distribution. Will need to estimate the parameters for the 2-d normal distribution for each class, using the training data. We will have two distributions, one for each class.<br>
***Task 5. Bayesian Decision Theory for optimal classification*** <br>
Use the estimated distributions for doing minimum-error-rate classification. Report the accuracy for the training set and the testing set respectively.
<br>
<br>

## Phase 2:-

_Task 1 specific requirements:_ <br><br>
1. Implement a 2-n<sub>H</sub>-1 MLP from scratch, without using any built-in library. Write the code in such a way that we may try different number of hidden nodes easily. Pick the own activation function; use MSE error as the loss. Decide on other details such as batch/mini-batch/stochastic learning, learning rate, whether to use momentum term, etc. <br>
2. Let n<sub>H</sub> take the following values: 2, 4, 6, 8, 10. For each case, train the network with the first 1500 training samples from each class, given in train_class0.mat (for class 0) and train_class1.mat (for class 1), respectively. In the data files, each line is a 2-D sample, and the number of lines is the number samples in that class. We have 2000 training samples for each class. We will use only the first 1500 training samples from each class for training the network and reserve the remaining 500 (1000 total for 2 class) as the “validation set”. Train the network until the learning loss/error (J or J/n as defined in the lecture slides) for the validation set no longer decreases. Then test the network on the testing data, given in test_class0.mat (for class 0) and test_class1.mat (for class 1), respectively. We have 1000 testing samples for each class.<br>
3. Plot the learning curves (for the training set, validation set, and the test set, respectively), for each value n<sub>H</sub>.<br>
4. Report at which value of n<sub>H</sub>, the network gives the best classification accuracy for the testing set.<br><br>

_Task 2 specific requirements:_ <br><br>
In this task, we will perform the classification task, using a convolutional neural network. The dataset is the CIFAR-10 dataset. We will experiment with a convolutional neural network with the following parameter settings: <br>
-- The input size is the size of the image (32x32x3).<br>
-- **First layer** – Convolution layer with 32 kernels of size 3x3. It is followed by ReLUactivation layer and batch normalization layer.<br>
-- **Second layer** – Convolution layer with 32 kernels of size 3x3. It is followed by ReLU activation layer and batch normalization layer.<br>
-- **Third layer** – Max pooling layer with 2x2 kernel. <br>
-- **Fourth layer** – Dropout layer with 0.2 probability of setting a node to 0 during training.<br>
-- **Fifth layer** – Convolution layer with 64 kernels of size 3x3. It is followed by ReLU activation layer and batch normalization layer.<br>
-- **Sixth layer** – Convolution layer with 64 kernels of size 3x3. It is followed by ReLU activation layer and batch normalization layer.<br>
-- **Seventh layer** – Max pooling layer with 2x2 kernel. <br>
-- **Eighth layer** – Dropout layer with 0.3 probability of setting a node to 0 during training.<br>
-- **Ninth layer** – Convolution layer with 128 kernels of size 3x3. It is followed by ReLU activation layer and batch normalization layer.<br>
-- **Tenth layer** – Convolution layer with 128 kernels of size 3x3. It is followed by ReLU activation layer and batch normalization layer.<br>
-- **Eleventh layer** – Max pooling layer with 2x2 kernel. <br>
-- **Twelfth layer** – Dropout layer with 0.4 probability of setting a node to 0 during training.
-- **Last layer** – Fully connected layer with 10 nodes (corresponding to the 10 classes) and SoftMax activation function.<br>
