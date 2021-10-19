# Dog breed classifier

This project is divided into two parts: The WebApp developed to classify dog breeds and the training process which was used to generate a machine learning model used by the web app.

## WebApp

The project can be used directly in this streamlit page: https://share.streamlit.io/gasparitiago/dog-breed-classifier/main/stream.py

If you want to deploy the webapp locally you can simply use streamlit:

```
    pip install streamlit
    streamlit run stream.py
```

## Project Overview

The project overview, including the motivation, approaches, methodology, results and conclusions can be found in this Medium post: https://medium.com/@gasparitiago/developing-a-dog-breed-classifier-e2348602cb99

If you want to re-train the model or take a look at how the project has been developed, or the motivations of it, you can take a look at the `dog_app.ipynb` file. This file was used to train the models and to analyze the data used in the Medium blog post and in the Web app.

### Requirements

All the requirements of the project are listed in the `requirements.txt` file and can be installed by running:

```
pip install requirements.txt
```

### Repository structure

- images: directory where the images used in the notebook for tests are located;
- requirements: requirement files to run the solution on desktop (windows, linux and mac)
- saved_models: directory containing checkpoints of the models used during the training process of different approaches
- dog_app.py: notebook containing different approaches used to create the models.
- extract_bottleneck_features: auxiliary file used to extract bootleneck features of different cnn models.
- stream.py: file containing the webapp to be deployed using streamlit.

---

## Results

Here is the accuracy of each model trained during the development of this project:

- CNN from scratch: 1.31%
- CNN from VGG-16: 41.26%
- CNN from InceptionV3: 78.94%

It is possible to see that the CNN written from scratch has better accuracy than just random guesses, which is 1 in 133: 0.75%, but it's far from acceptable.

Using the VGG16 pre-trained weights is a great improvement, but misses in more than half of the predictions.

Finally, using the InceptionV3 as starting point and just training the fully connected layer for predicting the custom classes, provided a model with an accuracy higher than 78%.


## Conclusion

As we can see in the results of the different methods used in this project, image classification can be considered a hard or an easy task, according to the techniques used in different approaches.

It's simple to write a CNN from scratch, selecting the number of layers, the order of them, and how the data will be processed, but this approach doesn't seem to be suitable for our solution. Few images to train and little time for training a model from zero can be problems for obtaining good results.

Using transfer learning techniques to improve the accuracy seems to be the best option in this kind of project and it's an easier solution as pre-trained models are widely available online.