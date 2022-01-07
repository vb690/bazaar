# Visualize Gradient Descent In UMAP Space

This project aims to visualize how different gradient descent algorithms optimize the weights of a 3 layers Multilayer Perceptron (MLP) tasked to predict the digits of the Fashion MNIST dataset.

# Motivation and Approach

The motivation behind this project is to get an idea of how the various algorithms "move" in the multidimensional space defined by the weights of the MLP.
Our approach cover the following steps

1. Generate the MLP model.
2. Train the model on batch.
3. After each batch update extract the weights from the 3 layers.
4. Once training is completed, for each layer the obtained weights are concatenated in a `N X Z` array, where `N` is the number of training steps and `Z` the dimensionality of the layer.
5. The resulting arrays are included in a list and transformed by [AllignedUMAP](https://umap-learn.readthedocs.io/en/latest/aligned_umap_basic_usage.html) in as many `N X 2` arrays.
6. The reductions produced by AllignedUMAP are then used for producing animated plots.  
  
Steps 1 to 4 are perfromed for each chosen algorithm and the weights are collated in a single array. This is done in order to better represent the space defined the MLP's weights.     
# Features

* Model building function
* Weights extraction function
* Utility function for visualizing the descent in 2D and 3D

# Results 

<p align="center">
  <img width="900" height="200" src="https://github.com/vb690/bazaar/blob/visualize_gradient_descent/shops/visualize_gradient_descent/results/gifs/combined_2d.gif">
</p>

<p align="center">
  <img width="900" height="200" src="https://github.com/vb690/bazaar/blob/visualize_gradient_descent/shops/visualize_gradient_descent/results/gifs/combined_3d.gif">
</p>
