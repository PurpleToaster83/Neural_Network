# Neural_Network

Backpropagation
* Propogate Method
    * defined architecture: 2 neurons per layer, 1 hidden layer
    * minimizes error of the system

* Class System
    * Modular: can create different architectures (add layer and # neurons)
    * Nested Classes: Netowork --> Layer --> Neuron
    * minimize use of external libraries (ideally only math)
    * TODO: work on math for backpropogation
        1. recursive path finding for ea. weight
        2. calculate all partials
        3. multiply partials according to path for cumulative partial
        4. correct weights with cumulative partials

Gradient Descent
* Basic linear regression model
    * uses sum of square residuals as error function
    * only works on magnitude of less than 10 on inputs
    * begins with assumed line y = x
    * graphs the results