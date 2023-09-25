# Modeling a Perceptron

## Contents
- What is a Perceptron?
- How Does a Neural Network Learn?
- What this Program Does
- How To Access this Program?

<br>

## What is a Perceptron?
![Perceptron_Diagram](https://github.com/MukundAravapalli/PerceptronModel/assets/105079738/ca495750-afb4-4263-869e-e9495621e30e)
<img width="1434" alt="Perceptron_Diagram" src="https://github.com/MukundAravapalli/PerceptronModel/assets/105079738/ca495750-afb4-4263-869e-e9495621e30e">

A **Perceptron** is the smallest unit of a Neural Network. It is an abstract element that takes in a set of pairs of **weights** and **inputs**. The perceptron multiples each input value, *x*,  with its corresponding weight, *w*, which results in a product. These products are all passed into an **Activity Function** where they are added together along with a term called the **bias**, *θ*. Note that the Activity Function we just described is labeled as the **Net Input Function** in the diagram above. 

The bias serves as a threshold that the sum of the input products must hurdle over in order to activate the perceptron. In other words, if the sum of the inputs is 1.7 and the bias is -2, the resulting sum would be a negative value. If a threshold function has a cutoff point at 0, then we can conclude that this perceptron did not get activated. This function of the bias becomes very useful in training multilayered neural networks where the correct answer is highly dependent on which perceptrons are activated and which are not. The bias ensures that only useful perceptrons are activated in a multilayer network.

Once all the inputs and the bias term have been added together, the sum gets passed onto the **Activation Function**. There are many different types of Activation Functions used in different scenarios, but for our Perceptron Model, we will use the popular **Sigmoid Function** shown below. 

![Sigmoid_Function](/public/diagrams/SigmoidFunc.png)

The Sigmoid Function is a differentiable function that stretches between 0 and 1. The graph follows the equation shown above **1/(1+e<sup>-x</sup>)** where x is the result of our **Net Input Function**. Therefore when we plug in the result of our Net Input Function for x, we get a value between 0 and 1. This result can either be passed on as an input to another perceptron, or it can serve as our network’s output. If the value is close 0 then it indicates a False value, and if the value is close to 1, it indicates True. We can use this principle to model logic operations such as AND, OR, NAND, XOR, etc using Perceptrons. 

<br>

## How does a Neural Network Learn?
In our model, the Neural Network will have access to a desired answer that we will call **delta**, *d*. When the network outputs a result, *y*, we will compare this output with the delta value using the equation e<sub>j</sub> = d<sub>j</sub> - y<sub>j</sub> where *e* is the error, *d* is our delta value, and *y* is the output value. 

Once we know the difference between our desired value and the network's output, we can adjust our weights to move in the direction of our desired output. In order to do this, we will use the equation: Δw<sub>ij</sub> = ηδ<sub>j</sub>x<sub>i</sub>. The full form of which is: Δw<sub>ij</sub> = -e<sub>j</sub>[1-y<sub>j</sub>]y<sub>j</sub>ηx<sub>i</sub>. 

This seems very confusing but let us break down the parts.
Δw<sub>ij</sub>: The value of this term is the amount by which we must adjust our weight in order to move towards our desired result.  

-e<sub>j</sub>[1-y<sub>j</sub>]y<sub>j</sub>:   Recall that we defined e<sub>j</sub> as e<sub>j</sub> = d<sub>j</sub> - y<sub>j</sub> earlier. We are multiplying this error term with the derivatie of our Activation Function with respect to y<sub>j</sub>.

η: This symbol is eta, and it denotes the step size that we take in order to move in the direction of our desired result. 

x<sub>i</sub>: x denotes our input value. 

The Nueral Network learns by comparing its output to our desired result and adjusting the weights and bias in a way that moves its output to the desired result. 

<br>

## What this program does
This Python program starts by accepting the following values from the user:
- 2 input values: x1 and x2
- 2 weight values: w1 and w2
- A bias value, θ
- An eta value, our step size
- A delta value, our desired result
- The number of iterations we want our Activation function to be performed for

This program features the Activation function, and a for loop which adjusts the weights and bias values based on the difference between the previous output and the desired result. 

The activation function we use for this model is the Sigmoid Function:
  
```
def activation_func(A):
    return 1.0 / (1 + np.exp(-1 * A))
```

As you can see below, the for loop runs for a certain amount of iterations to adjust the weights and bias values accordingly:

```
for i in range(0, num_iterations):
    A = w1 * x1 + w2 * x2 + bias

    #Activity function output
    y = activation_func(A)

    # adjusted error value based on 
    # desired value and network output
    delta = (d - y) * y * (1 - y)

    #Adjusted weights to be fed back into
    #into the Activity Function
    w1 = w1 + eta * delta * x1
    w2 = w2 + eta * delta * x2

    # Adjusted bias to be fed back into
    # the Activity Function
    bias = bias + eta * delta * 1
    print(y)
```

Suppose we start with the values on the left part of the picture below. You can see on the right hand terminal the output values that we strt out with. We have a **desired value** of **0.798**.

![Starting_Output](/public/diagrams/StartingOutput.png)

You can see, after 100 iterations of adjusting our weights to the Activity Function, we have converged on the desired value.

![Ending_Output](/public/diagrams/EndingOutput.png)

<br>

## How to access the program
You can access and edit the values in this program by downloading the **perceptron_model** file and opening up the **main.py** file in any Python editor. 


