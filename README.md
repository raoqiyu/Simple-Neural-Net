Simple Neural Net
This is a C++ implement of simple neural network. It's based on [Neural Net in C++ Tutorial](https://vimeo.com/19569529) by [David Miller](http://www.millermattson.com/dave/?p=54).

# Steps
1 Gernerate training data to slove XOR problem
```
g++ makeTrainingData-XOR.cpp -o makeTrainingData-XOR
./makeTrainingSamples > out.txt
```

2 Test neural netwrok
```
g++ ./NeuralNet.cpp -o NeuralNet
./NeuralNet
```
