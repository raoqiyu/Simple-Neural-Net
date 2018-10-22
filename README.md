Simple Neural Net
This is a C++ implement of simple neural network. It's based on [Neural Net in C++ Tutorial](https://vimeo.com/19569529) by David Miller.

# Step
1 Gernerate training data to slove XOR problem
```bash
g++ ./makeTrainingSamples.cpp -o makeTrainingSamples
./makeTrainingSamples > out.txt
```

2 Test neural netwrok
```
g++ ./neural-net.cpp -o neural-net
./neural-net
```

