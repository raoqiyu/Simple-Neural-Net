//
// An implemention of BP Neural Net
//
// 2015/4/6
// @author raoqiyu,


#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
using namespace std;

class Neuron;
typedef vector<Neuron> Layer;
struct Connection
{
	double weight;
	double deltaWeight; // previous delta weight ,multipy by alpha(momentum)
};

// *******************************    class TrainingData  *********************************
class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}

// *******************************    class Neuron    *********************************
class Neuron
{
public:
	Neuron(unsigned numOutputs,unsigned myIndex);
	void setOutputValue(double val){ m_outputValue = val; }
	double getOutputValue(void) const { return m_outputValue; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetValue);
	void calcHiddenGradients(Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static double randomWeight(void){ return rand()/double(RAND_MAX);}
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double sumDOW(const Layer &nextLayer);
	double m_outputValue;
	unsigned m_myIndex;
	double m_gradient;
	vector<Connection> m_outputWeights;

	// 
	static double eta; // [0.0 ... 1.0] , overall net training rate
	static double alpha; // [0.0 ... n] multiplier of last weight change(momentum)
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs,unsigned myIndex)
{
	for(unsigned i = 0; i < numOutputs; i++){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}

double Neuron::transferFunction(double x)
{
	// tanh - output range [ -1.0 .. 1.0]
	return tanh(x);
	
}

double Neuron::transferFunctionDerivative(double x)
{
	// tanh derivative
	return 1.0 - x*x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;
	// sum the previous layer's outputs(which are our inputs)
	// Include the bias node from the previous layer

	for(unsigned neuronNum = 0; neuronNum < prevLayer.size(); neuronNum++){
		sum += prevLayer[neuronNum].getOutputValue() *
				prevLayer[neuronNum].m_outputWeights[m_myIndex].weight;
	}

	m_outputValue = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetValue)
{
	double delta = targetValue - m_outputValue;
	m_gradient = delta*Neuron::transferFunctionDerivative(m_outputValue);
}

double Neuron::sumDOW(const Layer &nextLayer)
{
	double sum = 0.0;
	
	// Sum our contributions of the errors at nodes we feed
	for(unsigned neuronNum = 0; neuronNum < nextLayer.size() - 1; neuronNum++){
		sum += m_outputWeights[neuronNum].weight * nextLayer[neuronNum].m_gradient;
	}

	return sum;
}
void Neuron::calcHiddenGradients(Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow*Neuron::transferFunctionDerivative(m_outputValue);
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
	// The weights to be updated are in the Connection container
	// in the neuron in the preceding layer

	for(unsigned neuronNum = 0; neuronNum < prevLayer.size(); neuronNum++){
		Neuron &neuron = prevLayer[neuronNum];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

			//  Individual input,magnified by the gradient and train rate:
		double newDeltaWeight = eta * neuron.getOutputValue() * m_gradient + alpha*oldDeltaWeight;
			// eta: overall net learning rate , 0.0 - slow learner, 0.2 - medium learner, 1.0 - reckless learner
			// alpha: momentum , 0.0 - no momentum, 0.5 - moderate momentum
	
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

// *******************************   class NeuralNet  *********************************
class  NeuralNet
{
public:
	NeuralNet(vector<unsigned> &topology);
	void feedForward(const vector<double> &inputValue);	
	void backProp(const vector<double> &targetValues);	
	void getResults(vector<double> &resultValues) const;	
	double getRecentAverageError(void) const { return m_recentAverageError; }
private:

	vector<Layer> m_layers; // m_layre[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
};

NeuralNet::NeuralNet(vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0; layerNum < numLayers; layerNum++){
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0:topology[layerNum+1];

		// We have made a new Layre, now fill it with ith neurons, and 
		// add a bias neuron to the layer
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++){
			m_layers.back().push_back(Neuron(numOutputs,neuronNum));
			cout << "Made a Neuron!" << endl;
		}
		// Force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputValue(1.0);
	}
}

void NeuralNet::feedForward(const vector<double> &inputValues)
{
	assert(inputValues.size() == m_layers[0].size() - 1 );
	
	for(unsigned i = 0; i < inputValues.size(); i++){
		m_layers[0][i].setOutputValue(inputValues[i]); 
	}

	// Forward propagatre
	for(unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++){
		Layer &prevLayer = m_layers[layerNum-1];					// -1 ,bias node has no input 
		for(unsigned neuronNum = 0 ; neuronNum < m_layers[layerNum].size() - 1; neuronNum++){
			m_layers[layerNum][neuronNum].feedForward(prevLayer);
		}
	
	}
}

void NeuralNet::backProp(const vector<double> &targetValues)
{
	// Calculate overall net error (RMS of output neuron errors)
	
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;
	for(unsigned neuronNum = 0; neuronNum < outputLayer.size() - 1; neuronNum++){
		double t_error = targetValues[neuronNum] - outputLayer[neuronNum].getOutputValue();
		m_error += t_error * t_error;
	}
	m_error /= outputLayer.size() - 1 ;  // get average error squared , minus 1 :bias neuron
	m_error = sqrt(m_error); // RMS --- Root mean square 
		// Implement a recent average measurement:
	m_recentAverageError =
			( m_recentAverageError *m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor+1.0);

	// Calculate output layer gradients
	for(unsigned neuronNum = 0; neuronNum < outputLayer.size() - 1; neuronNum++){
		outputLayer[neuronNum].calcOutputGradients(targetValues[neuronNum]);
	}

	// Calculate gradients on hidden layers
	for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--){
		Layer &hiddenLayer  = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum+1];
		for(unsigned neuronNum = 0; neuronNum < m_layers[layerNum].size(); neuronNum++){
			hiddenLayer[neuronNum].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to the first hidden layer,
	// update connectionn weights
	for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--){
		Layer &layer  = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum-1];
		for(unsigned neuronNum = 0; neuronNum < layer.size()-1; neuronNum++){
			layer[neuronNum].updateInputWeights(prevLayer);
		}
	}
}

void NeuralNet::getResults(vector<double> &resultValues) const
{
	resultValues.clear();

	for(unsigned neuronNum = 0; neuronNum < m_layers.back().size() -1 ; neuronNum++){
			resultValues.push_back(m_layers.back()[neuronNum].getOutputValue());
	}

}

// *******************************   function showVectorVals  *********************************
void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}



// *******************************   function main  *********************************
int main(int argc, char *argv[])
{
    TrainingData trainData("./trainingData.txt");

    // e.g., { 3, 2, 1 }
    vector<unsigned> topology;
    trainData.getTopology(topology);

    NeuralNet myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "
                << myNet.getRecentAverageError() << endl;
    }

    cout << endl << "Done" << endl;
}
