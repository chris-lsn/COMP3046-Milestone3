#include <iostream>     
#include <algorithm>    
#include <vector>     
#include <ctime>        
#include <cstdlib>
#include "Perceptron.h"
#include "Utility.h"

using namespace std;

class Network {
public:
	Network(int layers, int neurons) {
		if (layers < 3) {
			cout << "The network requires at least 3 layers." << endl;
		}
		else {
			build(layers - 2, neurons); // Build hidden layers exclude input and output layers
		}
	}

	void train(vector<vector<float>> X_train, vector<float> y_train) {
		/*for (int i = 0; i < X_train.size(); i++) {
			feedforward(X_train[i], y_train[i]);
		}*/

		SGD(X_train, y_train, 10, 5, 0.7);
	}


	void feedforward(vector<float> X, float y) {
		// Feed forward from input layer to first hidden layer
		for (Perceptron &perceptron : hiddenLayers[0]) {
			float weightedSum = 0; // £U w*a 
			for (float pixel : X) {
				weightedSum += perceptron.getWeight() * pixel; // w*a
			}
	
			perceptron.setActivationValue(Utility::sigmoid(weightedSum + perceptron.getBias()));
			perceptron.setZ(weightedSum + perceptron.getBias());
		}

		//cout << to_string(hiddenLayers[0][0].getActivationValue()) << endl;

		// Feed forward from first hidden layer to last hidden layer
		for (int i = 1; i < hiddenLayers.size(); i++) {
			for (Perceptron &perceptron : hiddenLayers[i]) {
				float weightedSum = 0; // £U w*a
				for (Perceptron &perviousLayerPerceptron : hiddenLayers[i-1]) {
					weightedSum += perceptron.getWeight() * perviousLayerPerceptron.getActivationValue(); // Note page. 25
				}
				perceptron.setActivationValue(Utility::sigmoid(weightedSum + perceptron.getBias()));
				perceptron.setZ(weightedSum + perceptron.getBias());
				//cout << perceptron.getActivationValue() << ", ";
			}
		}

		// Feed forward from last hidden layer to output layer
		for (Perceptron &outputPerceptron : outputLayer) {
			float weightedSum = 0; // £U w*a
			for (Perceptron &perceptron : hiddenLayers[hiddenLayers.size() - 1]) {
				weightedSum += outputPerceptron.getWeight() * perceptron.getActivationValue(); // Note page. 25
			}
			outputPerceptron.setActivationValue(Utility::sigmoid(weightedSum + outputPerceptron.getBias()));
			outputPerceptron.setZ(weightedSum + outputPerceptron.getBias());
		}
		
		
		backprop(X, y);
	}

	void SGD(vector< vector<float> > X_train, vector<float> y_train, int epochs, int mini_batch_size, float learningRate) {
		vector<vector<vector<float>>> X_train_mini_batches;
		vector<vector<float>> y_train_mini_batches;

		
		for (int j = 0; j < epochs; j++) {
			srand(42);
			random_shuffle(X_train.begin(), X_train.end());
			srand(42);
			random_shuffle(y_train.begin(), y_train.end()); // Randomly shuffle training data
			for (int k = 0; k < X_train.size(); k += mini_batch_size) {
				// Split the training data into (length of training data / mini_batch_size) parts
				vector<vector<float>> X_batch;
				vector<float> y_batch;
				for (int z = k; z < mini_batch_size; z++) {
					X_batch.push_back(X_train[z]);
					y_batch.push_back(y_train[z]);
				}
				X_train_mini_batches.push_back(X_batch);
				y_train_mini_batches.push_back(y_batch);
			}
		}
		
		for (int p = 0; p < X_train_mini_batches.size(); p++) {
			update_mini_batch(X_train_mini_batches[p], y_train_mini_batches[p], learningRate);
		}
	}

	void update_mini_batch(vector<vector<float>> X_train_mini_batch, vector<float> y_train_mini_batch, float learningRate) {
		for (int i = 0; i < X_train_mini_batch.size(); i++) {
			feedforward(X_train_mini_batch[0], y_train_mini_batch[0]);
			backprop(X_train_mini_batch[0], y_train_mini_batch[0]);
		}
	}

	void backprop(vector<float>x, float y) {
		// Last layer
		for (Perceptron &perceptron : outputLayer) {
			// cout << 0.5 * pow(y - perceptron.getActivationValue(), 2) << " " << y << " " << perceptron.getActivationValue() << endl;
			//float cost = 0.5 * pow(y - perceptron.getActivationValue(), 2);
		
			float delta = (perceptron.getActivationValue() - y) * Utility::sigmoid_d(perceptron.getActivationValue()); // Output Error
			perceptron.setBias(delta);

			float weight = 0;
			for (Perceptron &p : hiddenLayers[hiddenLayers.size() - 1]) {
				weight += delta * Utility::sigmoid_d(p.getZ());
			}
			perceptron.setWeight(weight);
		}

		// Second-last layer
		for (int i = hiddenLayers.size() - 2; i >= 1; i--) {

			for (int j = 0; j < hiddenLayers[i].size(); j++) {
				float delta = 0;
				float weight = 0;
				for (int z = 0; z < hiddenLayers[i + 1].size(); z++) {
					delta += hiddenLayers[i + 1][z].getWeight() * hiddenLayers[i + 1][z].getBias(); // Bckpropagate error
				}
				
				for (int k = 0; k < hiddenLayers[i - 1].size(); k++) {
					weight += hiddenLayers[i - 1][k].getBias() * Utility::sigmoid_d(hiddenLayers[i - 1][k].getZ());
				}
				
				hiddenLayers[i][j].setBias(delta);
				hiddenLayers[i][j].setWeight(weight);
			}
		}
	}

	void printActivationValue() {
		 for (int i = 0; i < hiddenLayers.size(); i++) {
			string output = "Hidden Layer activationValue: " + to_string(i + 1) + ": [";
			for (Perceptron perceptron : hiddenLayers[i]) {
				output += to_string(perceptron.getActivationValue());
				output += ", ";
			}
			output += "]";
			cout << output << endl;
		} 

		
		for (int i = 0; i < hiddenLayers.size(); i++) {
			string output = "\nHidden Layer bias: " + to_string(i + 1) + ": [";
			for (Perceptron perceptron : hiddenLayers[i]) {
				output += to_string(perceptron.getBias());
				output += ", ";
			}
			output += "]";
			cout << output << endl;
		}
		 
		
		for (int i = 0; i < hiddenLayers.size(); i++) {
			string output = "\nHidden Layer weight: " + to_string(i + 1) + ": [";
			for (Perceptron perceptron : hiddenLayers[i]) {
				output += to_string(perceptron.getWeight());
				output += ", ";
			}
			output += "]";
			cout << output << endl;
		}

		// Print output layer bias
		cout << "\n Output layer bias: " << endl;
		for (Perceptron p : outputLayer) {
			cout << p.getBias() << " ";
		}


		cout << "\n Output layer weight: " << endl;
		for (Perceptron p : outputLayer) {
			cout << p.getWeight() << " ";
		} 
	}

private:
	vector<vector<Perceptron>> hiddenLayers;
	vector<Perceptron> outputLayer;
	vector<float> cost;

	// A function used to build ANN
	void build(int layers, int neurons) {
		// 1. Build hidden layers
		for (int layer = 0; layer < layers; layer++) {
			vector<Perceptron> tempHiddenLayer;
			for (int neuron = 0; neuron < neurons; neuron++) {
				tempHiddenLayer.push_back(Perceptron(1.0)); // Bias will be changed later
			}
			hiddenLayers.push_back(tempHiddenLayer);
		}

		// 2. Build output layer which contains 0-9
		for (int i = 0; i <= 9; i++) {
			outputLayer.push_back(Perceptron(1.0)); 
		}
	}

	
};