#include <vector>
#include "Utility.h"
#include "Network.h"

using namespace std;

int main() {
	vector< vector<float> > X_train;
	vector<float> y_train;

	Utility::loadData(X_train, y_train, "data/train_small.txt");

	Network network = Network(6, 10);

	network.train(X_train, y_train);
	network.printActivationValue();

	system("pause");
}