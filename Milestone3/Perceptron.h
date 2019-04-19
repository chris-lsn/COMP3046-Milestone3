class Perceptron {
public:
	Perceptron(float bias) {
		_bias = bias;
		_weight = 0.0001;
		_activationValue = 0;
		_error = 0;
		_z = 0;
	}

	void setZ(float z) {
		_z = z;
	}

	float getZ() {
		return _z;
	}


	void setError(float error) {
		_error = error;
	}

	float getError() {
		return _error;
	}

	void setBias(float bias) {
		_bias = bias;
	}

	void setActivationValue(float activationValue) {
		_activationValue = activationValue;
	}

	void setWeight(float weight) {
		_weight = weight;
	}

	float getBias() {
		return _bias;
	}

	float getWeight() {
		return _weight;
	}

	float getActivationValue() {
		return _activationValue;
	}

private:
	float _weight;
	float _bias;
	float _activationValue;
	float _error;
	float _z;
};