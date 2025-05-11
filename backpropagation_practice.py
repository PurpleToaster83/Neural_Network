import math
import numpy as np
from sympy import symbols

def propogation(learning_rate, sys_input, desired_output, weights, biases):
    net = symbols('net')
    logistic_funct = 1 / (1 + pow(math.e, net))

    #weights = np.random.rand(8)
    #biases = np.random.rand(4)
    loop_continue = True

    #non-specific function to calculate the net and output signal of neurons
    def results(weight_values):
        netH1 = sys_input[0] * weight_values[0] + sys_input[1] * weight_values[1] + biases[0]
        outH1 = logistic_funct.subs({net: -netH1})

        netH2 = sys_input[0] * weight_values[2] + sys_input[1] * weight_values[3] + biases[1]
        outH2 = logistic_funct.subs({net: -netH2})

        netO1 = outH1 * weight_values[4] + outH2 * weight_values[5] + biases[2]
        outO1 = logistic_funct.subs({net: -netO1})

        netO2 = (outH1 * weight_values[6]) + (outH2 * weight_values[7]) + biases[3]
        outO2 = logistic_funct.subs({net: -netO2})

        return [netH1, outH1, netH2, outH2, netO1, outO1, netO2, outO2]

    #non-specific function to calculate the error of the system
    def calc_error(calc_results):
        errorO1 = pow((desired_output[0] - calc_results[5]), 2) / 2
        errorO2 = pow((desired_output[1] - calc_results[7]), 2) / 2
        error = errorO1 + errorO2
        return error

    print(f'Original Error: {calc_error(results(weights))}')
    print(f'Original Biases: {biases}')
    print(f'Original Weights: {weights}')

    while loop_continue:
        result = results(weights)

        #Errors with respect to out
        dErrorO1_dOutO1 = -1 * (desired_output[0] - result[5])
        dErrorO2_dOutO2 = -1 * (desired_output[1] - result[7])
        #Out with respect to net
        dOutO1_dNetO1 = result[5] * (1 - result[5])
        dOutO2_dNetO2 = result[7] * (1 - result[7])
        dOutH1_dNetH1 = result[1] * (1 - result[1])
        dOutH2_dNetH2 = result[3] * (1 - result[3])
        #Net with respect to previous out
        dNetO1_dOutH1 = weights[4]
        dNetO2_dOutH1 = weights[6]
        dNetO1_dOutH2 = weights[5]
        dNetO2_dOutH2 = weights[7]
        #net with respect to weight
        dNetH_dW1_3 = sys_input[0]
        dNetH_dW2_4 = sys_input[1]
        dNetO_dW5_7 = result[1]
        dNetO_dW6_8 = result[3]

        #biases recalculations
        dErrorTotal_dB1 = dErrorO1_dOutO1 * dOutO1_dNetO1 * dNetO1_dOutH1 * dOutH1_dNetH1
        dErrorTotal_dB2 = dErrorO2_dOutO2 * dOutO2_dNetO2 * dNetO2_dOutH2 * dOutH2_dNetH2
        dErrorTotal_dB3 = dErrorO1_dOutO1 * dOutO1_dNetO1
        dErrorTotal_dB4 = dErrorO2_dOutO2 * dOutO2_dNetO2

        #wieghts recalculations
        dErrorTotal_dW1 = ((dErrorO1_dOutO1 * dOutO1_dNetO1 * dNetO1_dOutH1) + (dErrorO2_dOutO2 * dOutO2_dNetO2 * dNetO2_dOutH1)) * dOutH1_dNetH1 * dNetH_dW1_3
        dErrorTotal_dW2 = ((dErrorO1_dOutO1 * dOutO1_dNetO1 * dNetO1_dOutH1) + (dErrorO2_dOutO2 * dOutO2_dNetO2 * dNetO2_dOutH1)) * dOutH1_dNetH1 * dNetH_dW2_4
        dErrorTotal_dW3 = ((dErrorO1_dOutO1 * dOutO1_dNetO1 * dNetO1_dOutH2) + (dErrorO2_dOutO2 * dOutO2_dNetO2 * dNetO2_dOutH2)) * dOutH2_dNetH2 * dNetH_dW1_3
        dErrorTotal_dW4 = ((dErrorO1_dOutO1 * dOutO1_dNetO1 * dNetO1_dOutH2) + (dErrorO2_dOutO2 * dOutO2_dNetO2 * dNetO2_dOutH2)) * dOutH2_dNetH2 * dNetH_dW2_4
        dErrorTotal_dW5 = dErrorO1_dOutO1 * dOutO1_dNetO1 * dNetO_dW5_7
        dErrorTotal_dW6 = dErrorO1_dOutO1 * dOutO1_dNetO1 * dNetO_dW6_8
        dErrorTotal_dW7 = dErrorO2_dOutO2 * dOutO2_dNetO2 * dNetO_dW5_7
        dErrorTotal_dW8 = dErrorO2_dOutO2 * dOutO2_dNetO2 * dNetO_dW6_8

        step_sizes = [dErrorTotal_dW1, dErrorTotal_dW2, dErrorTotal_dW3, dErrorTotal_dW4, dErrorTotal_dW5, dErrorTotal_dW6, dErrorTotal_dW7, dErrorTotal_dW8]
        bStep_sizes = [dErrorTotal_dB1, dErrorTotal_dB2, dErrorTotal_dB3, dErrorTotal_dB4]

        threshold_count = 0
        for i, step in enumerate(step_sizes):
            step = step * learning_rate
            weights[i] = weights[i] - step
            if(abs(step) <= 0.00001):
                threshold_count += 1

        for j, bStep in enumerate(bStep_sizes):
            bStep = bStep * learning_rate
            biases[j] = biases[j] - bStep
            if(abs(step) <= 0.00001):
                threshold_count += 1

        if(threshold_count == len(weights) + len(biases)):
            loop_continue = False

    print(f'New Error: {calc_error(results(weights))}')
    print(f'New Weights: {weights}')
    print(f'New Biases: {biases}')

class Neuron():
    def __init__(self, inputs, bias, incoming_weights, affects):
        self.inputs = inputs
        self.bias = bias
        self.incoming_weights = incoming_weights
        self.affects = affects
        self.length = len(self.inputs)
        self.net = 0
        self.out = 0

    def updateNet(self):
        net = 0
        for i in range(self.length):
            net += self.inputs[i]
        net += self.bias
        self.net = net

    def updateOut(self):
        self.out = 1 / (1 + pow(math.e, -self.net))

    def getNet(self):
        return self.net

    def getOut(self):
        return self.out

    def dOut_dNet(self):
        return self.out * (1 - self.out)

    def getInputOfPrev(self, prev_n):
        for w, weight in enumerate(self.inputs):
            if(self.incoming_weights[w] * prev_n.getOut() == weight):
                return self.incoming_weights[w]
        return

    def getBias(self):
        return self.bias

class Layer():
    def __init__(self, num_neurons, prev_layer, layer_weights, biases, layer_type='hidden'):
        self.num_neurons = num_neurons
        self.prev_layer = prev_layer
        self.layer_weights = layer_weights
        self.biases = biases
        self.neurons = []
        self.outputs = []
        self.partial_dev = []
        self.layer_type = layer_type

    def weight_inputs(self):
        neuron_inputs = []
        if(self.layer_type == 'input'):
            for active in range(len(self.prev_layer) * self.num_neurons): #Think about using len(self.layer_weights)
                neuron_inputs.append(self.prev_layer[int(active / self.num_neurons)] * self.layer_weights[active])
        else:
            for active in range(self.prev_layer.num_neurons * self.num_neurons):
                neuron_inputs.append(self.prev_layer.outputs[int(active / self.num_neurons)] * self.layer_weights[active])
        return neuron_inputs

    def createNeurons(self):
        weights = self.weight_inputs()
        for z in range(self.num_neurons):
            active_neuron_weights = []
            original_weights = []
            weight_history = []

            for k in range(0, len(weights), int(len(weights) / self.prev_layer.num_neurons)):
                active_neuron_weights.append(weights[k + z])
                original_weights.append(self.layer_weights[k + z])
            weight_history = original_weights
            for neuron in self.prev_layer.neurons:
                for element in neuron.affects:
                    weight_history.append(element)
            self.neurons.append(Neuron(active_neuron_weights, self.biases[z], original_weights, weight_history))

    def createInputNeurons(self):
        weights = self.weight_inputs()
        for z in range(self.num_neurons):
            active_neuron_weights = []
            original_weights = []

            for k in range(0, len(weights), int(len(weights) / len(self.prev_layer))):
                active_neuron_weights.append(weights[k + z])
                original_weights.append(self.layer_weights[k + z])
            self.neurons.append(Neuron(active_neuron_weights, self.biases[z], self.prev_layer, original_weights))

    def setLayerOutputs(self):
        for o in range(self.num_neurons):
            self.neurons[o].updateNet()
            self.neurons[o].updateOut()
            self.outputs.append(self.neurons[o].getOut())
    
    def getNumNeurons(self):
        return self.num_neurons

    def addPartial(self, dev):
        self.partial_dev.append(dev)

class Network():
    def __init__(self, sys_inputs, target_output, learning_rate):
        self.sys_inputs = sys_inputs
        self.target_output = target_output
        self.network_weights = []
        self.network_biases = []
        self.layers = []
        self.learning_rate = learning_rate
        self.weight_dict = {}

    def addLayer(self, num_neurons, layer_type = 'hidden'):
        new_weights = []

        if layer_type == 'input':
            prev_layer_neurons = len(self.sys_inputs)
        else:
            prev_layer_neurons = self.layers[-1].getNumNeurons()

        for w in range((num_neurons * prev_layer_neurons)):  
            new_weights.append((np.random.rand() + 0.01) * (np.random.randint(5) + 0.01)) #TODO: figure out how to do better numbers
        self.network_weights.append(new_weights)

        new_biases = []
        for b in range(num_neurons):
            new_biases.append(np.random.rand() * np.random.randint(10))

        self.network_biases.append(new_biases)

        if layer_type != 'input':
            self.layers.append(Layer(num_neurons, self.layers[-1], new_weights, new_biases))
            self.layers[-1].createNeurons()
        else:
            self.layers.append(Layer(num_neurons, self.sys_inputs, new_weights, new_biases, 'input'))
            self.layers[-1].createInputNeurons()
        self.layers[-1].setLayerOutputs()

    def totalError(self):
        error = 0
        for k in range(len(self.layers[-1].num_neurons) - 1):
            error += (abs(self.target_output[k] - self.layers[-1].neurons[k].getOut()))

        return error

    # first calculate all the partial derivatives then use pathing for proper sequence
    def neuron_pathing(self, weight):
        path = []

        for layer in self.layers:
            for neuron in layer.neurons:
                if self.weight_dict.get(weight) in neuron.affects:
                    path.append(neuron)
        return path

    def labelWeights(self):
        wCount = 0
        print()

        for nLW, netLWeight in enumerate(self.network_weights):
            for lW, layerWeight in enumerate(netLWeight):
                self.weight_dict.update({wCount : layerWeight})
                wCount += 1

    def printInfo(self, arch=False, in_out=False, labelW=False, lookFor=None, dispPart=False, error=False):
        wCount = 0

        if arch:
            print('Architecture')
            for l, index_l in enumerate(self.layers):
                print(index_l)
                for n, index_n in enumerate(index_l.neurons):
                    print(f'\t{index_n}')
            print()

        if in_out:
            print('Neuron Input/Output:')
            for l, layer in enumerate(self.layers):
                for n, neuron in enumerate(layer.neurons):
                    print(f'\tL{l} N{n} Net: {neuron.getNet()}')
                    print(f'\tL{l} N{n} Out: {neuron.getOut()}')
                    print(f'\tL{l} N{n} Bias: {neuron.getBias()}')
            print()

        if labelW:
            print("Labeled Weights:")
            for nLW in self.weight_dict:
                print(f'\tW{nLW} --> {self.weight_dict.get(nLW)}')
            print()

        if lookFor:
            path = self.neuron_pathing(lookFor)

            print(f'Weight Path for weight {lookFor}')
            for segment in path:
                print(f'\t{segment}')
            print(f'\tPath length: {len(path)}')
            print()
        
        if dispPart:
            for l, layer in enumerate(self.layers):
                print(f'L{l} Partial Derv. : {layer.partial_dev}')
            print()

        if(error):
            print(f'System Error: {self.getError()}')
            print()

    def calc_all_partials(self):
        # calculate dError_dOut
        for n, neuron in enumerate(self.layers[-1].neurons):
            self.layers[-1].addPartial(-1 * (self.target_output[n] - neuron.getOut()))

        for layer in self.layers:

            # calculate dOut_dNet for layers
            for neuron in layer.neurons:
                layer.addPartial(neuron.getOut() * (1 - neuron.getOut()))

             # calculate dNet_dOut for layers
            if(layer != self.layers[0]):
                for w in layer.layer_weights:
                    layer.addPartial(w)
            else:
                for w in layer.layer_weights:
                    layer.addPartial(self.sys_inputs[int(layer.layer_weights.index(w) / len(self.sys_inputs))]) #TODO: recheck last layer partials

    def getError(self):
        sum = 0
        for i in range(len(self.target_output)):
            sum += (1 / len(self.target_output)) * pow((self.target_output[i] - self.layers[-1].neurons[i].getOut()), 2)
            return sum

    def cumulative_partial(self, weight, layer): #start with just one so easier to debug
        weight_path = self.neuron_pathing(weight)

        if not isinstance(layer, Layer):
            return [weight_path[0].getOut()]

        current = []
        #TODO: actually make work (follow the path!!!!!)
        for n, neuron in enumerate(layer.neurons):
            current.append(layer.partial_dev[n] * layer.partial_dev[n + 2]) #dErrorT_dNetO
        # sum these at the end

        elemts_of_active = []
        for curr_par in current:
            prevLayer_pars = self.cumulative_partial(weight, layer.prev_layer)
            for past_par in prevLayer_pars:
                elemts_of_active.append(curr_par * past_par)
        return elemts_of_active

        #TODO: figure out how to multiply/add partials

def main():
    learning_rate = 0.5
    sys_input = [0.05, 0.10]
    target_output = [0.01, 0.99]
    propagation_weights = [0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55]
    propagation_biases = [0.35, 0.35, 0.60, 0.60]
    weight_seek = 5

    #TODO: turned off suggestions for practicing for AP test

    # create a network and instatiate the weights, biases, and nuerons
    network = Network(sys_input, target_output, learning_rate)
    network.addLayer(2, layer_type='input')
    network.addLayer(2)
    network.addLayer(len(target_output))

    network.labelWeights()
    network.neuron_pathing(weight_seek)
    network.calc_all_partials()
    print(network.cumulative_partial(0, network.layers[-1]))
    network.printInfo()

    #need a way to save the model once parameters have been optimized

    # run the test propogate on 2 x 2 network
    # propogation(
    #            learning_rate,
    #            sys_input,
    #            target_output,
    #            propagation_weights,
    #            propagation_biases
    #            )

if __name__ == "__main__":
    main()
