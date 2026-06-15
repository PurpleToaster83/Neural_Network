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
        netH1 = sys_input[0] * weight_values[0] + sys_input[1] * weight_values[2] + biases[0]
        outH1 = logistic_funct.subs({net: -netH1})

        netH2 = sys_input[0] * weight_values[1] + sys_input[1] * weight_values[3] + biases[1]
        outH2 = logistic_funct.subs({net: -netH2})

        netO1 = outH1 * weight_values[4] + outH2 * weight_values[6] + biases[2]
        outO1 = logistic_funct.subs({net: -netO1})

        netO2 = (outH1 * weight_values[5]) + (outH2 * weight_values[7]) + biases[3]
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
        dNetO2_dOutH1 = weights[5]
        dNetO1_dOutH2 = weights[6]
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

        #weights recalculations
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

class Layer():
    def __init__(self, num_neurons, prev_layer, layer_weights, biases):
        self.num_neurons = num_neurons
        self.prev_layer = prev_layer
        self.layer_weights = layer_weights
        self.biases = biases
        self.outputs = []

    def setLayerOuts(self):
        net = self.matrix_add(self.matrix_mult(self.prev_layer.outputs, self.layer_weights), self.biases)
        out = []
        for n in net:
            o = 1 / (1 + pow(math.e, -n))
            out.append(o)
        self.outputs = out

    def matrix_mult(self, matrix_a, matrix_b): #TODO: use straussen algorithm instead to make faster, make the architecture set up with matrices
        is_array = False
        matrix_r = [[0 for i in range(len(matrix_a))] for j in range(len(matrix_b[0]))]

        if type(matrix_a[0]) != list:
            matrix_a = [matrix_a]
            is_array = True
        for i in range(len(matrix_a)):
            for j in range(len(matrix_b)):
                for k in range(len(matrix_a[0])):
                    matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j]

        if is_array:
            matrix_r = matrix_r[0]

        return matrix_r

    def matrix_add(self, matrix_a, matrix_b):
        new_matrix = []
        for e in range(len(matrix_a)):
            new_matrix.append(matrix_a[e] + matrix_b[e])
        return new_matrix

    def insert(self, key):
        #TODO
        pass

    def pop(self, n):
        #TODO
        pass

class Network():
    def __init__(self, sys_inputs, target_output, learning_rate, error_threshold):
        self.sys_inputs = sys_inputs
        self.target_output = target_output
        self.error_threshold = error_threshold
        self.network_weights = []
        self.network_biases = []
        self.layers = []
        self.learning_rate = learning_rate

    def truncate(self, number, decimals):
        factor = pow(10, decimals)
        return int(number * factor) / factor

    def addLayer(self, num_neurons, layer_type='hidden'):
        new_weights = []

        if layer_type == 'input':
            prev_layer_neurons = len(self.sys_inputs)
        else:
            prev_layer_neurons = self.layers[-1].num_neurons

        for r in range(prev_layer_neurons):
            row = []
            for w in range(num_neurons):
                row.append(self.truncate((np.random.rand() + 0.01) * (np.random.randint(5) + 0.01), 3)) #TODO: reimplement later, also don't use np
            new_weights.append(row)
        self.network_weights.append(new_weights)

        new_biases = []
        for b in range(num_neurons):
            new_biases.append(self.truncate(np.random.rand() * np.random.randint(10), 3)) #TODO: reimplement later, also don't use np
        self.network_biases.append(new_biases)

        if layer_type == 'input':
            ghost_layer = Layer(2, None, None, None)
            ghost_layer.outputs = self.sys_inputs
            self.layers.append(Layer(num_neurons, ghost_layer, new_weights, new_biases))
        else:
            self.layers.append(Layer(num_neurons, self.layers[-1], new_weights, new_biases))

        self.layers[-1].setLayerOuts()
        # self.layers[-1].setLayerOutputs()

    def cumulative_partial(self, layer_num, w=None, b=None):        
        layer = self.layers[layer_num]

        if(w != None):
        # dNet_dWeight
            if layer_num == 0:
                dInit = self.sys_inputs[int((w % self.size(layer.layer_weights)) / layer.num_neurons)]
            else:
                dInit = layer.prev_layer.outputs[int((w % self.size(layer.layer_weights)) / layer.num_neurons)]
            
            #dOut_dWeight
            out = layer.outputs[int(((w + 1) % self.size(layer.layer_weights)) / layer.num_neurons)]
            dInit *= out * (1 - out)
        else:
            dInit = layer.biases[b]
            dInit *= layer.outputs[b] * (1 - layer.outputs[b])

        path = []
        
        if(layer != self.layers[-1]): 
            n = int((w / self.layers[layer_num].num_neurons) if (w is not None) else b)
            path.append(self.layers[layer_num + 1].layer_weights[n])

        for l in range(layer_num + 1, len(self.layers)):
            current_layer = self.layers[l]
            m = []

            for o, out in enumerate(current_layer.outputs):
                sub_m = []
                for j in range(current_layer.prev_layer.num_neurons):
                    if j == o:
                        sub_m.append(out* (1 - out))
                    else:
                        sub_m.append(0)
                m.append(sub_m)
            path.append(m)

            if current_layer != self.layers[-1]:
                path.append(self.layers[l+1].layer_weights)
            else:
                break
        
        if(layer != self.layers[-1]):
            m = []
            for o, out in enumerate(self.layers[-1].outputs):
                #∂En_∂Outn * ∂Outn_∂Netn
                m.append([(-1 * (self.target_output[o] - out))])
            path.append(m)
        else:
            idx = (w if w is not None else b) % self.layers[-1].num_neurons
            resid = (-1 * (self.target_output[idx] - self.layers[-1].outputs[idx]))
            path.append(self.layers[-1].outputs[idx] * (1 - self.layers[-1].outputs[idx]))
            path.append(resid)

        if(layer != self.layers[-1]):
            starter = []
            for e in path[0]:
                starter.append(e * dInit)

            running = starter
            for element in range(len(path) - 2):
                running = self.matrix_mult(running, path[element + 1])
            
            total = 0
            for r, run in enumerate(running):
                total += (run * path[-1][r][0])
        else:
            total = dInit * path[0] * path[-1]
        
        return total
    
    def updateAll(self):
        new_network_weights = []
        new_network_biases = []
        for l, layer in enumerate(self.layers):
            new_layer_weights = []

            #TODO: this needs to make a matrix for layer_weights
            for rw, row in enumerate(layer.layer_weights):
                new_row = []
                for w, weight in enumerate(row):
                    new_row.append(weight - self.learning_rate * self.cumulative_partial(l, w=((rw * layer.num_neurons) + w)))
                new_layer_weights.append(new_row)

            new_layer_biases = []
            for b, bias in enumerate(layer.biases):
                new_layer_biases.append(bias - self.learning_rate * self.cumulative_partial(l, b=b))
            
            new_network_weights.append(new_layer_weights)
            new_network_biases.append(new_layer_biases)

        self.network_weights = new_network_weights
        self.network_biases = new_network_biases

        # distribute weights to appropriate layer
        for l, layer in enumerate(self.layers):
            layer.layer_weights = self.network_weights[l]
            layer.biases = self.network_biases[l]
            layer.setLayerOuts()

    def minimizeError(self):
        current_error = self.getError()
        while(current_error >= self.error_threshold):
            self.updateAll()
            current_error = self.getError()

    def getError(self):
        sum = 0
        for i in range(len(self.target_output)):
            sum += (1 / len(self.target_output)) * pow((self.target_output[i] - self.layers[-1].outputs[i]), 2)
            return sum

    def matrix_mult(self, matrix_a, matrix_b): #TODO: use straussen algorithm instead to make faster, make the architecture set up with matrices
        is_array = False
        matrix_r = [[0 for i in range(len(matrix_a))] for j in range(len(matrix_b[0]))]

        if type(matrix_a[0]) != list:
            matrix_a = [matrix_a]
            is_array = True
        for i in range(len(matrix_a)):
            for j in range(len(matrix_b)):
                for k in range(len(matrix_a[0])):
                    matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j]

        if is_array:
            matrix_r = matrix_r[0]

        return matrix_r

    def size(self, matrix):
        return len(matrix) * len(matrix[0])

def main():
    learning_rate = 1
    sys_input = [0.05, 0.10]
    target_output = [0.01, 0.99]
    error_threshold = 0.00001

    # create a network and instatiate the weights, biases, and neurons
    network = Network(sys_input, target_output, learning_rate, error_threshold)
    network.addLayer(2, layer_type='input')
    network.addLayer(2)
    network.addLayer(len(target_output))
    
    print(f'Original Error: {network.getError()}')
    network.minimizeError()
    print(f'Updated Error: {network.getError()}')

    print('finished')


if __name__ == "__main__":
    main()