import math
import numpy as np

def matrix_mult(matrix_a, matrix_b): #TODO: use straussen algorithm instead to make faster
    is_array = False
    a = len(matrix_a)

    if type(matrix_a[0]) != list:
        is_array = True
        a = 1
        matrix_a = [matrix_a]

    if type(matrix_b[0]) != list:
        matrix_b = [matrix_b]

    matrix_r = [[0 for i in range(len(matrix_b[0]))] for j in range(a)]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_a[0])):
                matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j]

    if is_array:
        matrix_r = matrix_r[0]

    return matrix_r

def matrix_add(matrix_a, matrix_b):
    new_matrix = []
    for e in range(len(matrix_a)):
        new_matrix.append(matrix_a[e] + matrix_b[e])
    return new_matrix

class Layer():
    def __init__(self, num_neurons, prev_layer, layer_weights, biases, drop_r=0.5):
        self.num_neurons = num_neurons
        self.prev_layer = prev_layer
        self.layer_weights = layer_weights
        self.biases = biases
        self.drop_r = drop_r
        self.outputs = []
        self.drop_store = {
            'cur': {},
            'for': {}
        }

    def setLayerOuts(self):
        net = matrix_add(matrix_mult(self.prev_layer.outputs, self.layer_weights), self.biases)
        out = []
        for n in net:
            o = 1 / (1 + pow(math.e, -n))
            out.append(o)
        self.outputs = out

    def insert(self, n, forward=False):
        if forward:
            # insert values into the forward layer
            self.layer_weights.append(self.drop_store['for'].pop(n)['w'])

            # call insert for the back layer
            self.prev_layer.insert(n)
        else:
            self.num_neurons += 1

            # grab dropped values from storage
            values = self.drop_store['cur'].pop(n)

            # put values back into matrices
            self.outputs.append(values['out'])
            self.biases.append(values['b'])

            for r, row in enumerate(self.layer_weights):
                #TODO: if flag, skip
                row.append(values['w'][r])

    def pop(self, glob_n, loc_n, forward=False):
        if forward:

            # pop weights for the forward layer
            self.drop_store['for'].update({
                glob_n : {
                    'w': self.layer_weights.pop(loc_n)
                }
            })

            # pop values for the back layer
            self.prev_layer.pop(glob_n, loc_n)
        else:
            self.num_neurons -= 1

            # remove pop values and store as new variables
            out = self.outputs.pop(loc_n)
            b = self.biases.pop(loc_n)

            pop_weights = []
            for row in self.layer_weights:
                w = row.pop(loc_n)
                pop_weights.append(w)

            # store dict of values
            self.drop_store['cur'].update({
                glob_n: {
                    'w': pop_weights,
                    'b': b,
                    'out': out
                }
            })        

class Network():
    def __init__(self, sys_inputs, target_output, learning_rate, drop_rate, error_threshold):
        self.sys_inputs = sys_inputs
        self.target_output = target_output
        self.error_threshold = error_threshold
        self.drop_rate = drop_rate
        self.network_weights = []
        self.network_biases = []
        self.layers = []
        self.learning_rate = learning_rate

    def truncate(self, number, decimals):
        factor = pow(10, decimals)
        return int(number * factor) / factor

    def size(self, matrix):
        return len(matrix) * len(matrix[0])

    def getError(self):
        sum = 0
        for i in range(len(self.target_output)):
            sum += (1 / len(self.target_output)) * pow((self.target_output[i] - self.layers[-1].outputs[i]), 2)
            return sum

    def addLayer(self, num_neurons, layer_type='hidden'):
        if layer_type == 'input':
            prev_layer_neurons = len(self.sys_inputs)
        else:
            prev_layer_neurons = self.layers[-1].num_neurons

        # create random weight values
        new_weights = []
        for r in range(prev_layer_neurons):
            row = []
            for w in range(num_neurons):
                row.append(self.truncate((np.random.rand() + 0.01) * (np.random.randint(5) + 0.01), 3)) #TODO: reimplement later, also don't use np
            new_weights.append(row)
        self.network_weights.append(new_weights)

        # create random bias values
        new_biases = []
        for b in range(num_neurons):
            new_biases.append(self.truncate(np.random.rand() * np.random.randint(10), 3)) #TODO: reimplement later, also don't use np
        self.network_biases.append(new_biases)

        # make a new layer with the weights and biases
        if layer_type == 'input':
            ghost_layer = Layer(len(self.sys_inputs), None, None, None)
            ghost_layer.outputs = self.sys_inputs
            self.layers.append(Layer(num_neurons, ghost_layer, new_weights, new_biases))
        else:
            self.layers.append(Layer(num_neurons, self.layers[-1], new_weights, new_biases))

        self.layers[-1].setLayerOuts()

    def cumulative_partial(self, layer_num, w=None, b=None):
        layer = self.layers[layer_num]

        if w is not None:
            # dNet_dWeight
            dInit = layer.prev_layer.outputs[int((w % self.size(layer.layer_weights)) / layer.num_neurons)]

            # dOut_dWeight
            z = w % layer.num_neurons 
            out = layer.outputs[z]
            dInit *= out * (1 - out)
        else:
            # dOut_dBias
            dInit = layer.biases[b]
            dInit *= layer.outputs[b] * (1 - layer.outputs[b])

        path = []

        if layer != self.layers[-1]:
            n = int((w / self.layers[layer_num].prev_layer.num_neurons) if (w is not None) else b)
            path.append(self.layers[layer_num + 1].layer_weights[n])

        for l in range(layer_num + 1, len(self.layers)):
            current_layer = self.layers[l]
            m = []

            # append a diagonal matrix for dOut_dNet
            for o, out in enumerate(current_layer.outputs):
                sub_m = []
                for j in range(current_layer.num_neurons):
                    if j == o:
                        sub_m.append(out * (1 - out))
                    else:
                        sub_m.append(0)
                m.append(sub_m)
            path.append(m)

            # append the weight matrix for the layer
            if current_layer != self.layers[-1]:
                path.append(self.layers[l+1].layer_weights)
            else:
                break

        #dE_dOut
        if layer != self.layers[-1]:
            m = []
            for o, out in enumerate(self.layers[-1].outputs):
                m.append([(-1 * (self.target_output[o] - out))])
            path.append(m)
        else:
            # if parameter in last layer error dependent on one neuron
            idx = (w if w is not None else b) % self.layers[-1].num_neurons
            resid = -1 * (self.target_output[idx] - self.layers[-1].outputs[idx])
            path.append(self.layers[-1].outputs[idx] * (1 - self.layers[-1].outputs[idx]))
            path.append(resid)

        # multiply the matrix elements of path togeather to get a scalar
        if layer != self.layers[-1]:
            starter = []
            for e in path[0]:
                starter.append(e * dInit)

            running = starter
            for element in range(len(path) - 2):
                running = matrix_mult(running, path[element + 1])

            total = 0
            for r, run in enumerate(running):
                total += (run * path[-1][r][0])
        else:
            total = dInit * path[0] * path[-1]

        return total

    def updateAll(self):
        # clear the new weight and bias network arrays
        new_network_weights = []
        new_network_biases = []

        for l, layer in enumerate(self.layers):

            # calculate new weight by w' = w - a * dE_dW
            new_layer_weights = []
            for rw, row in enumerate(layer.layer_weights):
                new_row = []
                for w, weight in enumerate(row):
                    new_row.append(weight - self.learning_rate * self.cumulative_partial(l, w=(rw * layer.num_neurons) + w))
                new_layer_weights.append(new_row)

            # calculate new bias by b' = b - a * dE_dB
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

        # update all parameters until the error is below the threshold
        while current_error >= self.error_threshold:
            self.updateAll()
            current_error = self.getError()

    def drop(self):

        # decide weither to drop or not based on rate
        exit = np.random.choice([True, False], p=[1 - self.drop_rate, self.drop_rate])
        if exit:
            return

        # establish rates + options and count neurons
        rates = []
        index_vals = []
        total_neurons = 0
        for i, a in enumerate(self.layers):
            if i == len(self.layers) - 1:
                break

            for j in range(a.num_neurons):
                index_vals.append((i, j))
                rates.append(a.drop_r)
            total_neurons += a.num_neurons


        # turn rate into a softmax probability distribution
        all = sum(pow(math.e, x) for x in rates)
        softmax = []
        for y in rates:
            softmax.append(pow(math.e, y) / all)

        options = []
        for n in range(total_neurons):
            options.append(n)

        num = np.random.randint(low=0, high=total_neurons) #TODO: make this around a normal distribution and skew more downward
        picks = np.random.choice(options, size=num, p=softmax, replace=False)

        affected_layers = {}
        for pick in picks:
            pair = index_vals[pick] # this is your (l, n) tuple
            layer_num = pair[0]
            n = pair[1]

            # keep track of neurons dropped in each layer
            if affected_layers.get(layer_num):
                affected_layers[layer_num]['times'] += 1
                affected_layers[layer_num]['neurons'].append(n)
            else:
                affected_layers.update({
                    layer_num: {
                        'times': 1,
                        'neurons': [n]
                    }
                })

        for l, layer in enumerate(self.layers):

            vals = affected_layers.get(l)
            if not vals:
                continue

            # makes sure can't remove all neurons from a layer
            if vals['times'] == layer.num_neurons:
                r = np.random.randint(vals['times'])
                vals['times'] -= 1
                vals['neurons'].pop(r)

            # remove neurons from layer
            for off, glob in enumerate(vals['neurons']):
                self.layers[l+1].pop(glob, glob-off, True)

        self.updateAll()

        #TODO: ssue with conflict between for and cur both trying to grab specific weights (only one currently have)

        for l, layer in enumerate(self.layers):

            vals = affected_layers.get(l)
            if not vals:
                continue

            for n in vals['neurons']:
                self.layers[l+1].insert(n, True)

        print('blah')

        # update drop rates for layers that have been choosen
        for l, layer in enumerate(self.layers):
            if affected_layers.get(l):
                t = affected_layers[l]['times']

                # linear - active
                layer.drop_r *= 1 - (t / total_neurons)

                # concave up - prev
                pd = self.layers[l - 1].drop_r
                exp1 = pow(math.e, -1 * (t / (total_neurons * pow(pd, 2))))
                exp2 = pow(math.e, -1 * (1 / pow(pd, 2))) * pow(t / total_neurons, 0.5)
                c = 2 * pow(math.e, -1 * (1 / pow(pd, 2))) - 1
                pd *= 1 + (exp1 - exp2 + c)

                # concave down - forward
                fd = self.layers[l + 1].drop_r
                mult = fd * pow(math.e, -1 * (1 / pow(fd, 2)))
                exp1 = pow(math.e, t / (total_neurons * pow(fd, 2)))
                exp2 = pow(-t / total_neurons, 0.5)
                fd -= mult * (exp1 - exp2 + 1)
            else:
                layer.drop_r *= 1 + (t / total_neurons)

        print('blah')

def main():
    # define constants for the network
    learning_rate = 1
    sys_input = [0.05, 0.10]
    target_output = [0.01, 0.99]
    error_threshold = 0.00001
    drop_rate = 0.3

    # create the architecture for a layered network
    network = Network(sys_input, target_output, learning_rate, drop_rate, error_threshold)
    network.addLayer(2, layer_type='input')
    network.addLayer(4)
    network.addLayer(3)
    network.addLayer(2)
    network.addLayer(len(target_output))

    # testing if can properly update when drop neurons
    network.drop()
    network.drop()


    # optimize the parameters of the network for the target
    print(f'Original Error: {network.getError()}')
    network.minimizeError()
    print(f'Updated Error: {network.getError()}')

    print('finished')

if __name__ == "__main__":
    main()
