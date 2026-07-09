import math
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