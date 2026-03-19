import math

inputs = [0.05, 0.1]
target_outputs = [0.01, 0.99]

netI0 = 0.1 * (inputs[0] + inputs[1]) + 0.5
netI1 = 0.1 * (inputs[0] + inputs[1]) + 0.5
outI0 = 1 / (1 + pow(math.e, -netI0))
outI1 = 1 / (1 + pow(math.e, -netI1))

netH0 = 0.1 * (outI0 + outI1) + 0.5
netH1 = 0.1 * (outI0 + outI1) + 0.5
outH0 = 1 / (1 + pow(math.e, -netH0))
outH1 = 1 / (1 + pow(math.e, -netH1))

netO0 = 0.1 * (outH0 + outH1) + 0.5
netO1 = 0.1 * (outH0 + outH1) + 0.5
outO0 = 1 / (1 + pow(math.e, -netI0))
outO1 = 1 / (1 + pow(math.e, -netI1))

error_total = (0.5 * math.pow(target_outputs[0] - outO0, 2)) + (0.5 * math.pow(target_outputs[1] - outO1, 2))

dEtotal_dOutO0 = -1 * (target_outputs[0] - outO0)
dEtotal_dOutO1 = -1 * (target_outputs[1] - outO1)

dOutO0_dNetO0 = outO0 * (1 - outO0)
dOutO1_dNetO1 = outO1 * (1 - outO1)

dNetO0_dOutH0 = 0.1
dNetO0_dOutH1 = 0.1
dNetO1_dOutH0 = 0.1
dNetO1_dOutH1 = 0.1

dOutH0_dNetH0 = outH0 * (1 - outH0)
dOutH1_dNetH1 = outH1 * (1 - outH1)

dNetH0_dOutI0 = 0.1
dNetH0_dOutI1 = 0.1
dNetH1_dOutI0 = 0.1
dNetH1_dOutI1 = 0.1

dOutI0_dNetI0 = outI0 * (1 - outI0)
dOutI1_dNetI1 = outI1 * (1 - outI1)

cumulative = (dEtotal_dOutO0 * dOutO0_dNetO0 * ((dNetO0_dOutH0 * dOutH0_dNetH0 * dNetH0_dOutI0 * dOutI0_dNetI0 * 0.05) + (dNetO0_dOutH1 * dOutH1_dNetH1 * dNetH1_dOutI0 * dOutI0_dNetI0 * 0.05))) + (dEtotal_dOutO1 * dOutO1_dNetO1 * ((dNetO1_dOutH0 * dOutH0_dNetH0 * dNetH0_dOutI0 * dOutI0_dNetI0 * 0.05) + (dNetO1_dOutH1 * dOutH1_dNetH1 * dNetH1_dOutI0 * dOutI0_dNetI0 * 0.05)))

print(f'Cumulative {cumulative}')