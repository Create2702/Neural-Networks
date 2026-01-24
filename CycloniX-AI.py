print('--- inputs ---')
inputs = [float(input('Ilnur: ')), float(input('Kirill: ')), float(input('Rome: '))]
print('--- targets ---')
targets = [float(input('Ilnur: ')), float(input('Kirill: ')), float(input('Rome: '))]
epochs = int(input('Enter a quantity of epochs: '))

weights = [[0.1, 0.0, 0.9],
           [0.7, 0.5, 0.2],
           [0.4, 0.1, 0.0]]

def dot(vect_a, vect_b):
    assert(len(vect_a) == len(vect_b))
    output = 0
    for i in range(len(vect_a)):
        output += vect_a[i] * vect_b[i]
    return output

def matrix_mul(vect, matrix):
    assert(len(vect) == len(matrix))
    output = [0, 0, 0]
    for i in range(len(vect)):
        output[i] = dot(vect, matrix[i])
    return output

def find_error(result_vect, target_vect):
    assert(len(result_vect) == len(target_vect))
    lose = [0, 0, 0]
    for i in range(len(result_vect)):
        lose[i] = (result_vect[i] - target_vect[i]) ** 2
    return lose

def find_delta(result_vect, target_vect):
    assert(len(result_vect) == len(target_vect))
    delta = [0, 0, 0]
    for i in range(len(result_vect)):
        delta[i] = result_vect[i] - target_vect[i]
    return delta

def correct_weight(input_data, delta, current_weights):
    assert(len(input_data) == len(delta))
    alpha = 0.01
    new_weights = [[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]]
    for i in range(len(new_weights)):
        for j in range(len(new_weights)):
            new_weights[j][i] = input_data[i] * delta[j]
    for string in range(len(new_weights)):
        for ele in range(len(new_weights)):
            current_weights[ele][string] -= new_weights[ele][string] * alpha
    return current_weights

def start(input_data, weight_data, target_data):
    for learn in range(epochs):
        pred = matrix_mul(input_data, weight_data)
        delta = find_delta(pred, target_data)
        weight_data = correct_weight(input_data, delta, weight_data)
    print('Done!')
    return weight_data

weights = start(inputs, weights, targets)
print(matrix_mul(inputs, weights))
