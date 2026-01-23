inputs = [float(input('Ильнур: ')), float(input('Кирилл: ')), float(input('Рома: '))]
print('-------')
targets = [float(input('Ильнур: ')), float(input('Кирилл: ')), float(input('Рома: '))]

weights = [[0.1, 0.2, 0.3],
           [0.3, 0.2, 0.1],
           [0.0, 0.1, 0.3]]

def dot(vect_a, vect_b):
    assert(len(vect_a) == len(vect_b))
    output = 0
    for i in range(len(vect_a)):
        output += vect_a[i] * vect_b[i]
    return output

def matrix(vect, matrix):
    assert(len(vect) == len(matrix))
    output = [0, 0, 0]
    for i in range(len(matrix)):
        output[i] = dot(vect, matrix[i])
    return output

def ele_mul(const, delta):
    weights_delta = [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]
    for i in range(len(weights_delta)):
        for j in range(len(weights_delta)):
            weights_delta[j][i] = const[i] * delta[j]
    return weights_delta

def forward(input_data, weight_data, target_data):
    for i in range(10000):
        final_weights = weight_data
        assert(len(input_data) == len(weight_data))
        pred = matrix(input_data, weight_data)
        error = [0, 0, 0]
        delta = [0, 0, 0]
        for ix in range(len(input_data)):
            error[ix] = (pred[ix] - target_data[ix]) ** 2
            delta[ix] = pred[ix] - target_data[ix]
        weights_delta = ele_mul(input_data, delta)
        for ir in range(len(weights_delta)):
            for j in range(len(weights_delta)):
                final_weights[ir][j] -= (weights_delta[ir][j]) * 0.01
    return final_weights

def final(final_weights, final_inputs):
    pred = matrix(final_inputs, final_weights)
    return pred

weight = forward(inputs, weights, targets)
pred = final(weight, inputs)
print(pred)
