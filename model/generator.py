import numpy as np


# para (bs) X sentences X words
def sentence_classifier_generator(fp, batch_size, max_sentence_len, output_size):
    batch_X = []
    batch_Y = []
    max_paragraph_len = 0
    paragraph_len = 0
    sequence_lengths = np.zeros(batch_size, dtype=np.int32)
    X = []
    Y = []
    count = 0
    for line in fp:
        if line == "\n":
            sequence_lengths[count] = len(X)
            batch_X.append(X)
            batch_Y.append(Y)
            X = []
            Y = []
            count += 1
            max_paragraph_len = max(max_paragraph_len, paragraph_len)
            if count >= batch_size:
                for i in range(count):
                    for j in range(max_paragraph_len-sequence_lengths[i]):
                        batch_X[i].append([0]*max_sentence_len)
                        y = np.zeros((output_size,))
                        y[0] = 1
                        batch_Y[i].append(y)
                yield np.array(batch_X, dtype=np.int32), np.array(batch_Y, dtype=np.int32), sequence_lengths, max_paragraph_len
                batch_X = []
                batch_Y = []
                max_paragraph_len = 0
                paragraph_len = 0
                sequence_lengths = np.zeros(batch_size, dtype=np.int32)
                X = []
                Y = []
                count = 0
        else:
            row = line.split('\t')
            X.append([int(i) for i in row[0].split(',')])
            y = np.zeros((output_size,))
            y[int(row[1])] = 1
            Y.append(y)
            paragraph_len += 1

    if count < batch_size and count > 0:
        for i in range(count):
            for j in range(max_paragraph_len - sequence_lengths[i]):
                batch_X[i].append([0] * max_sentence_len)
                y = np.zeros((output_size,))
                y[0] = 1
                batch_Y[i].append(y)
        batch_X = np.array(batch_X, dtype=np.int32) + np.zeros((batch_size - count, max_paragraph_len, max_sentence_len))
        y = np.zeros((batch_size - count, max_paragraph_len, output_size))
        y[:,:,0] = 1
        batch_Y = np.array(batch_Y, dtype=np.int32) + y
        yield np.array(batch_X, dtype=np.int32), np.array(batch_Y, dtype=np.int32), sequence_lengths, max_paragraph_len
