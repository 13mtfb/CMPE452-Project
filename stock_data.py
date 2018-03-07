import tensorflow as tf

CSV_COLUMN_NAMES = ['Date', 'Open','High', 'Low', 'Close', 'Volume']

CLASSES = ['Low', 'Same', 'High']

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    #features = dict(zip(CSV_COLUMN_NAMES, fields))

    inputs = fields[:len(fields)-1] #First values, ignoring string
    label = fields[len(fields)-1:]#Last value

    inputs = tf.stack(inputs)
    label = tf.stack(label)
    print ("inputs={}".format(inputs))

    # Don't have labeled data
    # Currently using the 'volume' data as the label - CHANGE
    #label = features.pop('Volume')

    #return features, label
    return {'rawdata': inputs}, label

def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    # Skip method skips over the header line of the get_file
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(1)

    # Return the dataset.
    return dataset
