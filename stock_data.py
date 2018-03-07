import tensorflow as tf

CSV_COLUMN_NAMES = ['Date', 'Open','High', 'Low', 'Close', 'Volume']

CLASSES = ['Low', 'Same', 'High']

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[""], [0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Don't have labeled data
    # Currently using the 'volume' data as the label - CHANGE
    label = features.pop('Volume')

    return features, label

def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    # Skip method skips over the header line of the get_file
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
