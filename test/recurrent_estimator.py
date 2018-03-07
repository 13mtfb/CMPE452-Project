# standard files
import tensorflow as tf
import pathlib as pl

# custom files
import stock_data

# paths to data
TRAIN_PATH = 'data/SP500_daily.csv'
TEST_PATH = 'data/SP500_daily.csv'

#NN properties
hidden_size = 2
batch_size = 1000

def main(argv):
    my_file = pl.Path(TRAIN_PATH)
    if my_file.is_file():
        # file exists
        dataset = stock_data.csv_input_fn(TRAIN_PATH,batch_size)
    else :
        print ("file does not exist")
    print(dataset)

    # create a BasicRNNCell
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

    # define initial state
    initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

    # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
    # 'state' is a tensor of shape [batch_size, cell_state_size]
    outputs, state = tf.nn.dynamic_rnn(rnn_cell,
                                   dataset,
                                   initial_state=initial_state,
                                   dtype=tf.float32)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
