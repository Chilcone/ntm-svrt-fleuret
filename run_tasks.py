import tensorflow as tf
from generate_data import FleuretTaskData
import utils

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser.add_argument('--mann', type=str, default='ntm', help='none | ntm')
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_units', type=int, default=128, help='size of the hidden state of LSTM')
parser.add_argument('--num_memory_locations', type=int, default=128)
parser.add_argument('--memory_size', type=int, default=32)
parser.add_argument('--num_read_heads', type=int, default=2)
parser.add_argument('--num_write_heads', type=int, default=2)
parser.add_argument('--conv_shift_range', type=int, default=2, help='only necessary for ntm')
parser.add_argument('--clip_value', type=int, default=128, help='Maximum absolute value of controller and outputs.')
parser.add_argument('--init_mode', type=str, default='constant', help='learned | constant | random')

parser.add_argument('--optimizer', type=str, default='Adam', help='RMSProp | Adam')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=5)
parser.add_argument('--num_train_steps', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--num_bits_per_vector', type=int, default=32)
parser.add_argument('--num_vectors_per_image', type=int, default=64)
parser.add_argument('--dataset_size', type=int, default=40000)

parser.add_argument('--verbose', type=str2bool, default=False, help='if true stores logs')
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--job-dir', type=str, required=False)
parser.add_argument('--use_local_impl', type=str2bool, default=True, help='whether to use the repos local NTM implementation or the TF contrib version')

args = parser.parse_args()

if args.mann == 'ntm':
    if args.use_local_impl:
        from ntm import NTMCell
    else:
        from tensorflow.contrib.rnn.python.ops.rnn_cell import NTMCell

if args.verbose:
    import pickle  # library for serializing Python objects
    utils.create_directory("head_logs")
    HEAD_LOG_FILE = 'head_logs/{0}.p'.format(args.experiment_name)
    GENERALIZATION_HEAD_LOG_FILE = 'head_logs/generalization_{0}.p'.format(args.experiment_name)

class BuildModel(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self._build_model()

    def _build_model(self):
        cell = NTMCell(args.num_layers, args.num_units, args.num_memory_locations, args.memory_size,
                        args.num_read_heads, args.num_write_heads, addressing_mode='content_and_location',
                        shift_range=args.conv_shift_range, reuse=False, output_dim=args.num_bits_per_vector,
                        clip_value=args.clip_value, init_mode=args.init_mode)

        # final state (c, h) - c is the hidden state of the last LSTM cell and h is its output
        _, final_state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=self.inputs,
            time_major=False,
            dtype=tf.float32,
            initial_state= None)

        # Get the output from the last LSTM cell
        self.final_output = final_state.controller_state[0].h # shape = (batch_size, hidden_state_size)

class BuildTModel(BuildModel):
    def __init__(self, inputs, outputs):
        super(BuildTModel, self).__init__(inputs)

        # define dense layer to classify the resulting feature vector
        self.dense_layer = tf.layers.dense(inputs=self.final_output, units=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                           activation=tf.sigmoid, name="dense_fin") #shape = (batch_size, 1)

        # Get dense layer weights for Tensorboard visualization
        with tf.variable_scope('dense_fin', reuse=True):
            w = tf.get_variable('kernel')
        tf.summary.histogram("dense_weights", w)

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs, logits=self.dense_layer)
        self.loss = tf.reduce_sum(cross_entropy) / args.batch_size
        tf.summary.scalar('loss', self.loss)

        if args.optimizer == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(args.learning_rate, momentum=0.9, decay=0.9)
        elif args.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), args.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

with tf.variable_scope('root'):
    inputs_placeholder = tf.placeholder(tf.float32, shape=(args.batch_size, args.num_vectors_per_image, args.num_bits_per_vector)) # (batch_size, 64, 32)
    outputs_placeholder = tf.placeholder(tf.float32, shape=(args.batch_size, 1))
    model = BuildTModel(inputs_placeholder, outputs_placeholder)
    initializer = tf.global_variables_initializer()

data_generator = FleuretTaskData(args.batch_size)

print("Initializing TensorFlow session...")
sess = tf.Session()
sess.run(initializer)

# Enable TensorBoard graph visualization
utils.create_directory("logs")
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs", sess.graph)

if args.verbose:
    print("Creating log files...")
    pickle.dump({}, open(HEAD_LOG_FILE, "wb"))
    pickle.dump({}, open(GENERALIZATION_HEAD_LOG_FILE, "wb"))

print("Started training...")
for i in range(args.num_train_steps):
    data_generator.shuffle_dataset()
    print("Running epoch", i)
    for iter in range(args.dataset_size // args.batch_size):
        inputs, labels = data_generator.get_batch(iter, args.batch_size, i)
        summary, train_loss, _= sess.run([merged, model.loss, model.train_op],
                                          feed_dict={
                                              inputs_placeholder: inputs,
                                              outputs_placeholder: labels,
                                          })
        writer.add_summary(summary, i * (args.dataset_size // args.batch_size) + iter)
        logger.info('Train loss at step {0}, iter {1}: {2}'.format(i, iter, train_loss))
