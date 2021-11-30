import tensorflow as tf
import numpy as np
from AudioHandler import AudioHandler
import os

seq_length = 22050
BATCH_SIZE = 64
BUFFER_SIZE = 10000

audio_arrays = AudioHandler.get_audio_arrays("AudioDataset", normalized=True)

'''
for i in range(300000, 300100):
    print(audio_arrays[0][i])
'''

dataset = AudioHandler.dataset_from_arrays(audio_arrays, seq_length, BATCH_SIZE, buffer_size=BUFFER_SIZE)

x = np.zeros((64, 22050, 2), dtype=np.float32)
y = np.ones((64, 22050, 2), dtype=np.float32)

print(dataset)

'''
for input, target in dataset.take(1):
    print(input)
    print(target)
'''


rnn_units = 256

def build_model(rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(batch_size, None, 2)),

        tf.keras.layers.GRU(rnn_units, return_sequences=False, stateful=True),

        tf.keras.layers.Dense(2)
    ])
    model.build()
    return model


model = build_model(rnn_units, BATCH_SIZE)

model.summary()

'''
for input, target in dataset.take(1):
    example_batch_predictions = model(input)
    print(example_batch_predictions[0], '(batch_size, seq_length, num_channels)')
'''

model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 10

history = model.fit(dataset, epochs=EPOCHS)


'''
tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

def generate_text(model, start_num):
  # Evaluation step (generating text using the learned model)

  # Number of samples to generate
  num_generate = 441000

  input_eval = start_num

  # Empty string to store our results
  samples_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature

      samples_generated.append(predictions)

  return samples_generated
'''