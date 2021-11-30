from pydub import AudioSegment
import numpy as np
from pathlib import Path
from tensorflow import data
import os


class AudioHandler:

    @staticmethod
    def print_audio_info(file_name):
        audio_segment = AudioSegment.from_file(file_name)
        print("Information of '" + file_name + "':")
        print("Sample rate: " + str(audio_segment.frame_rate) + "kHz")
        # Multiply frame_width by 8 to get bits, since it is given in bytes
        print("Sample width: " + str(audio_segment.frame_width * 8) + " bits per sample (" + str(
            int(audio_segment.frame_width * 8 / audio_segment.channels)) + " bits per channel)")
        print("Channels: " + str(audio_segment.channels))

    @staticmethod
    def get_audio_array(file_name, normalized=True, norm_bounds=(-1.0, 1.0)):
        audio_segment = AudioSegment.from_file(file_name)
        # Get bytestring of raw audio data
        raw_audio_bytestring = audio_segment.raw_data
        # Adjust sample width to accommodate multiple channels in each sample
        sample_width = audio_segment.frame_width / audio_segment.channels
        # Convert bytestring to numpy array
        if sample_width == 1:
            raw_audio = np.array(np.frombuffer(raw_audio_bytestring, dtype=np.int8))
        elif sample_width == 2:
            raw_audio = np.array(np.frombuffer(raw_audio_bytestring, dtype=np.int16))
        elif sample_width == 4:
            raw_audio = np.array(np.frombuffer(raw_audio_bytestring, dtype=np.int32))
        else:
            raw_audio = np.array(np.frombuffer(raw_audio_bytestring, dtype=np.int16))
        # Normalize the audio data
        if normalized:
            # Cast the audio data as 32 bit floats
            raw_audio = raw_audio.astype(dtype=np.float32)
            # Reduce range of audio data
            range = norm_bounds[1] - norm_bounds[0]
            raw_audio *= range / np.power(2, 8 * sample_width)

            # Shift values to match bounds
            raw_audio += norm_bounds[1] - range / 2

        # Reshape the array to accommodate multiple channels
        if audio_segment.channels > 1:
            raw_audio = raw_audio.reshape((-1, audio_segment.channels))

        return raw_audio

    @staticmethod
    # Return an array of all audio files in directory, as arrays of audio data
    def get_audio_arrays(directory, filetype='mp3', normalized=True):

        file_count_total = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]) - 1

        audio_arrays = []
        # Iterate through all audio files
        pathlist = Path(directory).glob('**/*.' + filetype)
        # Keep track of progress
        file_count = 0
        print("Loading audio files... 0%")
        for path in pathlist:
            path_string = str(path)
            audio_array = AudioHandler.get_audio_array(path_string, normalized=normalized)
            audio_arrays.append(audio_array)
            # Update Progress
            file_count += 1
            print('Loading audio files... ' + str(int(file_count/file_count_total*100)) + '%')

        return audio_arrays

    @staticmethod
    def export_to_file(audio_data_array, file_name, normalized=True, norm_bounds=(-1.0, 1.0), file_type="mp3", bitrate="256k"):
        if normalized:
            range = norm_bounds[1] - norm_bounds[0]
            audio_data_array -= norm_bounds[1] - range / 2
            audio_data_array *= np.power(2, 16) / range
        audio_data_array = audio_data_array.astype(np.int16)
        audio_data_array = audio_data_array.reshape((1, -1))[0]
        raw_audio = audio_data_array.tostring()
        audio_segment = AudioSegment(data=raw_audio, sample_width=2, frame_rate=44100, channels=2)
        audio_segment.export(file_name, format=file_type, bitrate=bitrate)

    # Splits a sequence into input values and target values
    @staticmethod
    def __split_input_target(chunk):
        input_audio = chunk[:-1]
        target_audio = chunk[1:]
        return input_audio, target_audio

    @staticmethod
    def dataset_from_arrays(audio_arrays, sequence_length, batch_size, buffer_size=10000):
        # Create main data set, starting with first audio array
        dataset = data.Dataset.from_tensor_slices(audio_arrays[0])
        dataset = dataset.batch(sequence_length + 1, drop_remainder=True)
        # Split each audio array into sequences individually,
        # then concatenate each individual data set with the main data set
        for i in range(1, len(audio_arrays)):
            audio_data = audio_arrays[i]
            tensor_slices = data.Dataset.from_tensor_slices(audio_data)
            audio_dataset = tensor_slices.batch(sequence_length + 1, drop_remainder=True)
            dataset.concatenate(audio_dataset)

        dataset = dataset.map(AudioHandler.__split_input_target)

        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

        return dataset





AUDIO_FILE = "test.mp3"
AUDIO_DIRECTORY = "AudioDataset"

#data_array = AudioHandler.get_audio_array(AUDIO_FILE)

#audio_arrays = AudioHandler.get_audio_arrays(AUDIO_DIRECTORY)

#dataset = AudioHandler.dataset_from_arrays(data_array, 22050, 64)



#print(len(audio_arrays))

#for array in audio_arrays:
    #print("-----------------------------------------------------------------")
    #for i in range(300000, 301000):
        #print(array[i])


