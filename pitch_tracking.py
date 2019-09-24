import librosa
import matplotlib.pyplot as plt
import numpy as np
import glob
import time as t
import crepe


sample_rate = 8000 # sample rate of signal
hop = 0.1 # seconds per hop
model_capacity = 'medium' # model size of CREPE, has to be one of ['tiny', 'small', 'medium', 'large', 'full']
offset = 5 # maximum distance between 2 consecutive fundamental frequency to be considered a potential line
count_thresh = 3 # number of adjacent potential filled-pause segments needed to be considered a filled pause
patience = 1 # maximum number of not-filled-pause segments can be skipped before the sequence is considered not filled pause
mean_thresh = 10 # maximum distance between 2 consecutive potential filled-pause segment to be considered in the same filled pause


sound_dict = {
    44100: {
        'incomplete_sentences': 'test_samples/incomplete_sentences/*.wav',
        'uhm': 'test_samples/uhm/44100/*.wav',
        'uhm_duong': 'uhm_duong.wav',
        'oh': 'test_samples/oh/oh_44100_mono_32bit.wav'
    },
    8000: {
        'uhm': 'test_samples/uhm/8000/*.wav',
        'oh': 'test_samples/oh/0938066840_20190820_143441_0:21_oh1.wav',
        'low_energy': 'test_samples/low_energy/*.wav',
        'incomplete_sentences': 'test_samples/incomplete_sentences/*.wav'
    }
}


def smooth_predictions(predicts, patience=1, target=1):
    if target == 0 and predicts.shape[0] <= 3:
        return predicts
    pos_idx = -1
    for i in range(0, predicts.shape[0]):
        predict = predicts[i]
        if predict == target:
            if pos_idx != -1 and i - pos_idx - 1 <= patience:
                predicts[pos_idx:i] = target
            pos_idx = i
    return predicts


def warm_up(sr, hop, model_capacity, rounds=20):
    for i in range(rounds):
        crepe.predict(np.empty(int(sr*hop)), sr, model_capacity=model_capacity, viterbi=True, verbose=0)


def uhm_confirm(preds, count_thresh, patience=1, mean_thresh=10):
    i = 0
    fixed_preds = []
    while i < len(preds):
        pred = preds[i][0]
        if pred == 1:
            prev_mean = preds[i][1]
            patience_temp = 0
            positive_pos = [i]
            temp_i = i + 1
            while temp_i < len(preds):
                pred_temp = preds[temp_i][0]
                mean_temp = preds[temp_i][1]
                # print('start: {}. current: {}. sub mean: {}. patience: {}'.format(i, temp_i, abs(mean_temp - prev_mean), patience_temp))
                if pred_temp == 1 and abs(mean_temp - prev_mean) <= mean_thresh:
                    positive_pos.append(temp_i)
                    prev_mean = mean_temp
                    patience_temp = 0
                else:
                    patience_temp += 1
                    if patience_temp > patience:
                        break
                temp_i += 1
            if len(positive_pos) >= count_thresh:
                final_pos = positive_pos[-1]
                print('Filled pause detected at %.1fs - %.1fs' % (0.1*i, 0.1*(final_pos + 1)))
                for _ in range(final_pos + 1 - i):
                    fixed_preds.append(1)
            else:
                final_pos = positive_pos[0]
                fixed_preds.append(0)
            i = final_pos
        else:
            fixed_preds.append(0)
        i += 1
    return np.array(fixed_preds)


def track_segments(preds, count_thresh=3, patience=1, mean_thresh=10):
    is_uhm = False
    patience_temp = 0
    prev_mean = -1
    count = 0
    for i in range(preds.shape[0]):
        prediction = preds[i, 0]
        mean = preds[i, 1]
        if prediction == 1:
            # patience_temp = 0
            if prev_mean == -1:
                prev_mean = mean
            if abs(mean - prev_mean) <= mean_thresh:
                count += 1
                if count >= count_thresh:
                    is_uhm = True
                prev_mean = mean
        else:
            patience_temp += 1
            if patience_temp > patience:
                count = 0
                prev_mean = -1
    return is_uhm


def analyze_segment(wav, sr):
    is_uhm = False
    time, frequency, confidence, activation = crepe.predict(wav, sr, model_capacity=model_capacity, viterbi=True,
                                                            verbose=0)
    min_length = 0.5 * frequency.shape[0]
    potential_lines = []
    i = 0
    on_a_line = False
    mean_uhm = 0
    line = [(time[i], frequency[i])]
    while i < frequency.shape[0] - 1:
        current_frame = frequency[i]
        next_frame = frequency[i + 1]
        if abs(next_frame - current_frame) <= offset:
            if not on_a_line:
                line.append((time[i + 1], next_frame))
        else:
            if len(line) >= min_length:
                potential_lines.append(line)
            line = [(time[i + 1], next_frame)]
        i += 1
    if len(line) >= min_length:
        potential_lines.append(line)
    for line in potential_lines:
        line = np.array(line)
        if np.std(line[:, 1]) <= 2. and 80 <= np.mean(line[:, 1]) <= 400.:
            is_uhm = True
            mean_uhm = np.mean(line[:, 1])
            break
    return is_uhm, mean_uhm


if __name__ == "__main__":
    warm_up(sample_rate, hop, model_capacity)
    filenames = []
    total_time = 0
    counter = 0
    for cat in sound_dict[sample_rate].keys():
        filenames.extend(glob.glob(sound_dict[sample_rate][cat]))
    for filename in filenames:
        print(filename)
        wav, sr = librosa.core.load(filename, sr=sample_rate)
        uhms = []
        for j in range(0, wav.size, int(sample_rate*hop)):
            counter += 1
            wav_temp = wav[j:j + int(sample_rate * hop)]
            t1 = t.time()
            # if is_uhm:
            is_uhm, mean_uhm = analyze_segment(wav_temp, sr)
            uhms.append([1 if is_uhm else 0, mean_uhm])

            # this method is for realtime tracking
            # is_uhm_realtime = track_segments(np.array(uhms)[-(count_thresh + patience):len(uhms)], count_thresh, patience, mean_thresh)
            total_time += t.time() - t1
        uhms = np.array(uhms)
        # print('label before: {}'.format(uhms[:, 0]))
        # print('mean before: {}'.format(uhms[:, 1]))
        new_uhms = uhm_confirm(uhms, count_thresh)
        # print('after: {}'.format(new_uhms))

    print('Average process time per segment: {}'.format(total_time*1./counter))