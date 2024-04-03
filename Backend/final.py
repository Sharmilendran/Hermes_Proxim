import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.signal import wiener
import pandas as pd
from statistics import NormalDist
from sklearn.decomposition import PCA
from librosa import load as load_audio
from librosa import stft, amplitude_to_db
from librosa.display import specshow
from scipy.ndimage import label as label_features
from scipy.ndimage import maximum_position as extract_region_maximums
import tensorflow as tf
import tensorflow_hub as hub
import csv
from IPython.display import Audio
from scipy.io import wavfile
import IPython
import noisereduce as nr
import soundfile as sf
from noisereduce.generate_noise import band_limited_noise
import urllib.request
import io
# import cv2
import wave
import os
import response
####### Methods to get CSV Files ################

# Load the model.

classLabels = ["Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue", "Babbling",
               "Speech synthesizer", "Shout", "Bellow", "Whoop", "Yell", "Children shouting", "Screaming", "Whispering",
               "Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Crying, sobbing",
               "Baby cry, infant cry", "Whimper", "Wail, moan", "Sigh", "Singing", "Choir", "Yodeling", "Chant",
               "Mantra", "Child singing", "Synthetic singing", "Rapping", "Humming", "Groan", "Grunt", "Whistling",
               "Breathing", "Wheeze", "Snoring", "Gasp", "Pant", "Snort", "Cough", "Throat clearing", "Sneeze", "Sniff",
               "Run", "Shuffle", "Walk, footsteps", "Chewing, mastication", "Biting", "Gargling", "Stomach rumble",
               "Burping, eructation", "Hiccup", "Fart", "Hands", "Finger snapping", "Clapping",
               "Heart sounds, heartbeat", "Heart murmur", "Cheering", "Applause", "Chatter", "Crowd",
               "Hubbub, speech noise, speech babble", "Children playing", "Animal", "Domestic animals, pets", "Dog",
               "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)", "Cat", "Purr", "Meow", "Hiss",
               "Caterwaul", "Livestock, farm animals, working animals", "Horse", "Clip-clop", "Neigh, whinny",
               "Cattle, bovinae", "Moo", "Cowbell", "Pig", "Oink", "Goat", "Bleat", "Sheep", "Fowl", "Chicken, rooster",
               "Cluck", "Crowing, cock-a-doodle-doo", "Turkey", "Gobble", "Duck", "Quack", "Goose", "Honk",
               "Wild animals", "Roaring cats (lions, tigers)", "Roar", "Bird",
               "Bird vocalization, bird call, bird song", "Chirp, tweet", "Squawk", "Pigeon, dove", "Coo", "Crow",
               "Caw", "Owl", "Hoot", "Bird flight, flapping wings", "Canidae, dogs, wolves", "Rodents, rats, mice",
               "Mouse", "Patter", "Insect", "Cricket", "Mosquito", "Fly, housefly", "Buzz", "Bee, wasp, etc.", "Frog",
               "Croak", "Snake", "Rattle", "Whale vocalization", "Music", "Musical instrument",
               "Plucked string instrument", "Guitar", "Electric guitar", "Bass guitar", "Acoustic guitar",
               "Steel guitar, slide guitar", "Tapping (guitar technique)", "Strum", "Banjo", "Sitar", "Mandolin",
               "Zither", "Ukulele", "Keyboard (musical)", "Piano", "Electric piano", "Organ", "Electronic organ",
               "Hammond organ", "Synthesizer", "Sampler", "Harpsichord", "Percussion", "Drum kit", "Drum machine",
               "Drum", "Snare drum", "Rimshot", "Drum roll", "Bass drum", "Timpani", "Tabla", "Cymbal", "Hi-hat",
               "Wood block", "Tambourine", "Rattle (instrument)", "Maraca", "Gong", "Tubular bells",
               "Mallet percussion", "Marimba, xylophone", "Glockenspiel", "Vibraphone", "Steelpan", "Orchestra",
               "Brass instrument", "French horn", "Trumpet", "Trombone", "Bowed string instrument", "String section",
               "Violin, fiddle", "Pizzicato", "Cello", "Double bass", "Wind instrument, woodwind instrument", "Flute",
               "Saxophone", "Clarinet", "Harp", "Bell", "Church bell", "Jingle bell", "Bicycle bell", "Tuning fork",
               "Chime", "Wind chime", "Change ringing (campanology)", "Harmonica", "Accordion", "Bagpipes",
               "Didgeridoo", "Shofar", "Theremin", "Singing bowl", "Scratching (performance technique)", "Pop music",
               "Hip hop music", "Beatboxing", "Rock music", "Heavy metal", "Punk rock", "Grunge", "Progressive rock",
               "Rock and roll", "Psychedelic rock", "Rhythm and blues", "Soul music", "Reggae", "Country",
               "Swing music", "Bluegrass", "Funk", "Folk music", "Middle Eastern music", "Jazz", "Disco",
               "Classical music", "Opera", "Electronic music", "House music", "Techno", "Dubstep", "Drum and bass",
               "Electronica", "Electronic dance music", "Ambient music", "Trance music", "Music of Latin America",
               "Salsa music", "Flamenco", "Blues", "Music for children", "New-age music", "Vocal music", "A capella",
               "Music of Africa", "Afrobeat", "Christian music", "Gospel music", "Music of Asia", "Carnatic music",
               "Music of Bollywood", "Ska", "Traditional music", "Independent music", "Song", "Background music",
               "Theme music", "Jingle (music)", "Soundtrack music", "Lullaby", "Video game music", "Christmas music",
               "Dance music", "Wedding music", "Happy music", "Sad music", "Tender music", "Exciting music",
               "Angry music", "Scary music", "Wind", "Rustling leaves", "Wind noise (microphone)", "Thunderstorm",
               "Thunder", "Water", "Rain", "Raindrop", "Rain on surface", "Stream", "Waterfall", "Ocean", "Waves, surf",
               "Steam", "Gurgling", "Fire", "Crackle", "Vehicle", "Boat, Water vehicle", "Sailboat, sailing ship",
               "Rowboat, canoe, kayak", "Motorboat, speedboat", "Ship", "Motor vehicle (road)", "Car",
               "Vehicle horn, car horn, honking", "Toot", "Car alarm", "Power windows, electric windows", "Skidding",
               "Tire squeal", "Car passing by", "Race car, auto racing", "Truck", "Air brake", "Air horn, truck horn",
               "Reversing beeps", "Ice cream truck, ice cream van", "Bus", "Emergency vehicle", "Police car (siren)",
               "Ambulance (siren)", "Fire engine, fire truck (siren)", "Motorcycle", "Traffic noise, roadway noise",
               "Rail transport", "Train", "Train whistle", "Train horn", "Railroad car, train wagon",
               "Train wheels squealing", "Subway, metro, underground", "Aircraft", "Aircraft engine", "Jet engine",
               "Propeller, airscrew", "Helicopter", "Fixed-wing aircraft, airplane", "Bicycle", "Skateboard", "Engine",
               "Light engine (high frequency)", "Dental drill, dentist's drill", "Lawn mower", "Chainsaw",
               "Medium engine (mid frequency)", "Heavy engine (low frequency)", "Engine knocking", "Engine starting",
               "Idling", "Accelerating, revving, vroom", "Door", "Doorbell", "Ding-dong", "Sliding door", "Slam",
               "Knock", "Tap", "Squeak", "Cupboard open or close", "Drawer open or close", "Dishes, pots, and pans",
               "Cutlery, silverware", "Chopping (food)", "Frying (food)", "Microwave oven", "Blender",
               "Water tap, faucet", "Sink (filling or washing)", "Bathtub (filling or washing)", "Hair dryer",
               "Toilet flush", "Toothbrush", "Electric toothbrush", "Vacuum cleaner", "Zipper (clothing)",
               "Keys jangling", "Coin (dropping)", "Scissors", "Electric shaver, electric razor", "Shuffling cards",
               "Typing", "Typewriter", "Computer keyboard", "Writing", "Alarm", "Telephone", "Telephone bell ringing",
               "Ringtone", "Telephone dialing, DTMF", "Dial tone", "Busy signal", "Alarm clock", "Siren",
               "Civil defense siren", "Buzzer", "Smoke detector, smoke alarm", "Fire alarm", "Foghorn", "Whistle",
               "Steam whistle", "Mechanisms", "Ratchet, pawl", "Clock", "Tick", "Tick-tock", "Gears", "Pulleys",
               "Sewing machine", "Mechanical fan", "Air conditioning", "Cash register", "Printer", "Camera",
               "Single-lens reflex camera", "Tools", "Hammer", "Jackhammer", "Sawing", "Filing (rasp)", "Sanding",
               "Power tool", "Drill", "Explosion", "Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire",
               "Cap gun", "Fireworks", "Firecracker", "Burst, pop", "Eruption", "Boom", "Wood", "Chop", "Splinter",
               "Crack", "Glass", "Chink, clink", "Shatter", "Liquid", "Splash, splatter", "Slosh", "Squish", "Drip",
               "Pour", "Trickle, dribble", "Gush", "Fill (with liquid)", "Spray", "Pump (liquid)", "Stir", "Boiling",
               "Sonar", "Arrow", "Whoosh, swoosh, swish", "Thump, thud", "Thunk", "Electronic tuner", "Effects unit",
               "Chorus effect", "Basketball bounce", "Bang", "Slap, smack", "Whack, thwack", "Smash, crash", "Breaking",
               "Bouncing", "Whip", "Flap", "Scratch", "Scrape", "Rub", "Roll", "Crushing", "Crumpling, crinkling",
               "Tearing", "Beep, bleep", "Ping", "Ding", "Clang", "Squeal", "Creak", "Rustle", "Whir", "Clatter",
               "Sizzle", "Clicking", "Clickety-clack", "Rumble", "Plop", "Jingle, tinkle", "Hum", "Zing", "Boing",
               "Crunch", "Silence", "Sine wave", "Harmonic", "Chirp tone", "Sound effect", "Pulse",
               "Inside, small room", "Inside, large room or hall", "Inside, public space", "Outside, urban or manmade",
               "Outside, rural or natural", "Reverberation", "Echo", "Noise", "Environmental noise", "Static",
               "Mains hum", "Distortion", "Sidetone", "Cacophony", "White noise", "Pink noise", "Throbbing",
               "Vibration", "Television", "Radio", "Field recording"]
peakList = []
peakList2 = []
araayFinal = []
toConverFrames = []
results = []
toConverFrames = []
toConverFrames2 = []
FinalSetter = []


def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])

    return class_names


model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


######### Methods to Get Repettions ###################


def get_highest_with_index(df, n):
    # Get the n largest items from the DataFrame
    highest_values = df.nlargest(n, 'Count', "all")

    # Create a 2D array to store the number and its index
    result = []

    # Iterate over the highest values to find their indices and store them in the result array
    for index, row in highest_values.iterrows():
        result.append([row['Count'], index])

    return result


def most_repetitive_numbers2(arr):
    # Dictionary to store counts of each number
    counts = {}

    # Count occurrences of each number
    for num in arr:
        if num in counts:
            counts[num] += 1
        else:
            counts[num] = 1

    # Find the maximum count
    max_count = max(counts.values())

    # Find the numbers with the maximum count
    most_repetitive2 = [num for num, count in counts.items() if count == max_count]

    return most_repetitive2, counts


################## Noise Reduction Handler ##############

def soundSorterMatcher():
    audio_file = 'recorded_audio.wav'
    sample_rate, wav_data = wavfile.read(audio_file, 'rb')
    data = wav_data
    print(sample_rate)
    noise_len = 2
    noise = band_limited_noise(min_freq=2000, max_freq=12000, samples=len(wav_data), samplerate=sample_rate) * 10
    noise_clip = noise[:sample_rate * noise_len]
    audio_clip_band_limited = data + noise

    reduced_noise = nr.reduce_noise(y=audio_clip_band_limited, sr=sample_rate, n_std_thresh_stationary=1.5,
                                    stationary=True)

    y1, sr = reduced_noise, sample_rate

    y1.shape
    print(y1.shape)

    ################# Delete the audio File for unneccessary repetitions. ######################
    file_path = 'test3.wav'
    output_filename = 'test3'

    try:
        os.remove(file_path)
        print("File", output_filename, "deleted successfully.")
    except FileNotFoundError:
        print("File", output_filename, "not found.")
    except Exception as e:
        print("An error occurred:", e)

    filename = "test3.wav"

    # Open the file in write mode for WAV format
    with wave.open(filename, 'wb') as wav_file:
        # Set the number of channels (mono = 1, stereo = 2)
        wav_file.setnchannels(1)

        # Set the sample rate (same as the 'rate' variable you used)
        wav_file.setsampwidth(2)  # 2 bytes per sample for 16-bit audio (common for WAV)
        wav_file.setframerate(sample_rate)

        # Convert the data to bytes (assuming it's in float format between -1 and 1)
        # You might need to modify this conversion based on your data type
        scaled_data = np.int16(reduced_noise * 32767)  # Scale to -32768 to 32767 for 16-bit

        # Write the data to the WAV file
        wav_file.writeframes(scaled_data.tobytes())

    # Generate the spectrogram
    D = librosa.amplitude_to_db(librosa.stft(y1), ref=np.max)
    spectrogram = D
    print(f'SPectrogram Shape is {spectrogram.shape[1]}')

    # Sound Activity detection Algorithm #
    # number of standard deviations away from the mean
    threshold = 2.75

    # remove zero values
    flattened = np.matrix.flatten(spectrogram)
    filtered = flattened[flattened > np.min(flattened)]

    # create a normal distribution from frequency intensities
    # then map a zscore onto each intensity value
    ndist = NormalDist(np.mean(filtered), np.std(filtered))
    zscore = np.vectorize(lambda x: ndist.zscore(x))
    zscore_matrix = zscore(spectrogram)

    # create label matrix from frequency intensities that are
    # above threshold
    mask_matrix = zscore_matrix > threshold
    labelled_matrix, num_regions = label_features(mask_matrix)
    print(num_regions)
    label_indices = np.arange(num_regions) + 1

    # for each isolated region in the mask, identify the maximum
    # value, then extract it position
    peak_positions = extract_region_maximums(
        zscore_matrix, labelled_matrix, label_indices)

    # finally, create list of peaks (time, frequency, intensity)
    peaks = [(x, y, spectrogram[y, x]) for y, x in peak_positions]

    for i in range(len(peaks)):
        print(f'{peaks[i][0]} and {peaks[i][2]}')
        peakList2.append([peaks[i][0], peaks[i][2]])
        peakList.append(peaks[i][0])
        print(peaks[i])

    print(peakList)

    most_repetitive, counts = most_repetitive_numbers2(peakList)

    # Create a DataFrame from the counts dictionary
    df = pd.DataFrame(counts.items(), columns=['Number', 'Count'])

    print("Most repetitive numbers:", most_repetitive)
    print("Counts for all numbers:", counts)
    print(df)

    counter = 0
    len(df)
    for i in range(len(df)):
        print(f'{df.iloc[i][0]} and {df.iloc[i][1]}')
        counter = counter + 1

    for i in range(len(most_repetitive)):
        for j in range(len(df)):
            if most_repetitive[i] == df.iloc[j][0]:
                araayFinal.append([df.iloc[j][0], df.iloc[j][1]])

    print(araayFinal)

    sumT = 0
    for i in range(len(araayFinal)):
        sumT = araayFinal[i][1] + sumT
        print(sumT)

    ArraytoBePassed = []

    proximLabel = ''
    print(sumT)
    for i in range(len(araayFinal)):
        y = araayFinal[i][1] / sumT
        if y < 0.4:
            proximLabel = "Far"
        elif 0.6 > y > 0.4:
            proximLabel = "nearby"
        elif 0.8 > y > 0.6:
            proximLabel = "Close"
        elif y > 0.8:
            proximLabel = "Very Close"

        ArraytoBePassed.append([araayFinal[i][0], araayFinal[i][1], proximLabel])
        print(y)

    sumT = 0
    for i in range(len(araayFinal)):
        sumT = araayFinal[i][1] + sumT
        print(sumT)

    # #Checking Code
    # for i in range(len(ArraytoBePassed)):
    #     print(f'Frame {ArraytoBePassed[i][0]} Count {ArraytoBePassed[i][1]} proximity {ArraytoBePassed[i][2]}')

    finalDF = pd.DataFrame(ArraytoBePassed, columns=["Frame", "Count", "Proximity"])
    finalDF3 = finalDF.nlargest(10, 'Count', 'all')

    ## Classifying Closest Sounds ##:
    model = hub.load('https://tfhub.dev/google/yamnet/1')

    # wav_file_name = 'speech_whistling2.wav'
    # Swav_file_name = 'miaow_16k.wav'
    wav_file_name = 'recorded_audio.wav'
    sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
    # sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    print(sample_rate)

    # Show some basic information about the audio.
    duration = len(wav_data) / sample_rate
    print(f'Sample rate: {sample_rate} Hz')
    print(f'Total duration: {duration:.2f}s')
    print(f'Size of the input: {len(wav_data)}')

    # Listening to the wav file.
    Audio(wav_data, rate=sample_rate)

    # YAMNET Parameters:
    n_fft = 0.96
    hop_length = 0.48
    sr = sample_rate  # 16000Hz

    waveform = wav_data / tf.int16.max

    scores, embeddings, spectrogram2 = model(waveform)
    scores_np = scores.numpy()
    spectrogram_np = spectrogram2.numpy()
    scores_np.shape
    spectrogram_np.shape
    print(f'{spectrogram2.shape} and the other {spectrogram.shape} and scoresNp{scores_np.shape}')

    # Spectrogram frames:
    frameO = spectrogram.shape[1]

    # Original   Frames To be found

    for i in range(len(finalDF)):
        print(spectrogram.shape[1])
        x = finalDF3.iloc[i][0] / frameO
        print(x)  # nlargest method sorted DataFrame to sort the values.
        y = round(x * scores_np.shape[0])
        print(y)
        toConverFrames.append(y)
        toConverFrames2.append([y, finalDF3.iloc[i][2]])
        print(f'The Values for frames were {x} before and now {y} Proxim {finalDF3.iloc[i][2]}')

    # Sound Value Extraction:

    # The needed Methods:

    for i in range(len(toConverFrames2) - 1):
        dflabel = pd.DataFrame(scores_np[toConverFrames2[i][0]], columns=["Count"])
        dflabel.columns = dflabel.columns.str.strip()
        infered_class1 = class_names[scores_np[toConverFrames2[i][0]].max(axis=0).argmax()]
        highest_with_index = get_highest_with_index(dflabel, 3)
        for j in range(3):
            print(f"-------------------------------------------- {counter}")

            print(f'Label :  {classLabels[highest_with_index[j][1]]} and Proximity {toConverFrames2[i][1]}')
            FinalSetter.append([classLabels[highest_with_index[j][1]], toConverFrames2[i][1]])

            print(f"======================================================+{counter} ")
        # print(f'The main sound is: {infered_class1}')

    dfFinalFrame = pd.DataFrame(FinalSetter, columns=["Class", "Proximity"])
    finalDictionary = dfFinalFrame.to_dict()

    resp = []
    for classification in finalDictionary["Class"]:
        for proximity in finalDictionary["Proximity"]:
            proxRes = response.ProximityResponse(classification, proximity)
            resp.append(proxRes.__dict__)
    print("final dictionary ", resp)
    return resp