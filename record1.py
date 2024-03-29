import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
#import tensorflow_hub as hub
import pyaudio
from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd
import pandas as pd
import wave
from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input
import csv

from plot import Plotter
from scipy.io import wavfile
from IPython.display import Audio
import heapq
import os


def get_highest_with_index(df, n):
    # Get the n largest items from the DataFrame
    highest_values = df.nlargest(n, 'Count',"all")
    
    # Create a 2D array to store the number and its index
    result = []
    
    # Iterate over the highest values to find their indices and store them in the result array
    for index, row in highest_values.iterrows():
        result.append([row['Count'], index])
    
    return result


def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
         class_names.append(row['display_name'])

    return class_names



def get_highest_with_index1(arr, n):

    # Use heapq's nlargest to get the n highest elements
    highest_values = heapq.nlargest(n, arr)
    
    # Initialize a 2D array to store both the number and its index
    result = []
    labelResult = []
    
    # Iterate over the highest values to find their indices and store them in the result array
    for value in highest_values:
        index = arr.index(value)
        val = value
        #index = np.where(val == value)
        result.append([value, index])
        labelResult.append(index)
    
    return labelResult

classLabels = ["Speech","Child speech, kid speaking","Conversation","Narration, monologue","Babbling","Speech synthesizer","Shout","Bellow","Whoop","Yell","Children shouting","Screaming","Whispering","Laughter","Baby laughter","Giggle","Snicker","Belly laugh","Chuckle, chortle","Crying, sobbing","Baby cry, infant cry","Whimper","Wail, moan","Sigh","Singing","Choir","Yodeling","Chant","Mantra","Child singing","Synthetic singing","Rapping","Humming","Groan","Grunt","Whistling","Breathing","Wheeze","Snoring","Gasp","Pant","Snort","Cough","Throat clearing","Sneeze","Sniff","Run","Shuffle","Walk, footsteps","Chewing, mastication","Biting","Gargling","Stomach rumble","Burping, eructation","Hiccup","Fart","Hands","Finger snapping","Clapping","Heart sounds, heartbeat","Heart murmur","Cheering","Applause","Chatter","Crowd","Hubbub, speech noise, speech babble","Children playing","Animal","Domestic animals, pets","Dog","Bark","Yip","Howl","Bow-wow","Growling","Whimper (dog)","Cat","Purr","Meow","Hiss","Caterwaul","Livestock, farm animals, working animals","Horse","Clip-clop","Neigh, whinny","Cattle, bovinae","Moo","Cowbell","Pig","Oink","Goat","Bleat","Sheep","Fowl","Chicken, rooster","Cluck","Crowing, cock-a-doodle-doo","Turkey","Gobble","Duck","Quack","Goose","Honk","Wild animals","Roaring cats (lions, tigers)","Roar","Bird","Bird vocalization, bird call, bird song", "Chirp, tweet","Squawk","Pigeon, dove","Coo","Crow","Caw","Owl","Hoot","Bird flight, flapping wings","Canidae, dogs, wolves","Rodents, rats, mice","Mouse","Patter","Insect","Cricket","Mosquito","Fly, housefly","Buzz","Bee, wasp, etc.","Frog","Croak","Snake","Rattle","Whale vocalization","Music","Musical instrument","Plucked string instrument","Guitar","Electric guitar","Bass guitar","Acoustic guitar","Steel guitar, slide guitar","Tapping (guitar technique)","Strum","Banjo","Sitar","Mandolin","Zither","Ukulele","Keyboard (musical)","Piano","Electric piano","Organ","Electronic organ","Hammond organ","Synthesizer","Sampler","Harpsichord","Percussion","Drum kit","Drum machine","Drum","Snare drum","Rimshot","Drum roll","Bass drum","Timpani","Tabla","Cymbal","Hi-hat","Wood block","Tambourine","Rattle (instrument)","Maraca","Gong","Tubular bells","Mallet percussion","Marimba, xylophone","Glockenspiel","Vibraphone","Steelpan","Orchestra","Brass instrument","French horn","Trumpet","Trombone","Bowed string instrument","String section","Violin, fiddle","Pizzicato","Cello","Double bass","Wind instrument, woodwind instrument","Flute","Saxophone","Clarinet","Harp","Bell","Church bell","Jingle bell","Bicycle bell","Tuning fork","Chime","Wind chime","Change ringing (campanology)","Harmonica","Accordion","Bagpipes","Didgeridoo","Shofar","Theremin","Singing bowl","Scratching (performance technique)","Pop music","Hip hop music","Beatboxing","Rock music","Heavy metal","Punk rock","Grunge","Progressive rock","Rock and roll","Psychedelic rock","Rhythm and blues","Soul music","Reggae","Country","Swing music","Bluegrass","Funk","Folk music","Middle Eastern music","Jazz","Disco","Classical music","Opera","Electronic music","House music","Techno","Dubstep","Drum and bass","Electronica","Electronic dance music","Ambient music","Trance music","Music of Latin America","Salsa music","Flamenco","Blues","Music for children","New-age music","Vocal music","A capella","Music of Africa","Afrobeat","Christian music","Gospel music","Music of Asia","Carnatic music","Music of Bollywood","Ska","Traditional music","Independent music","Song","Background music","Theme music","Jingle (music)","Soundtrack music","Lullaby","Video game music","Christmas music","Dance music","Wedding music","Happy music","Sad music","Tender music","Exciting music","Angry music","Scary music","Wind","Rustling leaves","Wind noise (microphone)","Thunderstorm","Thunder","Water","Rain","Raindrop","Rain on surface","Stream","Waterfall","Ocean","Waves, surf","Steam","Gurgling","Fire","Crackle","Vehicle","Boat, Water vehicle","Sailboat, sailing ship","Rowboat, canoe, kayak","Motorboat, speedboat","Ship","Motor vehicle (road)","Car","Vehicle horn, car horn, honking","Toot","Car alarm","Power windows, electric windows","Skidding","Tire squeal","Car passing by","Race car, auto racing","Truck","Air brake","Air horn, truck horn","Reversing beeps","Ice cream truck, ice cream van","Bus","Emergency vehicle","Police car (siren)","Ambulance (siren)","Fire engine, fire truck (siren)","Motorcycle","Traffic noise, roadway noise","Rail transport","Train","Train whistle","Train horn","Railroad car, train wagon","Train wheels squealing","Subway, metro, underground","Aircraft","Aircraft engine","Jet engine","Propeller, airscrew","Helicopter","Fixed-wing aircraft, airplane","Bicycle","Skateboard","Engine","Light engine (high frequency)","Dental drill, dentist's drill","Lawn mower","Chainsaw","Medium engine (mid frequency)","Heavy engine (low frequency)","Engine knocking","Engine starting","Idling","Accelerating, revving, vroom","Door","Doorbell","Ding-dong","Sliding door","Slam","Knock","Tap","Squeak","Cupboard open or close","Drawer open or close","Dishes, pots, and pans","Cutlery, silverware","Chopping (food)","Frying (food)","Microwave oven","Blender","Water tap, faucet","Sink (filling or washing)","Bathtub (filling or washing)","Hair dryer","Toilet flush","Toothbrush","Electric toothbrush","Vacuum cleaner","Zipper (clothing)","Keys jangling","Coin (dropping)","Scissors","Electric shaver, electric razor","Shuffling cards","Typing","Typewriter","Computer keyboard","Writing","Alarm","Telephone","Telephone bell ringing","Ringtone","Telephone dialing, DTMF","Dial tone","Busy signal","Alarm clock","Siren","Civil defense siren","Buzzer","Smoke detector, smoke alarm","Fire alarm","Foghorn","Whistle","Steam whistle","Mechanisms","Ratchet, pawl","Clock","Tick","Tick-tock","Gears","Pulleys","Sewing machine","Mechanical fan","Air conditioning","Cash register","Printer","Camera","Single-lens reflex camera","Tools","Hammer","Jackhammer","Sawing","Filing (rasp)","Sanding","Power tool","Drill","Explosion","Gunshot, gunfire","Machine gun","Fusillade","Artillery fire","Cap gun","Fireworks","Firecracker","Burst, pop","Eruption","Boom","Wood","Chop","Splinter","Crack","Glass","Chink, clink","Shatter","Liquid","Splash, splatter","Slosh","Squish","Drip","Pour","Trickle, dribble","Gush","Fill (with liquid)","Spray","Pump (liquid)","Stir","Boiling","Sonar","Arrow","Whoosh, swoosh, swish","Thump, thud","Thunk","Electronic tuner","Effects unit","Chorus effect","Basketball bounce","Bang","Slap, smack","Whack, thwack","Smash, crash","Breaking","Bouncing","Whip","Flap","Scratch","Scrape","Rub","Roll","Crushing","Crumpling, crinkling","Tearing","Beep, bleep","Ping","Ding","Clang","Squeal","Creak","Rustle","Whir","Clatter","Sizzle","Clicking","Clickety-clack","Rumble","Plop","Jingle, tinkle","Hum","Zing","Boing","Crunch","Silence","Sine wave","Harmonic","Chirp tone","Sound effect","Pulse","Inside, small room","Inside, large room or hall","Inside, public space","Outside, urban or manmade","Outside, rural or natural","Reverberation","Echo","Noise","Environmental noise","Static","Mains hum","Distortion","Sidetone","Cacophony","White noise","Pink noise","Throbbing","Vibration","Television","Radio","Field recording"]

def proxim():

    model2 = hub.load('https://tfhub.dev/google/yamnet/1')


    class_map_path = model2.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)

    # wav_file_name = 'speech_whistling2.wav'
    wav_file_name = 'recorded_audio.wav'
    sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
    #sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    # Show some basic information about the audio.
    duration = len(wav_data)/sample_rate
    print(f'Sample rate: {sample_rate} Hz')
    print(f'Total duration: {duration:.2f}s')
    print(f'Size of the input: {len(wav_data)}')

    # Listening to the wav file.
    Audio(wav_data, rate=sample_rate)

    #Preparing the wavfile for intensity data extraction.
    waveform = wav_data / tf.int16.max

    scores, embeddings, spectrogram = model2(waveform)
    #dflabel2 = pd.DataFrame(scores,columns=["Count"])
    #dflabel2.columns = dflabel.columns.str.strip()
    print("##################################################")
    #print( dflabel2.head)
    #print(dflabel2.columns)
    print(scores.shape)
    #print(scores[0])

    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    class_names[scores_np.mean(axis=0).argmax()]
    infered_class = class_names[scores_np.mean(axis=0).argmax()]
    print(f'The main sound is: {infered_class}')


    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    ####### Proximity Calculation #########

    df = pd.DataFrame(spectrogram)
    print(df.shape[0])


    sorted = []
    for i in range(df.shape[0]):
        a = df.iloc[i].max()
        sorted.append(a)
        print(f"{i}th frame: max mel value {a}")
    #print(sorted)
        
    # Reference Array
    unsorted = sorted.copy()

    plt.figure(figsize=(10, 6))

    # Plot the waveform.
    plt.subplot(3, 1, 1)
    plt.plot(waveform)
    plt.xlim([0, len(waveform)])

    sorted.sort(reverse=True)

    no_frames = 5
    frame = []
    for i in range(no_frames):
        for j in range(len(unsorted)):
            if sorted[i] == unsorted[j]:
                frame.append(j)

    print(frame)
    frame2 = []

    for i in range(len(frame)):
        x = (frame[i]/df.shape[0])
        #x = (frame[i]/df.shape[0])*100
        y = round(x*scores_np.shape[0])
        print(f"first its{x} now its {y}")
        frame2.append(y)
    counter = 0 
    for i in range(len(frame2)):
        print(frame2[i])
        dflabel = pd.DataFrame(scores_np[frame2[i]],columns=["Count"])
        dflabel.columns = dflabel.columns.str.strip()
        print(f"dflabel Head now {dflabel.shape}")
        infered_class1 = class_names[scores_np[frame2[i]].mean(axis=0).argmax()]
        print(len(scores_np[frame2[i]]))
        highest_with_index = get_highest_with_index(dflabel, 3)
        topOG = []
        for i in range(3):
                print(classLabels[highest_with_index[i][1]])
                counter = counter +1
                print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@+ {counter}")
        #print(f'The main sound is: {infered_class1}')






def getArray():

    
    ################### SETTINGS ###################
    plt_classes = [0,132,420,494,62] # Speech, Music, Explosion, Silence 
    class_labels=True
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = params.SAMPLE_RATE
    WIN_SIZE_SEC = 0.975
    CHUNK = int(WIN_SIZE_SEC * RATE)
    RECORD_SECONDS = 20

    print(sd.query_devices())
    MIC = None
    frames = []

    #################### MODEL #####################
    
    model = YAMNet(weights='keras_yamnet\yamnet.h5')
    yamnet_classes = class_names('keras_yamnet\yamnet_class_map.csv')

    #################### STREAM ####################
    audio = pyaudio.PyAudio()
    p = pyaudio.PyAudio()
    ##################### File Deletion ########################

    file_path = 'recorded_audio.wav'
    output_filename = 'recorded_audio'
    
    try:
        os.remove(file_path)
        print("File", output_filename, "deleted successfully.")
    except FileNotFoundError:
        print("File", output_filename, "not found.")
    except Exception as e:
        print("An error occurred:", e)


    # start Recording

    stream = audio.open(format=FORMAT,
                        input_device_index=MIC,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    print("recording...")
    
    output_filename = "recorded_audio.wav"
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(1)
    #wf.writeframes(b''.join(frames))
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    
    
    if plt_classes is not None:
        plt_classes_lab = yamnet_classes[plt_classes]
        n_classes = len(plt_classes)
    else:
        plt_classes = [k for k in range(len(yamnet_classes))]
        plt_classes_lab = yamnet_classes if class_labels else None
        n_classes = len(yamnet_classes)

    #monitor = Plotter(n_classes=n_classes, FIG_SIZE=(12,6), msd_labels=plt_classes_lab)
  


    recordings_list = []
    label = ''
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        dataR = stream.read(CHUNK)
        frames.append(dataR)
        data = preprocess_input(np.frombuffer(
            stream.read(CHUNK), dtype=np.float32), RATE)
        
        
        prediction = model.predict(np.expand_dims(data,0))[0]
        dflabel2 = pd.DataFrame(prediction,columns=["Count"])
        dflabel2.columns = dflabel2.columns.str.strip()
        #Distance snippet:
        a = dflabel2.iloc[i].sum()
        dflabel2
        print(a)
        distance = ''

        if a > 0.005:
            distance = 'Far'
            if 0.5 > a > 0.005:
                distance = 'Close'
                if a > 1:
                    distance = 'Very Close'

        highest_with_index2 = get_highest_with_index(dflabel2, 3)
        for j in range(2):
            label = classLabels[highest_with_index2[j][1]]
            recordings_list.append([i,label])
            res = response.PredictResponse(label,distance)
            await websocket.send(res)
            print(f'The sound {label} is {distance}')
        recordings_list.append([i,label])
        

    print("finished recording")
    soundDf = pd.DataFrame(recordings_list, columns=['Time', 'SoundEvent'])
    print(f"The 10s recorded size is {soundDf.shape}")
    print(soundDf.head)
    
    

    # num = 0
    # for c in prediction:
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(classLabels[2])
    predictedClass = max(prediction)
    print(predictedClass)
    dflabel = pd.DataFrame(prediction,columns=["Count"])
    dflabel.columns = dflabel.columns.str.strip()
    print( dflabel.head)
    print( dflabel.shape)
    print(dflabel.columns)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    output_filename = "recorded_audio.wav"
    #wf = wave.open(output_filename, 'wb')
    # wf.setnchannels(1)
    wf.writeframes(b''.join(frames))
    
    wf.close()
    print("Recording finished. Audio saved as:", output_filename)
    
    highest_with_index = get_highest_with_index(dflabel, 10)
    names = []
    #print(highest_with_index)
    # print(classLabels[highest_with_index[0][1]])
    for i in range(10):
            print(classLabels[highest_with_index[i][1]])
            names.append(classLabels[highest_with_index[i][1]])

    print(output_filename.type)
    return names, output_filename


if __name__ == "__main__":

    getArray()