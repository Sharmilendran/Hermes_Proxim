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

if __name__ == "__main__":

    classLabels = ["Speech","Child speech, kid speaking","Conversation","Narration, monologue","Babbling","Speech synthesizer","Shout","Bellow","Whoop","Yell","Children shouting","Screaming","Whispering","Laughter","Baby laughter","Giggle","Snicker","Belly laugh","Chuckle, chortle","Crying, sobbing","Baby cry, infant cry","Whimper","Wail, moan","Sigh","Singing","Choir","Yodeling","Chant","Mantra","Child singing","Synthetic singing","Rapping","Humming","Groan","Grunt","Whistling","Breathing","Wheeze","Snoring","Gasp","Pant","Snort","Cough","Throat clearing","Sneeze","Sniff","Run","Shuffle","Walk, footsteps","Chewing, mastication","Biting","Gargling","Stomach rumble","Burping, eructation","Hiccup","Fart","Hands","Finger snapping","Clapping","Heart sounds, heartbeat","Heart murmur","Cheering","Applause","Chatter","Crowd","Hubbub, speech noise, speech babble","Children playing","Animal","Domestic animals, pets","Dog","Bark","Yip","Howl","Bow-wow","Growling","Whimper (dog)","Cat","Purr","Meow","Hiss","Caterwaul","Livestock, farm animals, working animals","Horse","Clip-clop","Neigh, whinny","Cattle, bovinae","Moo","Cowbell","Pig","Oink","Goat","Bleat","Sheep","Fowl","Chicken, rooster","Cluck","Crowing, cock-a-doodle-doo","Turkey","Gobble","Duck","Quack","Goose","Honk","Wild animals","Roaring cats (lions, tigers)","Roar","Bird","Bird vocalization, bird call, bird song", "Chirp, tweet","Squawk","Pigeon, dove","Coo","Crow","Caw","Owl","Hoot","Bird flight, flapping wings","Canidae, dogs, wolves","Rodents, rats, mice","Mouse","Patter","Insect","Cricket","Mosquito","Fly, housefly","Buzz","Bee, wasp, etc.","Frog","Croak","Snake","Rattle","Whale vocalization","Music","Musical instrument","Plucked string instrument","Guitar","Electric guitar","Bass guitar","Acoustic guitar","Steel guitar, slide guitar","Tapping (guitar technique)","Strum","Banjo","Sitar","Mandolin","Zither","Ukulele","Keyboard (musical)","Piano","Electric piano","Organ","Electronic organ","Hammond organ","Synthesizer","Sampler","Harpsichord","Percussion","Drum kit","Drum machine","Drum","Snare drum","Rimshot","Drum roll","Bass drum","Timpani","Tabla","Cymbal","Hi-hat","Wood block","Tambourine","Rattle (instrument)","Maraca","Gong","Tubular bells","Mallet percussion","Marimba, xylophone","Glockenspiel","Vibraphone","Steelpan","Orchestra","Brass instrument","French horn","Trumpet","Trombone","Bowed string instrument","String section","Violin, fiddle","Pizzicato","Cello","Double bass","Wind instrument, woodwind instrument","Flute","Saxophone","Clarinet","Harp","Bell","Church bell","Jingle bell","Bicycle bell","Tuning fork","Chime","Wind chime","Change ringing (campanology)","Harmonica","Accordion","Bagpipes","Didgeridoo","Shofar","Theremin","Singing bowl","Scratching (performance technique)","Pop music","Hip hop music","Beatboxing","Rock music","Heavy metal","Punk rock","Grunge","Progressive rock","Rock and roll","Psychedelic rock","Rhythm and blues","Soul music","Reggae","Country","Swing music","Bluegrass","Funk","Folk music","Middle Eastern music","Jazz","Disco","Classical music","Opera","Electronic music","House music","Techno","Dubstep","Drum and bass","Electronica","Electronic dance music","Ambient music","Trance music","Music of Latin America","Salsa music","Flamenco","Blues","Music for children","New-age music","Vocal music","A capella","Music of Africa","Afrobeat","Christian music","Gospel music","Music of Asia","Carnatic music","Music of Bollywood","Ska","Traditional music","Independent music","Song","Background music","Theme music","Jingle (music)","Soundtrack music","Lullaby","Video game music","Christmas music","Dance music","Wedding music","Happy music","Sad music","Tender music","Exciting music","Angry music","Scary music","Wind","Rustling leaves","Wind noise (microphone)","Thunderstorm","Thunder","Water","Rain","Raindrop","Rain on surface","Stream","Waterfall","Ocean","Waves, surf","Steam","Gurgling","Fire","Crackle","Vehicle","Boat, Water vehicle","Sailboat, sailing ship","Rowboat, canoe, kayak","Motorboat, speedboat","Ship","Motor vehicle (road)","Car","Vehicle horn, car horn, honking","Toot","Car alarm","Power windows, electric windows","Skidding","Tire squeal","Car passing by","Race car, auto racing","Truck","Air brake","Air horn, truck horn","Reversing beeps","Ice cream truck, ice cream van","Bus","Emergency vehicle","Police car (siren)","Ambulance (siren)","Fire engine, fire truck (siren)","Motorcycle","Traffic noise, roadway noise","Rail transport","Train","Train whistle","Train horn","Railroad car, train wagon","Train wheels squealing","Subway, metro, underground","Aircraft","Aircraft engine","Jet engine","Propeller, airscrew","Helicopter","Fixed-wing aircraft, airplane","Bicycle","Skateboard","Engine","Light engine (high frequency)","Dental drill, dentist's drill","Lawn mower","Chainsaw","Medium engine (mid frequency)","Heavy engine (low frequency)","Engine knocking","Engine starting","Idling","Accelerating, revving, vroom","Door","Doorbell","Ding-dong","Sliding door","Slam","Knock","Tap","Squeak","Cupboard open or close","Drawer open or close","Dishes, pots, and pans","Cutlery, silverware","Chopping (food)","Frying (food)","Microwave oven","Blender","Water tap, faucet","Sink (filling or washing)","Bathtub (filling or washing)","Hair dryer","Toilet flush","Toothbrush","Electric toothbrush","Vacuum cleaner","Zipper (clothing)","Keys jangling","Coin (dropping)","Scissors","Electric shaver, electric razor","Shuffling cards","Typing","Typewriter","Computer keyboard","Writing","Alarm","Telephone","Telephone bell ringing","Ringtone","Telephone dialing, DTMF","Dial tone","Busy signal","Alarm clock","Siren","Civil defense siren","Buzzer","Smoke detector, smoke alarm","Fire alarm","Foghorn","Whistle","Steam whistle","Mechanisms","Ratchet, pawl","Clock","Tick","Tick-tock","Gears","Pulleys","Sewing machine","Mechanical fan","Air conditioning","Cash register","Printer","Camera","Single-lens reflex camera","Tools","Hammer","Jackhammer","Sawing","Filing (rasp)","Sanding","Power tool","Drill","Explosion","Gunshot, gunfire","Machine gun","Fusillade","Artillery fire","Cap gun","Fireworks","Firecracker","Burst, pop","Eruption","Boom","Wood","Chop","Splinter","Crack","Glass","Chink, clink","Shatter","Liquid","Splash, splatter","Slosh","Squish","Drip","Pour","Trickle, dribble","Gush","Fill (with liquid)","Spray","Pump (liquid)","Stir","Boiling","Sonar","Arrow","Whoosh, swoosh, swish","Thump, thud","Thunk","Electronic tuner","Effects unit","Chorus effect","Basketball bounce","Bang","Slap, smack","Whack, thwack","Smash, crash","Breaking","Bouncing","Whip","Flap","Scratch","Scrape","Rub","Roll","Crushing","Crumpling, crinkling","Tearing","Beep, bleep","Ping","Ding","Clang","Squeal","Creak","Rustle","Whir","Clatter","Sizzle","Clicking","Clickety-clack","Rumble","Plop","Jingle, tinkle","Hum","Zing","Boing","Crunch","Silence","Sine wave","Harmonic","Chirp tone","Sound effect","Pulse","Inside, small room","Inside, large room or hall","Inside, public space","Outside, urban or manmade","Outside, rural or natural","Reverberation","Echo","Noise","Environmental noise","Static","Mains hum","Distortion","Sidetone","Cacophony","White noise","Pink noise","Throbbing","Vibration","Television","Radio","Field recording"]

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


    file_path2 = 'recorded_audio2.wav'
    output_filename2 = 'recorded_audio2'

    
    try:
        os.remove(file_path2)
        print("File", output_filename2, "deleted successfully.")
    except FileNotFoundError:
        print("File", output_filename2, "not found.")
    except Exception as e:
        print("An error occurred:", e)


    # start Recording
        
    frames2 = []

    CHUNK2       = 1024
    FORMAT2      = pyaudio.paInt16
    RATE2        = 44100
    CHANNELS2    = 1

    frames = []
    seconds = 5
    
    stream2 = audio.open(format=FORMAT2,
                        channels=CHANNELS2,
                        rate=RATE2,
                        input=True,
                        frames_per_buffer=CHUNK2)
    print("recording2...")

    for i in range(0, int(RATE2 / CHUNK2 * seconds)):
        data = stream2.read(CHUNK2)
        frames.append(data)

    
    print("recording2 stopped")
    stream2.stop_stream()
    stream2.close()
    p.terminate()

    output_filename2 = "recorded_audio2.wav"
    wf2 = wave.open(output_filename2, 'wb')
    wf2.setnchannels(1)
    wf2.setsampwidth(audio.get_sample_size(FORMAT2))
    wf2.setframerate(RATE2)
    wf2.writeframes(b''.join(frames))
    wf2.close()


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
    
    
        # Break the loop if no more audio data is available
    # if len(dataR) == 0:
    #         break

    
    



    if plt_classes is not None:
        plt_classes_lab = yamnet_classes[plt_classes]
        n_classes = len(plt_classes)
    else:
        plt_classes = [k for k in range(len(yamnet_classes))]
        plt_classes_lab = yamnet_classes if class_labels else None
        n_classes = len(yamnet_classes)

    monitor = Plotter(n_classes=n_classes, FIG_SIZE=(12,6), msd_labels=plt_classes_lab)
  

    transcript = []
   
    recordings_list = []
    label = ''
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # Waveform
        #data = preprocess_input(np.fromstring(
        dataR = stream.read(CHUNK)
        frames2.append(dataR)
        #wf.writeframes(dataR)
        data = preprocess_input(np.frombuffer(
            stream.read(CHUNK), dtype=np.float32), RATE)
        #print(model.predict(np.expand_dims(data,0))[0])
        #print(data)
        prediction = model.predict(np.expand_dims(data,0))[0]
        dflabel2 = pd.DataFrame(prediction,columns=["Count"])
        dflabel2.columns = dflabel2.columns.str.strip()
        highest_with_index2 = get_highest_with_index(dflabel2, 3)
        transcript.append([i,prediction])
        for j in range(2):
            label = classLabels[highest_with_index2[j][1]]
            recordings_list.append([i,label])
            print(label)
        recordings_list.append([i,label])
       




    print("finished recording")
    soundDf = pd.DataFrame(recordings_list, columns=['Time', 'SoundEvent'])
    print(f"The 10s recorded size is {soundDf.shape}")
    print(soundDf.head)
    transcriptDf = pd.DataFrame(transcript,columns=['Time', 'Score'])
    print(transcriptDf.head)
    
    
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
    wf.writeframes(b''.join(frames2))
    # #wf.writeframes(frames)
    # wf.setsampwidth(audio.get_sample_size(FORMAT))
    # wf.setframerate(RATE)
    wf.close()
    print("Recording finished. Audio saved as:", output_filename)
    
    highest_with_index = get_highest_with_index(dflabel, 10)
    #print(highest_with_index)
    # print(classLabels[highest_with_index[0][1]])
for i in range(10):
        print(classLabels[highest_with_index[i][1]])



 
    