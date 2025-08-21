#For Our Beginners:

Welcome to the Neural Encoding Project! If you're reading this, it means you're embarking on a journey to build in a field that may have seemed like science fiction not long ago: **Brain-Computer Interfaces**.
In this readme, we will go over a number of topics to get you caught up, so that you can start building.

#Be careful:

**In case of emergency, turn off and remove the device. Please double check to make sure device is turned off before removing. If you are allergic to silver, do NOT wear the device. If you are allergic to rubbing/medical alcohol, please do NOT use the device. Do not place FNIRS or red light sensors near eye. If, while using the device, you feel any discomfort (skin irriation, anxiety, or other), turn off and remove the device. Do NOT pressure ANYONE to keep wearing the device on for ANY reason.**

batting 100 for now team, keep up the good standards in safety and ethics!

#The Device:

We will using a device named the Muse S Athena, of which we have two. The devices, which are built as fabric headbands, are worn around the head to collect signals from within the brain. 
There are no chips that go **in** the brain in our lab. Instead, we rely on non-invasive devices, which sit on the skin rather than in the brain itself.

Each device contains the following sensors:

EEG: The primary sensor used. EEG (or ElectroEncephaloGrams) are sensors which measure the small sparks of electricity our brains produce when groups of neurons fire. The data collected is typically measured in Micro Volts.

FNIRS: Another highly important sensor. FNIRS (or Functional Near InfraRed Systems) are sensors which shine near infrared light into the brain, and measure the amount of light reflected back.
The light is designed carefully, so that when the oxygen in the brain is high, more light is absorbed. When oxygen is lower, more light is reflected back to the device.

Because neurons consume oxygen and depending on how active they are, areas of the brain with higher blood oxygen levels have more brain activity. Bloodflow to areas of the brain with high utilization also accelerates, this is called hyperemia.

The sensors data is the amount of light reflected back to the device.

When we build our models, our goal is to collect clean data we can use to train our AI models to correlate the data we collect to a tangible effect (like pressing a key on or keyboard, or moving our mouse).

Easier said than done. The data we collect is VERY noisy, and very little in quantity, but with patience comes the satisfying award of a working project!

#The Data:

The Muse S Athena, which we will be using, has 4 EEG sensors (or electrodes). The data you will recieve is the raw microvolt data of the EEG. We will use fast fourier transforms to determine frequency bands.

There are also 4 FNIRS sensors (Optodes), each with 730nm and 850nm wavelengths.
The reason for this is that at 850nm, the device detects mostly from the number of blood cells which are rich in oxygen (Oxyhemoglobin, or HbO2), and at 730nm it detects mostly from blood cells which have alreadu given the brain its oxygen (Deoxyhemoglobin, or HbR).

The device also has a red light sensor (660nm), which also is good at capturing oxygen-rich blood cells, and is used for measuring Heart-rate Variability, and the expansion/contraction of blood vessels at each heartbeat.

The device contains Accelerometer and Gyroscopic data, so feel free to use these.

On top of all this, we have built in tracking for artifact detections such as jaw clenching and blinkinging, which we would otherwise need to filter out from our data by hand.

#The AI:

For the time being, we will be using the python library 'TensorFlow' to build the AI models required to classify our eeg data. 

For more information, I recommend reading about TensorFlow and Keras here: https://keras.io/
You can find examples of how this is used for EEG data here: https://keras.io/examples/timeseries/eeg_signal_classification/
