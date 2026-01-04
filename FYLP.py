import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import folium
from streamlit_folium import st_folium

# -URL
url1 = "https://raw.githubusercontent.com/Emz-oamk/FysiikanLP/refs/heads/main/Linear%20Acceleration.csv"
url2 = "https://raw.githubusercontent.com/Emz-oamk/FysiikanLP/refs/heads/main/Location.csv"

# -Data
acc = pd.read_csv(url1)
loc = pd.read_csv(url2)

# -Otsikko
st.title('Fysiikan Loppuprojekti')

# -Näytteenottotaajuus
a = acc['Linear Acceleration z (m/s^2)']
t = acc['Time (s)']
fs = 1 / np.mean(np.diff(t))

# -Suodattimet
def butter_lowpass_filter(data, cutoff, nyq, order=5):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, nyq, order=5):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y


# -Suodatus
st.subheader('Suodatettu kiihtyvyysdata | z-komponentti')

data = acc['Linear Acceleration z (m/s^2)']
T_tot = acc['Time (s)'].max()
n = len(acc['Time (s)'])
fs = n / T_tot
nyq = fs/2
order = 3
cutoff = 1/0.2
# Käytetään oikeaa nyquist-arvoa suodattimen kutsussa
data_filtered = butter_lowpass_filter(data, cutoff, nyq, order)

# -Piirretään suodatettu data (Streamlit: käytä st.pyplot)
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(acc['Time (s)'], data, label='data')
ax.plot(acc['Time (s)'], data_filtered, label='suodatettu data')
ax.set_ylabel("Kiihtyvyys (m/s²)")
ax.set_xlabel("Aika (s)")
ax.grid()
ax.legend()
st.pyplot(fig)

# -Askelmäärä laskettuna suodatetusta datasta
jaksot = 0
for i in range(n-1):
    if data_filtered[i]/data_filtered[i+1] < 0:
        jaksot = jaksot + 1/2

st.write("Askelmäärä suodatetusta datasta:", jaksot)


# -Fourier-analyysi
Y = np.fft.fft(data_filtered)
freq = np.fft.fftfreq(len(Y), d=1/fs)

# -Vain positiiviset taajuudet
mask = freq >= 0
freq = freq[mask]
power = np.abs(Y[mask])**2

# -Rajaa kävelytaajuudet
walk_mask = (freq >= 0.5) & (freq <= 3.0)
freq_walk = freq[walk_mask]
power_walk = power[walk_mask]

# -Askeltaajuus Fourier-analyysistä
step_freq = freq_walk[np.argmax(power_walk)]

# -Askelmäärä
step_fft = step_freq * T_tot #/ 2 ?

st.write("Askelmäärä Fourier-analyysistä:", round(step_fft, 2))


# -Keskinopeus
st.write("Keskinopeus:", loc['Velocity (m/s)'].mean(), "m/s" )

# -Kokonaismatka
dt = loc['Time (s)'].diff()
distance = (loc['Velocity (m/s)'] * dt).sum()
st.write("Kokonaismatka:", round(distance, 2), "m" )

# -Askelpituus
step_length = distance / jaksot
st.write("Askelpituus:", round(step_length, 2), "m" )


# -Tehospektrin laskeminen ja piirto
st.subheader('Tehospektri')

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(freq_walk, power_walk)
ax.set_xlabel("Taajuus (Hz)")
ax.set_ylabel("Teho")
ax.set_title("Kiihtyvyysdatan tehospektri")
ax.grid()
st.pyplot(fig)


# -Otsikko kartalle ja karttakuva
st.subheader('Karttakuva')

start_lat = loc['Latitude (°)'].mean()
start_lon = loc['Longitude (°)'].mean()
map = folium.Map(location=[start_lat, start_lon], zoom_start=14)

folium.PolyLine(loc[['Latitude (°)', 'Longitude (°)']], color='red', weight = 3, opacity = 1).add_to(map)

st_map = st_folium(map, width=700, height=500)

