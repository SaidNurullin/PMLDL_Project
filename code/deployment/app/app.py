import streamlit as st
import os
import requests

uri = os.environ.get('API_URI')

st.title("AI Music Generation")

length = st.number_input("Song length(in seconds):", min_value=5, max_value=600, step=1, value=30)
instruments = st.multiselect("Music Instruments:", ['Piano right', 'Piano left'])

if not instruments:
    st.error("Please choose at least one music instrument!")
    st.stop() 

if st.button("Generate Music"):
    with st.spinner("Music generation..."):
        response = requests.post(
            f"http://{uri}:8000/generate",
            json={"length": length, "instrument": instruments},
        )
        if response.status_code == 200:
            with open("generated_music.mid", "wb") as f:
                f.write(response.content)

            st.success("Music has been successfully generated!")
            
            with open("generated_music.mid", "rb") as f:
                st.download_button(
                    label="Download MIDI file",
                    data=f,
                    file_name="generated_music.mid",
                    mime="audio/midi"
                )

        else:
            st.error("Error occurred when generating music")
