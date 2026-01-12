
import os
import time
import uuid
import gradio as gr

from the_doctor import encode_image, analyze_image_with_query
from patient import transcribe_with_groq
from doctor import text_to_speech


system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose.
What's in this image?. Do you find anything wrong with it medically?
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot,
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""


def process_inputs(audio_filepath, image_filepath):
    try:
        # ---------- TOTAL TIMER ----------
        start_total = time.perf_counter()

        # ---------- STT ----------
        start_stt = time.perf_counter()
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )
        if not speech_to_text_output:
            speech_to_text_output = "Could not understand the audio clearly."
        stt_latency = time.perf_counter() - start_stt

        # ---------- IMAGE + LLM ----------
        start_image = time.perf_counter()
        if image_filepath:
            doctor_response = analyze_image_with_query(
                query=system_prompt + " " + speech_to_text_output,
                encoded_image=encode_image(image_filepath),
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        else:
            doctor_response = "With what I understand from your symptoms, I need an image to analyze further."
        if not doctor_response:
            doctor_response = "I am unable to give a confident assessment at the moment."
        image_latency = time.perf_counter() - start_image

        # ---------- TTS ----------
        start_tts = time.perf_counter()
        audio_path = f"doctor_reply_{uuid.uuid4().hex}.mp3"
        text_to_speech(
            input_text=doctor_response,
            output_filepath=audio_path
        )
        tts_latency = time.perf_counter() - start_tts

        # ---------- TOTAL ----------
        total_latency = time.perf_counter() - start_total

        # ---------- FORMAT METRICS ----------
        stt_s = f"{stt_latency:.2f} sec"
        img_s = f"{image_latency:.2f} sec"
        tts_s = f"{tts_latency:.2f} sec"
        total_s = f"{total_latency:.2f} sec"

        # Terminal log for debugging / demo
        print("----- METRICS -----")
        print(f"STT latency        : {stt_s}")
        print(f"Image inference    : {img_s}")
        print(f"TTS latency        : {tts_s}")
        print(f"End-to-end latency : {total_s}")
        print("-------------------")

        return (
            speech_to_text_output,
            doctor_response,  # only show doctor's response, no latency in UI
            audio_path
        )

    except Exception as e:
        print("❌ Error:", e)
        return (
            "Error processing audio",
            "Something went wrong. Please try again.",
            None
        )


# ---------- GRADIO INTERFACE ----------
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak"),
        gr.Image(type="filepath", label="🖼 Upload Medical Image (optional)")
    ],
    outputs=[
        gr.Textbox(label="🗣 Speech to Text", lines=3),
        gr.Textbox(label="🧠 Doctor's Response", lines=4),
        gr.Audio(type="filepath", label="🔊 Doctor's Voice")
    ],
    title="AI Doctor Assistant"
)


if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        debug=False
    )