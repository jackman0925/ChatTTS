#!/usr/bin/env python3
import time
import wave
import base64
import os
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
import pydantic
import ChatTTS

app = FastAPI()


class TTSInput(pydantic.BaseModel):
    text: str
    output_path: str
    seed: int = 697

class TTS2Input(pydantic.BaseModel):
    text: str
    seed: int = 697
    promptspeed: str = '[speed_2]'  # 语速，从慢到快，共有10个等级[speed_0]~[speed_9]

def get_chat_model() -> ChatTTS.Chat:
    chat = ChatTTS.Chat()
    chat.load_models()
    return chat


@app.post("/tts")
def tts(input: TTSInput, chat: ChatTTS.Chat = Depends(get_chat_model)):
    try:
        texts = [input.text]
        r = chat.sample_random_speaker(seed=input.seed)

        params_infer_code = {
            'spk_emb': r,  # add sampled speaker
            'temperature': .3,  # using customtemperature
            'top_P': 0.7,  # top P decode
            'top_K': 20,  # top K decode
        }

        params_refine_text = {
            'prompt': '[oral_2][laugh_0][break_6]'
        }

        wavs = chat.infer(texts,
                          params_infer_code=params_infer_code,
                          params_refine_text=params_refine_text, use_decoder=True)

        audio_data = np.array(wavs[0], dtype=np.float32)
        sample_rate = 24000
        audio_data = (audio_data * 32767).astype(np.int16)

        with wave.open(input.output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        return {"output_path": input.output_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts-extend")
def tts_extend(input: TTS2Input, chat: ChatTTS.Chat = Depends(get_chat_model)):
    try:
        texts = [input.text]
        r = chat.sample_random_speaker(seed=input.seed)

        params_infer_code = {
            'prompt': input.promptspeed,  # '[speed_2]',  # 语速，从慢到快，共有10个等级[speed_0]~[speed_9]
            'spk_emb': r,  # add sampled speaker
            'temperature': .3,  # using customtemperature
            'top_P': 0.7,  # top P decode
            'top_K': 20,  # top K decode
        }

        params_refine_text = {
            'prompt': '[oral_2][laugh_0][break_6]'
        }

        wavs = chat.infer(texts,
                          params_infer_code=params_infer_code,
                          params_refine_text=params_refine_text, use_decoder=True)

        audio_data = np.array(wavs[0], dtype=np.float32)
        sample_rate = 24000
        audio_data = (audio_data * 32767).astype(np.int16)

        output_path = f"{int(time.time())}.wav"
        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        # 将 WAV 文件读取并转换为 base64 编码
        wav_base64 = ""
        with open(output_path, "rb") as wav_file:
            # 读取文件内容
            wav_data = wav_file.read()

            # 将音频数据转换为 base64 编码
            wav_base64 = base64.b64encode(wav_data).decode('utf-8')
            # 删除文件
            os.remove(output_path)

        return {"wav_base64": wav_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
