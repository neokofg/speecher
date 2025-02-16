from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
from initialize_tts import tts    

# Временный кэш для copy_voice_path файлов
def vocal_text(to_language, text, copy_voice_path):
    if to_language == 'spanish':
        pipeline = KPipeline(lang_code='e') # <= make sure lang_code matches voice
        generator = pipeline(
        text, voice='ef_dora', # <= change voice here
        speed=1, split_pattern=r'\n+'
        )   
        for i, (gs, ps, audio) in enumerate(generator):
            print(i)  # i => index
            print(gs) # gs => graphemes/text
            print(ps) # ps => phonemes
            display(Audio(data=audio, rate=24000, autoplay=i==0))
            sf.write(f'{i}.wav', audio, 24000) # save each audio file
    elif to_language == 'american_english':
        pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice
        generator = pipeline(
        text, voice='af_heart', # <= change voice here
        speed=1, split_pattern=r'\n+'
        )   
        for i, (gs, ps, audio) in enumerate(generator):
            print(i)  # i => index
            print(gs) # gs => graphemes/text
            print(ps) # ps => phonemes
            display(Audio(data=audio, rate=24000, autoplay=i==0))
            sf.write(f'{i}.wav', audio, 24000) # save each audio file
    elif to_language == 'british_english':
        pipeline = KPipeline(lang_code='b') # <= make sure lang_code matches voice
        generator = pipeline(
        text, voice='bf_emma', # <= change voice here
        speed=1, split_pattern=r'\n+'
        )   
        for i, (gs, ps, audio) in enumerate(generator):
            print(i)  # i => index
            print(gs) # gs => graphemes/text
            print(ps) # ps => phonemes
            display(Audio(data=audio, rate=24000, autoplay=i==0))
            sf.write(f'{i}.wav', audio, 24000) # save each audio file
    elif to_language == 'japaneese':
        pipeline = KPipeline(lang_code='j') # <= make sure lang_code matches voice
        generator = pipeline(
        text, voice='jf_alpha', # <= change voice here
        speed=1, split_pattern=r'\n+'
        )   
        for i, (gs, ps, audio) in enumerate(generator):
            print(i)  # i => index
            print(gs) # gs => graphemes/text
            print(ps) # ps => phonemes
            display(Audio(data=audio, rate=24000, autoplay=i==0))
            sf.write(f'{i}.wav', audio, 24000) # save each audio file
    elif to_language == 'mandarin_chinese':
        pipeline = KPipeline(lang_code='z') # <= make sure lang_code matches voice
        generator = pipeline(
        text, voice='zf_xiaobei', # <= change voice here
        speed=1, split_pattern=r'\n+'
        )   
        for i, (gs, ps, audio) in enumerate(generator):
            print(i)  # i => index
            print(gs) # gs => graphemes/text
            print(ps) # ps => phonemes
            display(Audio(data=audio, rate=24000, autoplay=i==0))
            sf.write(f'{i}.wav', audio, 24000) # save each audio file
    elif to_language == 'french':
        pipeline = KPipeline(lang_code='f') # <= make sure lang_code matches voice
        generator = pipeline(
        text, voice='ff_siwis', # <= change voice here
        speed=1, split_pattern=r'\n+'
        )   
        for i, (gs, ps, audio) in enumerate(generator):
            print(i)  # i => index
            print(gs) # gs => graphemes/text
            print(ps) # ps => phonemes
            display(Audio(data=audio, rate=24000, autoplay=i==0))
            sf.write(f'{i}.wav', audio, 24000) # save each audio file
    elif to_language == 'hindi':
        pipeline = KPipeline(lang_code='h') # <= make sure lang_code matches voice
        generator = pipeline(
        text, voice='hf_alpha', # <= change voice here
        speed=1, split_pattern=r'\n+'
        )   
        for i, (gs, ps, audio) in enumerate(generator):
            print(i)  # i => index
            print(gs) # gs => graphemes/text
            print(ps) # ps => phonemes
            display(Audio(data=audio, rate=24000, autoplay=i==0))
            sf.write(f'{i}.wav', audio, 24000) # save each audio file
    elif to_language == 'italian':
        pipeline = KPipeline(lang_code='i') # <= make sure lang_code matches voice
        generator = pipeline(
        text, voice='if_sara', # <= change voice here
        speed=1, split_pattern=r'\n+'
        )   
        for i, (gs, ps, audio) in enumerate(generator):
            print(i)  # i => index
            print(gs) # gs => graphemes/text
            print(ps) # ps => phonemes
            display(Audio(data=audio, rate=24000, autoplay=i==0))
            sf.write(f'{i}.wav', audio, 24000) # save each audio file
    elif to_language == 'brazil/portuquese':
        pipeline = KPipeline(lang_code='p') # <= make sure lang_code matches voice
        generator = pipeline(
        text, voice='pf_dora', # <= change voice here
        speed=1, split_pattern=r'\n+'
        )   
        for i, (gs, ps, audio) in enumerate(generator):
            print(i)  # i => index
            print(gs) # gs => graphemes/text
            print(ps) # ps => phonemes
            display(Audio(data=audio, rate=24000, autoplay=i==0))
            sf.write(f'{i}.wav', audio, 24000) # save each audio file
    elif to_language == 'german':
        tts.tts_to_file(text=f"{text}",
                file_path="output.wav",
                speaker_wav=fr"{copy_voice_path}",
                language="de")
        file_path = "output.wav"
        return Audio(file_path, autoplay=True)
    elif to_language == 'polish':
        tts.tts_to_file(text=f"{text}",
                file_path="output.wav",
                speaker_wav=fr"{copy_voice_path}",
                language="pl")
        file_path = "output.wav"
        return Audio(file_path, autoplay=True)
    elif to_language == 'turkish':
        tts.tts_to_file(text=f"{text}",
                file_path="output.wav",
                speaker_wav=fr"{copy_voice_path}",
                language="tr")
        file_path = "output.wav"
        return Audio(file_path, autoplay=True)
    elif to_language == 'russian':
        tts.tts_to_file(text=f"{text}",
                file_path="output.wav",
                speaker_wav=fr"{copy_voice_path}",
                language="ru")
        file_path = "output.wav"
        return Audio(file_path, autoplay=True)
    elif to_language == 'dutch':
        tts.tts_to_file(text=f"{text}",
                file_path="output.wav",
                speaker_wav=fr"{copy_voice_path}",
                language="nl")
        file_path = "output.wav"
        return Audio(file_path, autoplay=True)
    elif to_language == 'czech':
        tts.tts_to_file(text=f"{text}",
                file_path="output.wav",
                speaker_wav=fr"{copy_voice_path}",
                language="cs")
        file_path = "output.wav"
        return Audio(file_path, autoplay=True)
    elif to_language == 'Arabic':
        tts.tts_to_file(text=f"{text}",
                file_path="output.wav",
                speaker_wav=fr"{copy_voice_path}",
                language="ar")
        file_path = "output.wav"
        return Audio(file_path, autoplay=True)
    elif to_language == 'chinese':
        tts.tts_to_file(text=f"{text}",
                file_path="output.wav",
                speaker_wav=fr"{copy_voice_path}",
                language="zh-cn")
        file_path = "output.wav"
        return Audio(file_path, autoplay=True)
    elif to_language == 'hungarian':
        tts.tts_to_file(text=f"{text}",
                file_path="output.wav",
                speaker_wav=fr"{copy_voice_path}",
                language="hu")
        file_path = "output.wav"
        return Audio(file_path, autoplay=True)
    elif to_language == 'korean':
        tts.tts_to_file(text=f"{text}",
                file_path="output.wav",
                speaker_wav=fr"{copy_voice_path}",
                language="ko")
        file_path = "output.wav"
        return Audio(file_path, autoplay=True)