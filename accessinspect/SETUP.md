# AccessInspect voice + Gemini defect check: setup

## Files
- `speak_alert.py`: spoken announcements (Ava neural voice), background playback
- `voice_cache/`: pre-made audio for all 83 phrases (works offline)
- `gemini_defect.py`: Gemini vision check, none | tape | sticker | marker
- `test_voice_loop.py`: checks the voice never blocks the robot loop
- `alert.mp3`: sample alert

## Install (new laptop)
```
pip install -r requirements.txt
```
Linux only, for audio playback: `sudo apt install mpg123`

## API key
Create a file named `.env` in this folder containing:
```
GEMINI_API_KEY=your-key-here
```
It is git-ignored. Do not commit or share it.

## Try it
```
python speak_alert.py start
python speak_alert.py reject tape
python speak_alert.py complete 1 1
python test_voice_loop.py
python gemini_defect.py brick.jpg --speak
```
If you edit phrases in `speak_alert.py`, run `python speak_alert.py --pregen` (needs internet once).
