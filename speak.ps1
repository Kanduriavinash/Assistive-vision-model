# speak.ps1 — Helper script for assistive vision TTS
# Called by live_cam.py via subprocess
# Usage: powershell -File speak.ps1 "text to speak"
param([string]$text)
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 2
$synth.Speak($text)
