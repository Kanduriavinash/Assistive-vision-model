Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 2
$synth.Speak('Testing speech. Caution, 2 persons detected ahead. Path is clear.')
Write-Host "Speech completed successfully"
