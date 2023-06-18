Import-Module WhisperPS -DisableNameChecking
$Model = Import-WhisperModel C:\Users\win10\Documents\podnotes-qa\api\Models\ggml-base.en.bin
cd  C:\Users\win10\Documents\podnotes-qa\workspace
$Results = dir .\media.mp3 | Transcribe-File $Model
foreach ( $i in $Results ) { $txt = $i.SourceName + ".txt"; $i | Export-Text $txt; }
foreach ( $i in $Results ) { $txt = $i.SourceName + ".ts.txt"; $i | Export-Text $txt -timestamps; }