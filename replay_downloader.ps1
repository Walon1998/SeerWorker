for ($i=0; $i -le 49; $i=$i+1 ) {
    Invoke-WebRequest http://nevillewalo.ch/assets/Replays_$i.zip -OutFile Replays_$i.zip
    Expand-Archive -DestinationPath ".\Replays" Replays_$i.zip
}