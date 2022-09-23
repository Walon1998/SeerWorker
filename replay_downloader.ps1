for ($i=0; $i -le 49; $i=$i+1 ) {
    wget http://nevillewalo.ch/assets/Replays_$i.zip
    Expand-Archive -DestinationPath ".\Replays" .\Replays_$i.zip
}