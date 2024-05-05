P=$HOME/sd10t/verse
for s in test training validation; do
    python complete_label.py $P/processed-verse19/$s $P/processed-verse19_ts-bone-label/$s
done
