
Arc=('GRU' 'LSTM')
LearningCurr=('DeCu' 'VoCu' 'NaCu')
Instance=('0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10')
Mode=('Control' 'SlowPoints' 'CenterMass' 'NoReg')


for k in {0..9}
do
    for j in {0..1}
    do
        for i in {0..1}
        do
            python3 TriggerMNISTCloud.py ${Arc[i]} ${LearningCurr[j]} ${Instance[k]}
            python3 BlackBoxMNIST.py ${Arc[i]} ${LearningCurr[j]} ${Instance[k]}
        done
    done
done

for q in {0..2}
do
    for k in {0..9}
    do
        for j in {0..1}
        do
            for i in {0..1}
            do
                python3 TriggerSpeedCont.py ${Arc[i]} ${LearningCurr[j]} ${Instance[k]} ${Mode[q]}
            done
        done
    done
done

for q in {0..3}
do
    for k in {0..9}
    do
        for j in {0..1}
        do
            for i in {0..1}
            do
                python3 AccMNIST.py ${Arc[i]} ${LearningCurr[j]} ${Instance[k]} ${Mode[q]}
            done
        done
    done
done
