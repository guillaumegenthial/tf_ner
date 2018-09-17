cd lstm_crf
pwd

python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results1
python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results2
python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results3
python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results4
python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results5

cd ..
cd chars_lstm_lstm_crf
pwd

python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results1
python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results2
python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results3
python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results4
python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results5

cd ..
cd chars_conv_lstm_crf
pwd

python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results1
python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results2
python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results3
python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results4
python main.py
../conlleval < results/score/train.preds.txt > results/score.train.metrics.txt
../conlleval < results/score/testa.preds.txt > results/score.testa.metrics.txt
../conlleval < results/score/testb.preds.txt > results/score.testb.metrics.txt
mv results results5

cd ..
