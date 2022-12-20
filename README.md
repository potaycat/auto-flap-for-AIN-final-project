## Gym environments with neural evolution

Using a small dense NN. Tested gyms with good results are:
 - CartPole-v1
 - FlappyBird-v0
 - Snake-v0


## Setup

*Using virtualenv is recomended*
```sh
python -m venv venv
./venv/bin/activate
```

Install dependancies
```sh
pip install --force-reinstall -r requirements.txt
```

To train the model:
`python ./gym_train.py`

To run the model:
`python ./gym_play.py`


## Report
[Report paper](https://www.overleaf.com/read/rhgfnwkbdtsv)
