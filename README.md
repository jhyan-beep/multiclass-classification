# Multiclass Classification with Logistic Regression and a Neural Network

This project compares a multinomial logistic regression baseline with a small
feedforward neural network implemented from scratch in NumPy. The experiment
uses a reproducible two-feature, three-class synthetic dataset from
`sklearn.datasets.make_classification`, which keeps the decision boundaries easy
to visualize while still showing the effect of nonlinear modeling.

## Project Highlights

- Includes a polished static dashboard (`index.html`) for portfolio and GitHub
  Pages presentation.
- Builds a reproducible multiclass classification dataset with a fixed random seed.
- Trains a multinomial logistic regression model as a baseline.
- Uses an explicit 10-fold cross-validation setting for the logistic regression
  baseline so the original validation accuracy is reproducible across newer
  scikit-learn versions.
- Implements a 3-layer neural network from scratch using NumPy, including:
  - ReLU, sigmoid, tanh, and softmax activations
  - Cross-entropy loss with L2 weight decay
  - Backpropagation and batch gradient descent
  - Numerical gradient checking
  - Dropout regularization
- Visualizes validation-set decision boundaries for the baseline and neural network models.

## Results

These are the real validation results preserved from the notebook run. They are
not inflated or converted into unrelated performance claims.

| Model | Validation accuracy |
| --- | ---: |
| Multinomial logistic regression | 0.696970 |
| Neural network, 3 hidden units | 0.727273 |
| Neural network, 200 hidden units | 0.727273 |
| Neural network, 200 hidden units with dropout | 0.757576 |

Gradient checks in the notebook also produced low mean absolute error:

| Check | Mean absolute error |
| --- | ---: |
| Quadratic function gradient check | 9.379164112033e-11 |
| Neural network backpropagation check | 8.971588121880e-11 |

## Repository Contents

```text
.
├── index.html
├── styles.css
├── app.js
├── model-data.js
├── README.md
├── requirements.txt
├── multiclass_logistic_regression_neural_network.ipynb
└── notebook_graphics/
    ├── loss_landscape.png
    └── nn-diagram.png
```

## Interactive UI

Open `index.html` in a browser to view the dashboard. The UI is built with
plain HTML, CSS, and JavaScript, with no external CDN, font, or image
dependencies. It uses exported model parameters from the notebook to render
decision surfaces and live point predictions in the browser.

For GitHub Pages, publish the repository from the `main` branch and root
directory. The dashboard will load as the project homepage.

## How to Run

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook multiclass_logistic_regression_neural_network.ipynb
```

Run the cells from top to bottom. The notebook uses fixed random seeds for the
dataset split, logistic regression baseline, and neural network initialization.
The logistic regression cell also sets `cv=10` explicitly to avoid changes in
library defaults.

## Notes

The notebook was originally developed as a machine learning assignment and has
been cleaned for portfolio presentation. The code cells now include the missing
implementation pieces needed for reproducible execution while preserving the
original experiment design and reported validation results.
