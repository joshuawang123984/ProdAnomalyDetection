import numpy as np
import torch

import src.constants as constants
from src.models import load_model

def main(): 
    model, _, _, _ = load_model()

    X_test = np.load(constants.TEST_WINDOWS_PATH)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    model.eval()
    reconstructed = model(X_test)
    print(reconstructed)

if __name__ == '__main__':
    main()