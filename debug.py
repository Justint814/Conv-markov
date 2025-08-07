import numpy as np
from datetime import datetime, timedelta
import time
import pytz


arr_1d = np.array([1, 2, 2, 3, 1, 4, 3, 3, 5])

arr_1d[0:5] = 6
print(arr_1d)