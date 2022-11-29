import Pystudent
import numpy as np

student = Pystudent.Pystudent("YunYang1994", 28)
score = np.array([89, 59, 98]).astype(np.float32)
student.set_score(score, 3)
print("score: ",student.get_score())
print("age: ",  student.get_age())

