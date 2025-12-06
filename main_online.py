from Class_perceptron import Perceptron
import numpy as np
import threading

# perceptron = Perceptron()
# x, y = perceptron.generate_LS(n=2, p=20)

# perceptron_to_verify = Perceptron()

# perceptron_to_verify.train_online(x, y)

def worker_online(i,X,y,eta,results,perceptron):
  p = Perceptron(eta=eta)
  p.train_online(X,y)
  w = np.array(p.w)
  perceptron_maitre = np.array(perceptron.perceptron_maitre).flatten() 
  R = np.dot(perceptron_maitre, w)
  recouvrement = R / (np.linalg.norm(perceptron_maitre) * np.linalg.norm(w)) 
  results[i]={
    "weights": w,
    "epochs": p.epoch,
    "recouvrement": recouvrement
  }
#  QUESTION 2
# def thread_perceptron():
#   threads=[]
#   p = Perceptron()
#   X,y = p.generate_LS(2,20)
#   results = [None] * 20
#   for i in range(20):
#     t= threading.Thread(target=worker_online, args=(i,X,y,0.01,results,p))
#     threads.append(t)
#     t.start()
#   for t in threads:
#     t.join()
#   for i in range(20):
#     print(f"Thread {i}: Weights: {results[i]['weights']}, Epochs: {results[i]['epochs']}, Recouvrement: {results[i]['recouvrement']}")  # type: ignore

# thread_perceptron()

# QUESTION 3 

def thread_perceptronv2(number_threads,N,P,eta):
  threads=[]
  results = [None] * number_threads
  for i in range(number_threads):
    p = Perceptron()
    X,y = p.generate_LS(N,P)
    t= threading.Thread(target=worker_online, args=(i,X,y,eta,results,p))
    threads.append(t)
    t.start()
  for t in threads:
    t.join()
  IT_moyen=0
  R_moyen=0  
  for i in range(number_threads):
    IT_moyen+=results[i]['epochs'] # type: ignore
    R_moyen+=results[i]['recouvrement'] # type: ignore
#   print(f"IT moyen: {IT_moyen/number_threads}, R moyen: {R_moyen/number_threads}")
  return IT_moyen/number_threads, R_moyen/number_threads

def main():
  eta0=0.1
  eta=[eta0,eta0/2,eta0/10]
  N=[2,10] #100,500,1000,5000
  P=[10] #,100,200,500,1000
  table_results=[]
  for e in eta:
    table_e=[]
    for N_i in N:
      for P_i in P:
        IT,R=thread_perceptronv2(100,N_i,P_i,e)
        table_e.append((N_i,P_i,IT,R))
    table_results.append((e,table_e))
  for result in table_results:
    e=result[0]
    print(f"=== Résultats pour eta={e} ===")
    for res in result[1]:
      N_i,P_i,IT,R=res
      print(f"N={N_i}, P={P_i} => IT moyen: {IT}, R moyen: {R}")  

main()