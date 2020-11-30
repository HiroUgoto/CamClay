import time
import camclay

# start = time.time()

clay = camclay.CamClay(e0=1.5,p0=100.e3)
clay.simple_shear(100.e3,gamma_max=0.2,print_result=True,plot=True)

# elapsed_time = time.time() - start
# print ("elapsed_time: {0}".format(elapsed_time) + "[sec]")
