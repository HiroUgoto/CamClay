import camclay

clay = camclay.CamClay(e0=1.5,p0=100.e3)
# clay.isotropic_compression(50.e3)
clay.triaxial_compression(100.e3,print_result=True,plot=True)
