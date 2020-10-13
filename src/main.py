import camclay

clay = camclay.CamClay(e0=1.5,p0=100.e3)
clay.isotropic_compression(150.e3)

print(clay.e)
