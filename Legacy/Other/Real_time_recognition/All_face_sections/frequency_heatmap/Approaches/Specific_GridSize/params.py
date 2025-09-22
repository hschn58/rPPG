
######################################################################

#frame parameters

TOT_FRAMES=900
FRAME_WIDTH=300
FRAME_HEIGHT=FRAME_WIDTH
FRAME_INTERVAL=15
SIZE=20
DELTA=SIZE//2
# #face detection parameters
# RADIUS=3
# COLOR=(0,0,255) #green
# THICKNESS=-1 #filled circle
# #cheek detection parameters-distance from vertical edge of detection:
CHEEK_HPAD=1/6
# #cheek detection parameters-fractional distance from top edge of detection:
CHEEK_VPAD=7/12
# #chin location-horizontal
CHIN_HPAD=1/2
# #chin location-vertical (fraction from the top)
CHIN_VPAD=7/8
# #middle left cheek detection parameters-distance from vertical edge of detection:
MLCHEEK_HPAD=1/5
MRCHEEK_HPAD=1-MLCHEEK_HPAD
# #middle right cheek detection parameters-fractional distance from top edge of detection:
MRCHEEK_VPAD=3/4
MLCHEEK_VPAD=MRCHEEK_VPAD
######################################################################

#signal processing parameters

sampling_rate= 19.802 # 95.2381
lowcut_heart = 1.0
highcut_heart = 2.0
lowcut_breath = 0.3
highcut_breath = 1.0

graph_height=400
graph_width=600


#data visualization parameters(how many arrays x,y to split up each identified face into)
X_SIZE=48
Y_SIZE=48



