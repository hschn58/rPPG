
######################################################################

#frame parameters

TOT_FRAMES=900
FRAME_WIDTH=300
FRAME_HEIGHT=FRAME_WIDTH
FRAME_INTERVAL=15
SIZE=80 #was 20
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

#sampling_rate= 19.802 # 95.2381
lowcut_heart = 0.5
highcut_heart = 4
lowcut_breath = 0.3
highcut_breath = 1.0

graph_height=400
graph_width=600

######################################################################
#bbox:     _, faces = detector.detect(frame) 
#faces[0][:4]:
# 1: x-value of top left corner
# 2: y-value of top left corner
# 3: width of bounding box
# 4: height of bounding box


#its frame[rows, columns]
def left_cheek(frame, bbox, DELTA=DELTA, CHEEK_HPAD=CHEEK_HPAD, CHEEK_VPAD=CHEEK_VPAD):
    return frame[(bbox[1] + int(bbox[3] * CHEEK_VPAD) - DELTA):(bbox[1] + int(bbox[3] * CHEEK_VPAD) + DELTA),
                 (bbox[0] + int(bbox[2] * CHEEK_HPAD) - DELTA):(bbox[0] + int(bbox[2] * CHEEK_HPAD) + DELTA)]


def right_cheek(frame, bbox, DELTA=DELTA, CHEEK_HPAD=CHEEK_HPAD, CHEEK_VPAD=CHEEK_VPAD):
    return frame[(bbox[1] + int(bbox[3] * CHEEK_VPAD) - DELTA):(bbox[1] + int(bbox[3] * CHEEK_VPAD) + DELTA),
                 (bbox[0] + int(bbox[2] * (1 - CHEEK_HPAD)) - DELTA):(bbox[0] + int(bbox[2] * (1 - CHEEK_HPAD)) + DELTA)]


def chin(frame, bbox, DELTA=DELTA, CHIN_HPAD=CHIN_HPAD, CHIN_VPAD=CHIN_VPAD):
    return frame[(bbox[1] + int(bbox[3] * CHIN_VPAD) - DELTA):(bbox[1] + int(bbox[3] * CHIN_VPAD) + DELTA),
                 (bbox[0] + int(bbox[2] * CHIN_HPAD) - DELTA):(bbox[0] + int(bbox[2] * CHIN_HPAD) + DELTA)]


def midleft_cheek(frame, bbox, DELTA=DELTA, MLCHEEK_HPAD=MLCHEEK_HPAD, MLCHEEK_VPAD=MLCHEEK_VPAD):
    return frame[(bbox[1] + int(bbox[3] * MLCHEEK_VPAD) - DELTA):(bbox[1] + int(bbox[3] * MLCHEEK_VPAD) + DELTA),
                 (bbox[0] + int(bbox[2] * MLCHEEK_HPAD) - DELTA):(bbox[0] + int(bbox[2] * MLCHEEK_HPAD) + DELTA)]


def midright_cheek(frame, bbox, DELTA=DELTA, MRCHEEK_HPAD=MRCHEEK_HPAD, MRCHEEK_VPAD=MRCHEEK_VPAD):
    return frame[(bbox[1] + int(bbox[3] * MRCHEEK_VPAD) - DELTA):(bbox[1] + int(bbox[3] * MRCHEEK_VPAD) + DELTA),
                 (bbox[0] + int(bbox[2] * MRCHEEK_HPAD) - DELTA):(bbox[0] + int(bbox[2] * MRCHEEK_HPAD) + DELTA)]
